"""
Stage2 tree portfolio with strictly cleaned open-data feature groups.

This is the first controlled challenger after the baseline replay: same 5-day
target, same validation-shaped portfolio layer, but with optional open-data
features produced by stage2_open_data_features.py.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "archive" / "legacy_scripts"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from baseline_xgboost import EMBARGO_DAYS, FEATURE_COLUMNS, FORWARD_HORIZON, TARGET_COLUMN, build_features, prediction_frame
from score_submission import score_window
from stage2_open_data_features import add_stage2_open_features
from tuned_xgboost_portfolio import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TOPK_GRID,
    DEFAULT_VAL_DAYS,
    DEFAULT_POWER_GRID,
    PortfolioShape,
    build_shaped_portfolio,
    split_train_val,
)

DATA_DIR = ROOT / "data"


@dataclass
class AverageModel:
    models: list

    def predict(self, x):
        preds = [np.asarray(model.predict(x), dtype=float) for model in self.models]
        # Rank-normalize before averaging so model scale cannot dominate.
        ranked = [pd.Series(pred).rank(pct=True).to_numpy() for pred in preds]
        return np.mean(ranked, axis=0)


def _train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=700,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_alpha=0.05,
        reg_lambda=3.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=40,
    )
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def _train_lgb(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="huber",
        learning_rate=0.03,
        n_estimators=900,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def train_model(model_type: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    if model_type == "xgb":
        return _train_xgb(train_df, val_df, feature_cols)
    if model_type == "lightgbm":
        return _train_lgb(train_df, val_df, feature_cols)
    if model_type == "blend":
        return AverageModel([
            _train_xgb(train_df, val_df, feature_cols),
            _train_lgb(train_df, val_df, feature_cols),
        ])
    raise ValueError(f"Unknown model_type={model_type}")


def validation_windows(
    val_dates: np.ndarray,
    trading_dates: np.ndarray,
    horizon: int = FORWARD_HORIZON,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for offset in range(0, len(val_dates), horizon):
        as_of = pd.Timestamp(val_dates[offset])
        idx = date_to_idx.get(as_of)
        if idx is None or idx + horizon >= len(trading_dates):
            continue
        windows.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + horizon])))
    return windows


def select_shape(
    model,
    panel: pd.DataFrame,
    feature_cols: list[str],
    val_dates: np.ndarray,
    trading_dates: np.ndarray,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    shape_horizon: int = FORWARD_HORIZON,
) -> tuple[PortfolioShape, pd.DataFrame]:
    windows = validation_windows(val_dates, trading_dates, horizon=shape_horizon)
    if not windows:
        return PortfolioShape(top_k=50, rank_power=0.75), pd.DataFrame()

    score_cache = {}
    for as_of, _, _ in windows:
        pred = prediction_frame(panel, as_of=as_of).dropna(subset=feature_cols).copy()
        pred["score"] = model.predict(pred[feature_cols])
        score_cache[as_of] = pred.set_index("stock_code")["score"]

    rows = []
    for top_k in DEFAULT_TOPK_GRID:
        for rank_power in DEFAULT_POWER_GRID:
            scores = []
            shape = PortfolioShape(top_k=top_k, rank_power=rank_power)
            for as_of, start, end in windows:
                weights = build_shaped_portfolio(score_cache[as_of], shape)
                result = score_window(weights, prices, index_df, start, end)
                scores.append(result["excess_return"])
            rows.append({
                "top_k": top_k,
                "rank_power": rank_power,
                "mean_excess_return": float(np.mean(scores)),
                "sum_excess_return": float(np.sum(scores)),
                "min_excess_return": float(np.min(scores)),
                "std_excess_return": float(np.std(scores)),
                "utility_score": float(np.mean(scores) + np.min(scores) - 0.35 * np.std(scores)),
            })
    table = pd.DataFrame(rows).sort_values(
        ["utility_score", "mean_excess_return", "min_excess_return"],
        ascending=False,
    )
    best = table.iloc[0]
    return PortfolioShape(int(best["top_k"]), float(best["rank_power"])), table


def fit_open_tree_model(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    model_type: str,
    groups: list[str],
    open_dir: str | Path,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    corr_threshold: float = 0.92,
    min_coverage: float = 0.55,
    shape_horizon: int = FORWARD_HORIZON,
):
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    if lookback_days is not None and lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()

    panel = build_features(prices, index_df)
    panel, open_cols, feature_report = add_stage2_open_features(
        panel,
        open_dir=open_dir,
        groups=groups,
        corr_threshold=corr_threshold,
        min_coverage=min_coverage,
    )
    feature_cols = list(FEATURE_COLUMNS) + open_cols
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    train_pool = panel.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    train_pool = train_pool[train_pool["date"] <= train_cutoff].copy()
    train_df, val_df, train_end, val_start = split_train_val(
        train_pool,
        val_days=val_days,
        embargo_days=EMBARGO_DAYS,
    )
    model = train_model(model_type, train_df, val_df, feature_cols)
    shape, shape_table = select_shape(
        model,
        panel,
        feature_cols,
        np.sort(val_df["date"].unique()),
        trading_dates,
        prices,
        index_df,
        shape_horizon=shape_horizon,
    )
    return {
        "panel": panel,
        "feature_cols": feature_cols,
        "open_cols": open_cols,
        "feature_report": feature_report,
        "trading_dates": trading_dates,
        "train_df": train_df,
        "val_df": val_df,
        "train_end": train_end,
        "val_start": val_start,
        "model": model,
        "shape": shape,
        "shape_table": shape_table,
    }


def generate_submission(fit_result: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    pred = prediction_frame(fit_result["panel"], as_of=as_of).dropna(subset=fit_result["feature_cols"]).copy()
    pred["score"] = fit_result["model"].predict(pred[fit_result["feature_cols"]])
    weights = build_shaped_portfolio(pred.set_index("stock_code")["score"], fit_result["shape"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(ROOT / "archive" / "data_unused" / "open"))
    parser.add_argument("--groups", nargs="+", default=["valuation"], choices=["valuation", "market_regime", "fund_flow", "all"])
    parser.add_argument("--model-type", default="xgb", choices=["xgb", "lightgbm", "blend"])
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/experiments/stage2_open_tree.csv")
    parser.add_argument("--shape-out", default=None)
    parser.add_argument("--feature-report-out", default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--min-coverage", type=float, default=0.55)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = fit_open_tree_model(
        prices,
        index_df,
        as_of,
        model_type=args.model_type,
        groups=args.groups,
        open_dir=args.open_dir,
        lookback_days=args.lookback_days,
        val_days=args.val_days,
        corr_threshold=args.corr_threshold,
        min_coverage=args.min_coverage,
    )
    sub = generate_submission(fit, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.shape_out:
        shape_path = Path(args.shape_out)
        shape_path.parent.mkdir(parents=True, exist_ok=True)
        fit["shape_table"].to_csv(shape_path, index=False)
    if args.feature_report_out:
        report_path = Path(args.feature_report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fit["feature_report"].to_csv(report_path, index=False)

    print(f">> model={args.model_type} groups={','.join(args.groups)} as_of={as_of.date()}")
    print(f">> open_features={len(fit['open_cols'])}: {', '.join(fit['open_cols'])}")
    print(f">> train: {len(fit['train_df']):,} rows up to {fit['train_end'].date()}")
    print(f">> val:   {len(fit['val_df']):,} rows from {fit['val_start'].date()}")
    print(f">> selected shape: top_k={fit['shape'].top_k}, rank_power={fit['shape'].rank_power:.2f}")
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
