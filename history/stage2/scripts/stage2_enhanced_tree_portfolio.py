"""
Stage2 enhanced stock-feature tree portfolio.

This route uses only the original competition OHLCV/index data.  It adds
stock-level enhanced features, selects them on the train split only, and keeps
the same validation-shaped portfolio construction used by the tuned tree routes.
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
from stage2_stock_features import add_stage2_stock_features, select_stage2_stock_features
from tuned_xgboost_portfolio import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_POWER_GRID,
    DEFAULT_TOPK_GRID,
    DEFAULT_VAL_DAYS,
    PortfolioShape,
    build_shaped_portfolio,
    split_train_val,
)

DATA_DIR = ROOT / "data"


@dataclass
class RankAverageModel:
    models: list

    def predict(self, x):
        preds = [np.asarray(model.predict(x), dtype=float) for model in self.models]
        ranks = [pd.Series(pred).rank(pct=True).to_numpy() for pred in preds]
        return np.mean(ranks, axis=0)


def _train_lgb(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="huber",
        learning_rate=0.028,
        n_estimators=1000,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=70,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.08,
        reg_lambda=3.5,
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


def _train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=800,
        max_depth=4,
        learning_rate=0.032,
        subsample=0.86,
        colsample_bytree=0.82,
        min_child_weight=20,
        reg_alpha=0.08,
        reg_lambda=3.5,
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


def train_model(model_type: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    if model_type == "lightgbm":
        return _train_lgb(train_df, val_df, feature_cols)
    if model_type == "xgb":
        return _train_xgb(train_df, val_df, feature_cols)
    if model_type == "blend":
        return RankAverageModel([_train_lgb(train_df, val_df, feature_cols), _train_xgb(train_df, val_df, feature_cols)])
    raise ValueError(f"Unknown model_type={model_type}")


def validation_windows(val_dates: np.ndarray, trading_dates: np.ndarray, horizon: int = FORWARD_HORIZON):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    out = []
    for offset in range(0, len(val_dates), horizon):
        as_of = pd.Timestamp(val_dates[offset])
        idx = date_to_idx.get(as_of)
        if idx is None or idx + horizon >= len(trading_dates):
            continue
        out.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + horizon])))
    return out


def select_shape(model, panel, feature_cols, val_dates, trading_dates, prices, index_df):
    windows = validation_windows(val_dates, trading_dates)
    if not windows:
        return PortfolioShape(50, 0.75), pd.DataFrame()
    score_cache = {}
    for as_of, _, _ in windows:
        pred = prediction_frame(panel, as_of=as_of).dropna(subset=feature_cols).copy()
        pred["score"] = model.predict(pred[feature_cols])
        score_cache[as_of] = pred.set_index("stock_code")["score"]

    rows = []
    for top_k in DEFAULT_TOPK_GRID:
        for rank_power in DEFAULT_POWER_GRID:
            shape = PortfolioShape(top_k, rank_power)
            scores = []
            for as_of, start, end in windows:
                weights = build_shaped_portfolio(score_cache[as_of], shape)
                scores.append(score_window(weights, prices, index_df, start, end)["excess_return"])
            mean_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            std_score = float(np.std(scores))
            rows.append({
                "top_k": top_k,
                "rank_power": rank_power,
                "mean_excess_return": mean_score,
                "sum_excess_return": float(np.sum(scores)),
                "min_excess_return": min_score,
                "std_excess_return": std_score,
                "utility_score": mean_score + min_score - 0.35 * std_score,
            })
    table = pd.DataFrame(rows).sort_values(["utility_score", "mean_excess_return", "min_excess_return"], ascending=False)
    best = table.iloc[0]
    return PortfolioShape(int(best["top_k"]), float(best["rank_power"])), table


def fit_enhanced_tree(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    model_type: str = "lightgbm",
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    max_enhanced_features: int = 24,
    min_abs_ic: float = 0.003,
    corr_threshold: float = 0.90,
):
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    if lookback_days is not None and lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()

    panel, enhanced_candidates = add_stage2_stock_features(build_features(prices, index_df))
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    base_train_cols = list(FEATURE_COLUMNS)
    train_pool = panel.dropna(subset=base_train_cols + [TARGET_COLUMN]).copy()
    train_pool = train_pool[train_pool["date"] <= train_cutoff].copy()
    train_df, val_df, train_end, val_start = split_train_val(train_pool, val_days=val_days, embargo_days=EMBARGO_DAYS)

    selection = select_stage2_stock_features(
        train_df,
        enhanced_candidates,
        max_features=max_enhanced_features,
        min_abs_ic=min_abs_ic,
        corr_threshold=corr_threshold,
    )
    feature_cols = base_train_cols + selection.selected
    train_df = train_df.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    val_df = val_df.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    model = train_model(model_type, train_df, val_df, feature_cols)
    shape, shape_table = select_shape(
        model,
        panel,
        feature_cols,
        np.sort(val_df["date"].unique()),
        trading_dates,
        prices,
        index_df,
    )
    return {
        "panel": panel,
        "feature_cols": feature_cols,
        "enhanced_cols": selection.selected,
        "feature_report": selection.report,
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
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--model-type", choices=["lightgbm", "xgb", "blend"], default="lightgbm")
    parser.add_argument("--out", default="submissions/stage2/experiments/stage2_enhanced_tree.csv")
    parser.add_argument("--shape-out", default=None)
    parser.add_argument("--feature-report-out", default=None)
    parser.add_argument("--max-enhanced-features", type=int, default=24)
    parser.add_argument("--min-abs-ic", type=float, default=0.003)
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])
    fit = fit_enhanced_tree(
        prices,
        index_df,
        as_of,
        model_type=args.model_type,
        max_enhanced_features=args.max_enhanced_features,
        min_abs_ic=args.min_abs_ic,
        corr_threshold=args.corr_threshold,
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
    print(f">> model={args.model_type} as_of={as_of.date()}")
    print(f">> enhanced_features={len(fit['enhanced_cols'])}: {', '.join(fit['enhanced_cols'])}")
    print(f">> train: {len(fit['train_df']):,} rows up to {fit['train_end'].date()}")
    print(f">> val:   {len(fit['val_df']):,} rows from {fit['val_start'].date()}")
    print(f">> selected shape: top_k={fit['shape'].top_k}, rank_power={fit['shape'].rank_power:.2f}")
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
