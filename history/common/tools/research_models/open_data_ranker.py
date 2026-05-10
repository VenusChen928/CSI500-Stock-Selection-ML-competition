"""
Open-data boosted ranker for the CSI500 portfolio task.

This challenger keeps the baseline's robust portfolio layer, but changes the
supervised task from "predict raw future return" to "rank stocks by future
5-day excess return".  It also merges optional open datasets from
download_open_data.py (QVIX, broad PB regime, per-stock valuation/size).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_xgboost import EMBARGO_DAYS, MAX_WEIGHT, MIN_STOCKS
from features import (
    CANDIDATE_FEATURE_GROUPS,
    CORE_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
)
from open_data_features import add_open_data_features
from score_submission import score_window

DATA_DIR = ROOT / "data"
DEFAULT_VAL_DAYS = 15
DEFAULT_LOOKBACK_DAYS = 475
DEFAULT_TOPK_GRID = (35, 40, 45, 50, 60, 70, 80, 100)
DEFAULT_POWER_GRID = (0.35, 0.50, 0.75, 1.00, 1.25, 1.50)

EXTRA_TECHNICAL_FEATURES = (
    CANDIDATE_FEATURE_GROUPS["short_reversal"]
    + CANDIDATE_FEATURE_GROUPS["risk_liquidity"]
    + CANDIDATE_FEATURE_GROUPS["price_action"]
    + ["excess_ret_5d", "excess_ret_20d", "excess_ret_5d_rank", "excess_ret_20d_rank"]
)


@dataclass(frozen=True)
class PortfolioShape:
    top_k: int
    rank_power: float


def _apply_weight_cap(weights: pd.Series) -> pd.Series:
    w = weights[weights > 0].copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} names")
    w = w / w.sum()
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def build_rank_weighted_portfolio(scores: pd.Series, shape: PortfolioShape) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(shape.top_k)
    ranks = np.arange(len(chosen), 0, -1, dtype=float)
    raw = ranks ** shape.rank_power
    return _apply_weight_cap(pd.Series(raw / raw.sum(), index=chosen.index))


def add_excess_targets(panel: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df[["date", "close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx["idx_target_5d"] = idx["close"].shift(-FORWARD_HORIZON) / idx["close"] - 1.0
    out = panel.merge(idx[["date", "idx_target_5d"]], on="date", how="left")
    out["future_excess_5d"] = out[TARGET_COLUMN] - out["idx_target_5d"]
    out["target_rank_5d"] = out.groupby("date")["future_excess_5d"].rank(method="average", pct=True)
    out["rank_relevance_5d"] = np.floor(out["target_rank_5d"].clip(0, 1) * 9).astype("float")
    return out


def feature_columns(panel: pd.DataFrame, open_columns: list[str]) -> list[str]:
    cols = list(dict.fromkeys(CORE_FEATURE_COLUMNS + list(EXTRA_TECHNICAL_FEATURES) + open_columns))
    return [c for c in cols if c in panel.columns]


def split_train_val(df: pd.DataFrame, val_days: int = DEFAULT_VAL_DAYS):
    all_dates = np.sort(df["date"].unique())
    if len(all_dates) < val_days + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-val_days])
    train_end = pd.Timestamp(all_dates[-(val_days + EMBARGO_DAYS + 1)])
    train_df = df[df["date"] <= train_end].copy()
    val_df = df[df["date"] >= val_start].copy()
    return train_df, val_df, train_end, val_start


def _sort_for_ranker(df: pd.DataFrame, features: list[str], label: str):
    sorted_df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    group = sorted_df.groupby("date", sort=False).size().to_numpy()
    return sorted_df[features], sorted_df[label].to_numpy(), group


def train_ranker(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]):
    x_train, y_train, g_train = _sort_for_ranker(train_df, features, "rank_relevance_5d")
    x_val, y_val, g_val = _sort_for_ranker(val_df, features, "rank_relevance_5d")
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@50",
        n_estimators=650,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_alpha=0.05,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=40,
    )
    model.fit(
        x_train,
        y_train,
        group=g_train,
        eval_set=[(x_val, y_val)],
        eval_group=[g_val],
        verbose=False,
    )
    return model


def train_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]):
    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=650,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        reg_alpha=0.05,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=40,
    )
    model.fit(
        train_df[features],
        train_df["future_excess_5d"],
        eval_set=[(val_df[features], val_df["future_excess_5d"])],
        verbose=False,
    )
    return model


def validation_windows(val_dates: np.ndarray, trading_dates: np.ndarray):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for offset in range(0, len(val_dates), FORWARD_HORIZON):
        as_of = pd.Timestamp(val_dates[offset])
        idx = date_to_idx.get(as_of)
        if idx is None or idx + FORWARD_HORIZON >= len(trading_dates):
            continue
        windows.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])))
    return windows


def select_shape(model, panel, features, val_dates, trading_dates, prices, index_df):
    windows = validation_windows(val_dates, trading_dates)
    if not windows:
        return PortfolioShape(50, 0.75), pd.DataFrame()

    score_cache = {}
    for as_of, _, _ in windows:
        pred = prediction_frame_for_features(panel, features, as_of)
        pred["score"] = model.predict(pred[features])
        score_cache[as_of] = pred.set_index("stock_code")["score"]

    rows = []
    for top_k in DEFAULT_TOPK_GRID:
        for rank_power in DEFAULT_POWER_GRID:
            shape = PortfolioShape(top_k, rank_power)
            scores = []
            for as_of, start, end in windows:
                weights = build_rank_weighted_portfolio(score_cache[as_of], shape)
                scores.append(score_window(weights, prices, index_df, start, end)["excess_return"])
            rows.append(
                {
                    "top_k": top_k,
                    "rank_power": rank_power,
                    "mean_excess_return": float(np.mean(scores)),
                    "sum_excess_return": float(np.sum(scores)),
                    "min_excess_return": float(np.min(scores)),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["mean_excess_return", "min_excess_return", "sum_excess_return"],
        ascending=False,
    )
    best = table.iloc[0]
    return PortfolioShape(int(best["top_k"]), float(best["rank_power"])), table


def prediction_frame_for_features(panel: pd.DataFrame, features: list[str], as_of) -> pd.DataFrame:
    df = prediction_frame(panel, as_of=as_of)
    return df.dropna(subset=features).copy()


def fit_open_model(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    model_type: str = "ranker",
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    open_dir: str | Path = DATA_DIR / "open",
):
    raw_trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(as_of)
    if lookback_days and lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()

    panel = build_features(prices, index_df)
    panel, open_cols = add_open_data_features(panel, open_dir=open_dir)
    panel = add_excess_targets(panel, index_df)
    features = feature_columns(panel, open_cols)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    train_pool = panel[panel["date"] <= train_cutoff].dropna(
        subset=features + ["future_excess_5d", "target_rank_5d", "rank_relevance_5d"]
    )
    train_df, val_df, train_end, val_start = split_train_val(train_pool, val_days=val_days)
    model = train_ranker(train_df, val_df, features) if model_type == "ranker" else train_regressor(train_df, val_df, features)
    shape, shape_table = select_shape(
        model=model,
        panel=panel,
        features=features,
        val_dates=np.sort(val_df["date"].unique()),
        trading_dates=trading_dates,
        prices=prices,
        index_df=index_df,
    )
    return {
        "panel": panel,
        "features": features,
        "open_cols": open_cols,
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
    pred = prediction_frame_for_features(fit_result["panel"], fit_result["features"], as_of)
    pred["score"] = fit_result["model"].predict(pred[fit_result["features"]])
    weights = build_rank_weighted_portfolio(pred.set_index("stock_code")["score"], fit_result["shape"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DATA_DIR / "open"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="submissions/open_data_ranker.csv")
    parser.add_argument("--model-type", choices=["ranker", "regressor"], default="ranker")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = fit_open_model(
        prices=prices,
        index_df=index_df,
        as_of=as_of,
        model_type=args.model_type,
        lookback_days=args.lookback_days,
        val_days=args.val_days,
        open_dir=args.open_dir,
    )
    submission = generate_submission(fit, as_of=as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f">> model_type={args.model_type} as_of={as_of.date()}")
    print(f">> train: {len(fit['train_df']):,} rows up to {fit['train_end'].date()}")
    print(f">> val:   {len(fit['val_df']):,} rows from {fit['val_start'].date()}")
    print(f">> features={len(fit['features'])} open_features={len(fit['open_cols'])}")
    print(f">> selected shape: top_k={fit['shape'].top_k}, rank_power={fit['shape'].rank_power:.2f}")
    print(">> validation shape table")
    print(fit["shape_table"].head(12).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
