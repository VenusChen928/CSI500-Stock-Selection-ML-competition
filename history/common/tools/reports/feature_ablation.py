"""
Score-window feature ablation for the tuned XGBoost pipeline.

This script evaluates feature groups by the same realized excess-return scoring
used by the competition helpers.  It is intentionally stricter than model loss
or feature importance: a feature group is only useful if it improves the
portfolio that would actually be submitted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import (
    CANDIDATE_FEATURE_GROUPS,
    CORE_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
)
from score_submission import score_window
from tuned_xgboost_portfolio import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_VAL_DAYS,
    PortfolioShape,
    build_shaped_portfolio,
    split_train_val,
)

DATA_DIR = ROOT / "data"
DEFAULT_TOPK_GRID = (40, 45, 50, 55, 60, 65, 70, 80, 100)
DEFAULT_POWER_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)


def _fit_model(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        random_state=42,
    )
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def _validation_windows(val_dates: np.ndarray, trading_dates: np.ndarray):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for offset in range(0, len(val_dates), FORWARD_HORIZON):
        as_of = pd.Timestamp(val_dates[offset])
        idx = date_to_idx.get(as_of)
        if idx is None or idx + FORWARD_HORIZON >= len(trading_dates):
            continue
        windows.append(
            (
                as_of,
                pd.Timestamp(trading_dates[idx + 1]),
                pd.Timestamp(trading_dates[idx + FORWARD_HORIZON]),
            )
        )
    return windows


def _select_shape(model, panel, feature_cols, val_dates, trading_dates, prices, index_df):
    windows = _validation_windows(val_dates, trading_dates)
    score_cache = {}
    for as_of, _, _ in windows:
        pred = panel[panel["date"] == as_of].dropna(subset=feature_cols).copy()
        pred["score"] = model.predict(pred[feature_cols])
        score_cache[as_of] = pred.set_index("stock_code")["score"]

    best = None
    for top_k in DEFAULT_TOPK_GRID:
        for rank_power in DEFAULT_POWER_GRID:
            shape = PortfolioShape(top_k=top_k, rank_power=rank_power)
            scores = []
            for as_of, start, end in windows:
                weights = build_shaped_portfolio(score_cache[as_of], shape)
                scores.append(score_window(weights, prices, index_df, start, end)["excess_return"])
            row = (
                float(np.mean(scores)),
                float(np.min(scores)),
                float(np.sum(scores)),
                top_k,
                rank_power,
            )
            if best is None or row > best:
                best = row
    return best


def feature_variants() -> dict[str, list[str]]:
    variants = {"core": CORE_FEATURE_COLUMNS}
    for name, cols in CANDIDATE_FEATURE_GROUPS.items():
        variants[f"core+{name}"] = CORE_FEATURE_COLUMNS + cols
    variants["core+relative+state"] = (
        CORE_FEATURE_COLUMNS
        + CANDIDATE_FEATURE_GROUPS["market_relative"]
        + CANDIDATE_FEATURE_GROUPS["market_state"]
    )
    variants["core+liquidity+relative"] = (
        CORE_FEATURE_COLUMNS
        + CANDIDATE_FEATURE_GROUPS["risk_liquidity"]
        + CANDIDATE_FEATURE_GROUPS["market_relative"]
    )
    variants["all"] = CORE_FEATURE_COLUMNS + [
        feature
        for group in CANDIDATE_FEATURE_GROUPS.values()
        for feature in group
    ]
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in prices")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--out", default="submissions/feature_ablation.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    as_of = pd.Timestamp(args.as_of) if args.as_of else prices["date"].max()
    if args.lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=args.lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()

    panel = build_features(prices, index_df)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])

    rows = []
    for name, feature_cols in feature_variants().items():
        feature_cols = [col for col in feature_cols if col in panel.columns]
        train_pool = panel.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
        train_pool = train_pool[train_pool["date"] <= train_cutoff]
        train_df, val_df, _, _ = split_train_val(train_pool, val_days=args.val_days)
        model = _fit_model(train_df, val_df, feature_cols)
        val_mean, val_min, val_sum, top_k, rank_power = _select_shape(
            model,
            panel,
            feature_cols,
            np.sort(val_df["date"].unique()),
            trading_dates,
            prices,
            index_df,
        )

        pred = panel[panel["date"] == as_of].dropna(subset=feature_cols).copy()
        pred["score"] = model.predict(pred[feature_cols])
        weights = build_shaped_portfolio(
            pred.set_index("stock_code")["score"],
            PortfolioShape(top_k=top_k, rank_power=rank_power),
        )
        latest = score_window(
            weights,
            prices,
            index_df,
            pd.Timestamp(trading_dates[as_of_idx + 1]),
            pd.Timestamp(trading_dates[as_of_idx + FORWARD_HORIZON]),
        )["excess_return"]
        rows.append(
            {
                "variant": name,
                "n_features": len(feature_cols),
                "val_mean": val_mean,
                "val_min": val_min,
                "val_sum": val_sum,
                "top_k": top_k,
                "rank_power": rank_power,
                "latest_excess": latest,
            }
        )

    result = pd.DataFrame(rows).sort_values(["latest_excess", "val_mean"], ascending=False)
    print(result.to_string(index=False))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f">> wrote {out_path}")


if __name__ == "__main__":
    main()
