"""Stage2 CatBoost portfolio challenger.

This keeps the same no-leakage train/validation/portfolio-shape pipeline used by
the tuned tree baselines, but swaps in CatBoost with a robust loss.  The goal is
to test whether ordered boosting contributes a genuinely different stock ranking
signal before we blend it into the production consensus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "archive" / "legacy_scripts"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_xgboost import EMBARGO_DAYS, FEATURE_COLUMNS, FORWARD_HORIZON
from features import TARGET_COLUMN, build_features, prediction_frame, training_frame
from tuned_xgboost_portfolio import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_VAL_DAYS,
    build_shaped_portfolio,
    select_shape,
    split_train_val,
    time_decay_weights,
)

DATA_DIR = ROOT / "data"


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
    target_column: str = TARGET_COLUMN,
    loss_function: str = "Quantile:alpha=0.55",
    iterations: int = 700,
    depth: int = 5,
    learning_rate: float = 0.035,
    l2_leaf_reg: float = 8.0,
    bagging_temperature: float = 0.25,
) -> CatBoostRegressor:
    features = features or FEATURE_COLUMNS
    model = CatBoostRegressor(
        loss_function=loss_function,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        random_seed=44,
        allow_writing_files=False,
        thread_count=1,
        verbose=False,
    )
    model.fit(
        train_df[features],
        train_df[target_column],
        sample_weight=sample_weight,
        eval_set=(val_df[features], val_df[target_column]),
        use_best_model=True,
        early_stopping_rounds=60,
    )
    return model


def fit_catboost_model(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    val_days: int = DEFAULT_VAL_DAYS,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    shape_horizon: int = FORWARD_HORIZON,
    features: list[str] | None = None,
    half_life: int | None = None,
    weight_floor: float = 0.0,
    target_column: str = TARGET_COLUMN,
    target_horizon: int = FORWARD_HORIZON,
    loss_function: str = "Quantile:alpha=0.55",
    iterations: int = 700,
    depth: int = 5,
    learning_rate: float = 0.035,
    l2_leaf_reg: float = 8.0,
    bagging_temperature: float = 0.25,
) -> dict:
    features = features or FEATURE_COLUMNS
    as_of = pd.Timestamp(as_of)
    if lookback_days is not None and lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()

    panel = build_features(prices, index_df)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - target_horizon)])
    train_pool = training_frame(panel, max_date=train_cutoff, target_column=target_column).dropna(subset=features)
    train_df, val_df, train_end, val_start = split_train_val(
        train_pool,
        val_days=val_days,
        embargo_days=EMBARGO_DAYS,
    )
    sample_weight = time_decay_weights(train_df, half_life=half_life, floor=weight_floor)
    model = train_model(
        train_df,
        val_df,
        features=features,
        sample_weight=sample_weight,
        target_column=target_column,
        loss_function=loss_function,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
    )
    shape, shape_table = select_shape(
        model=model,
        panel=panel,
        val_dates=np.sort(val_df["date"].unique()),
        trading_dates=trading_dates,
        prices=prices,
        index_df=index_df,
        shape_horizon=shape_horizon,
        features=features,
    )
    return {
        "panel": panel,
        "features": features,
        "target_column": target_column,
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
    pred = prediction_frame(fit_result["panel"], as_of=as_of).copy()
    features = fit_result.get("features", FEATURE_COLUMNS)
    pred = pred.dropna(subset=features)
    pred["score"] = fit_result["model"].predict(pred[features])
    weights = build_shaped_portfolio(pred.set_index("stock_code")["score"], fit_result["shape"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/experiments/catboost_portfolio.csv")
    parser.add_argument("--shape-out", default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--time-decay-half-life", type=int, default=0)
    parser.add_argument("--time-decay-floor", type=float, default=0.0)
    parser.add_argument("--loss-function", default="Quantile:alpha=0.55")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--bagging-temperature", type=float, default=0.25)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])
    half_life = args.time_decay_half_life if args.time_decay_half_life > 0 else None

    fit = fit_catboost_model(
        prices,
        index_df,
        as_of=as_of,
        val_days=args.val_days,
        lookback_days=args.lookback_days if args.lookback_days > 0 else None,
        half_life=half_life,
        weight_floor=args.time_decay_floor,
        loss_function=args.loss_function,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        bagging_temperature=args.bagging_temperature,
    )
    sub = generate_submission(fit, as_of=as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.shape_out:
        shape_path = Path(args.shape_out)
        shape_path.parent.mkdir(parents=True, exist_ok=True)
        fit["shape_table"].to_csv(shape_path, index=False)

    print(f">> model=stage2_catboost as_of={as_of.date()}")
    print(f">> train: {len(fit['train_df']):,} rows up to {fit['train_end'].date()}")
    print(f">> val:   {len(fit['val_df']):,} rows from {fit['val_start'].date()}")
    print(f">> selected shape: top_k={fit['shape'].top_k}, rank_power={fit['shape'].rank_power:.2f}")
    print(">> validation shape table")
    print(fit["shape_table"].head(10).to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
