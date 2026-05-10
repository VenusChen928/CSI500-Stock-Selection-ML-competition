"""
Validation-tuned LightGBM portfolio.

This is a controlled challenger to tuned_xgboost_portfolio.py: same features,
same train/validation split, same rank-weighted portfolio layer, but a LightGBM
regressor as the scoring model.  Keeping the portfolio construction identical
makes the comparison attributable to the model class rather than confounded
workflow changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline_xgboost import (
    EMBARGO_DAYS,
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    build_features,
    prediction_frame,
    training_frame,
)
from tuned_xgboost_portfolio import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_VAL_DAYS,
    build_shaped_portfolio,
    select_shape,
    split_train_val,
    time_decay_weights,
)

DATA_DIR = Path(__file__).parent / "data"


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
    target_column: str = "target_5d",
    learning_rate: float = 0.035,
    n_estimators: int = 800,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 80,
    reg_alpha: float = 0.05,
    reg_lambda: float = 2.0,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    objective: str = "huber",
) -> lgb.LGBMRegressor:
    features = features or FEATURE_COLUMNS
    model = lgb.LGBMRegressor(
        objective=objective,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train_df[features],
        train_df[target_column],
        sample_weight=sample_weight,
        eval_set=[(val_df[features], val_df[target_column])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def fit_lightgbm_model(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    val_days: int = DEFAULT_VAL_DAYS,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    learning_rate: float = 0.035,
    n_estimators: int = 800,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 80,
    reg_alpha: float = 0.05,
    reg_lambda: float = 2.0,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    objective: str = "huber",
    shape_horizon: int = FORWARD_HORIZON,
    features: list[str] | None = None,
    half_life: int | None = None,
    weight_floor: float = 0.0,
    target_column: str = "target_5d",
    target_horizon: int = FORWARD_HORIZON,
):
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
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="submissions/lightgbm_portfolio.csv")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument(
        "--shape-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon used to select top_k/rank_power on validation windows.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = fit_lightgbm_model(
        prices,
        index_df,
        as_of=as_of,
        val_days=args.val_days,
        lookback_days=args.lookback_days,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        shape_horizon=args.shape_horizon,
    )
    submission = generate_submission(fit, as_of=as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f">> model=lightgbm as_of={as_of.date()}")
    print(f">> train: {len(fit['train_df']):,} rows up to {fit['train_end'].date()}")
    print(f">> val:   {len(fit['val_df']):,} rows from {fit['val_start'].date()}")
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
