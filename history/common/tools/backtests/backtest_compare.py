"""
Compare the original baseline and the tuned XGBoost portfolio on rolling windows.

The score uses the exact same `score_window` logic as `score_submission.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_xgboost import (
    FEATURE_COLUMNS as BASELINE_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    build_features as build_baseline_features,
    build_portfolio as build_baseline_portfolio,
    prediction_frame as baseline_prediction_frame,
    train_model as train_baseline_model,
    training_frame as baseline_training_frame,
)
from score_submission import score_window
from tuned_xgboost_portfolio import (
    fit_tuned_model,
    generate_submission as generate_tuned_submission,
    split_train_val,
)

DATA_DIR = ROOT / "data"
VAL_DAYS = 10
EMBARGO_DAYS = 5


def fit_baseline_for_date(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp):
    panel = build_baseline_features(prices, index_df)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = baseline_training_frame(panel, max_date=train_cutoff)
    train_df, val_df, _, _ = split_train_val(
        train_pool,
        val_days=VAL_DAYS,
        embargo_days=EMBARGO_DAYS,
    )
    model = train_baseline_model(train_df, val_df)
    pred_df = baseline_prediction_frame(panel, as_of=as_of)
    pred_df = pred_df.assign(score=model.predict(pred_df[BASELINE_FEATURE_COLUMNS]))
    weights = build_baseline_portfolio(pred_df.set_index("stock_code")["score"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def rolling_windows(trading_dates: np.ndarray, n_windows: int, step: int):
    windows = []
    start_idx = max(120, len(trading_dates) - step * n_windows - FORWARD_HORIZON)
    idx = start_idx
    while idx + FORWARD_HORIZON < len(trading_dates) and len(windows) < n_windows:
        as_of = pd.Timestamp(trading_dates[idx])
        start = pd.Timestamp(trading_dates[idx + 1])
        end = pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])
        windows.append((as_of, start, end))
        idx += step
    return windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--csv-out", default=None)
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(pd.to_datetime(prices["date"].unique()))

    rows = []
    for as_of, start, end in rolling_windows(trading_dates, args.windows, args.step):
        print(f">> backtesting as_of={as_of.date()} window={start.date()}..{end.date()}")

        baseline_sub = fit_baseline_for_date(prices, index_df, as_of)
        baseline_score = score_window(
            baseline_sub.set_index("stock_code")["weight"],
            prices,
            index_df,
            start,
            end,
        )

        tuned_fit = fit_tuned_model(prices=prices, index_df=index_df, as_of=as_of)
        tuned_sub = generate_tuned_submission(tuned_fit, as_of=as_of)
        tuned_score = score_window(
            tuned_sub.set_index("stock_code")["weight"],
            prices,
            index_df,
            start,
            end,
        )

        rows.extend(
            [
                {
                    "model": "baseline",
                    "as_of": as_of.date().isoformat(),
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "portfolio_return": baseline_score["portfolio_return"],
                    "benchmark_return": baseline_score["benchmark_return"],
                    "excess_return": baseline_score["excess_return"],
                    "n_names": len(baseline_sub),
                },
                {
                    "model": "tuned_xgb_portfolio",
                    "as_of": as_of.date().isoformat(),
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "portfolio_return": tuned_score["portfolio_return"],
                    "benchmark_return": tuned_score["benchmark_return"],
                    "excess_return": tuned_score["excess_return"],
                    "n_names": len(tuned_sub),
                },
            ]
        )

    result = pd.DataFrame(rows)
    summary = (
        result.groupby("model")["excess_return"]
        .agg(["mean", "sum", "median", "min", "max"])
        .reset_index()
    )
    print(">> summary")
    print(summary.to_string(index=False))
    print(">> per-window detail")
    print(result.to_string(index=False))

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f">> wrote detail csv to {out_path}")


if __name__ == "__main__":
    main()
