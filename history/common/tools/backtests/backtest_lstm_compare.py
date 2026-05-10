"""
Rolling comparison for baseline, tuned XGBoost, and LSTM rank-weight model.
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

from backtest_compare import fit_baseline_for_date, rolling_windows
from lstm_rank_weight import fit_lstm, generate_submission as generate_lstm_submission
from score_submission import score_window
from tuned_xgboost_portfolio import fit_tuned_model, generate_submission as generate_tuned_submission

DATA_DIR = ROOT / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--csv-out", default="submissions/backtest_lstm_compare.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(pd.to_datetime(prices["date"].unique()))

    rows = []
    for as_of, start, end in rolling_windows(trading_dates, args.windows, args.step):
        print(f">> backtesting as_of={as_of.date()} window={start.date()}..{end.date()}")

        baseline_sub = fit_baseline_for_date(prices, index_df, as_of)
        tuned_fit = fit_tuned_model(prices=prices, index_df=index_df, as_of=as_of)
        tuned_sub = generate_tuned_submission(tuned_fit, as_of=as_of)
        lstm_fit = fit_lstm(prices=prices, index_df=index_df, as_of=as_of)
        lstm_sub = generate_lstm_submission(lstm_fit, as_of=as_of)

        for model_name, sub in [
            ("baseline", baseline_sub),
            ("tuned_xgb_portfolio", tuned_sub),
            ("lstm_rank_weight", lstm_sub),
        ]:
            result = score_window(
                sub.set_index("stock_code")["weight"],
                prices,
                index_df,
                start,
                end,
            )
            rows.append(
                {
                    "model": model_name,
                    "as_of": as_of.date().isoformat(),
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "portfolio_return": result["portfolio_return"],
                    "benchmark_return": result["benchmark_return"],
                    "excess_return": result["excess_return"],
                    "n_names": len(sub),
                }
            )

    result = pd.DataFrame(rows)
    summary = result.groupby("model")["excess_return"].agg(["mean", "sum", "median", "min", "max"]).reset_index()
    print(">> summary")
    print(summary.to_string(index=False))
    print(">> detail")
    print(result.to_string(index=False))

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f">> wrote {out_path}")


if __name__ == "__main__":
    main()
