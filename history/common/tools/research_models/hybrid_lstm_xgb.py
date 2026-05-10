"""
Hybrid portfolio: use LSTM rank-weight model only when validation evidence is strong.

The LSTM can add useful sequence alpha, but it is more variable than the tuned
XGBoost route.  This wrapper treats tuned XGBoost as the default and switches to
LSTM only when the LSTM validation policy clears a realized-return threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import FORWARD_HORIZON
from lstm_rank_weight import fit_lstm, generate_submission as generate_lstm_submission
from tuned_xgboost_portfolio import fit_tuned_model, generate_submission as generate_tuned_submission

DATA_DIR = ROOT / "data"
DEFAULT_LSTM_VAL_THRESHOLD = 0.006


def generate_hybrid_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lstm_val_threshold: float = DEFAULT_LSTM_VAL_THRESHOLD,
):
    lstm_fit = fit_lstm(prices=prices, index_df=index_df, as_of=as_of)
    lstm_best = float(lstm_fit["policy_table"].iloc[0]["mean_excess_return"])
    if lstm_best >= lstm_val_threshold:
        return generate_lstm_submission(lstm_fit, as_of=as_of), "lstm_rank_weight", lstm_best
    tuned_fit = fit_tuned_model(prices=prices, index_df=index_df, as_of=as_of)
    tuned_sub = generate_tuned_submission(tuned_fit, as_of=as_of)
    return tuned_sub, "tuned_xgb_portfolio", lstm_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--lstm-val-threshold", type=float, default=DEFAULT_LSTM_VAL_THRESHOLD)
    parser.add_argument("--out", default="submissions/hybrid_lstm_xgb.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = sorted(pd.to_datetime(prices["date"].unique()))
    as_of = pd.Timestamp(args.as_of) if args.as_of else trading_dates[-(FORWARD_HORIZON + 1)]

    submission, selected_model, lstm_val_mean = generate_hybrid_submission(
        prices,
        index_df,
        as_of,
        lstm_val_threshold=args.lstm_val_threshold,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f">> selected_model={selected_model}")
    print(f">> lstm_validation_mean={lstm_val_mean:.6f}")
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
