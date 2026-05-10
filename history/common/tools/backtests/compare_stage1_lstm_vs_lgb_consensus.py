"""Compare current stage1 LSTM route against the previous LGB-led route."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "archive" / "legacy_scripts"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_submission import score_window  # noqa: E402
from stage1_lstm_lgb_confidence import (  # noqa: E402
    AGGRESSIVE_LSTM_POLICY,
    fit_hybrid,
    generate_submission as generate_lstm_submission,
)
from stage1_consensus_portfolio import generate_consensus_submission  # noqa: E402


DEFAULT_WINDOWS = (
    ("20260415", "20260416", "20260420"),
    ("20260420", "20260421", "20260423"),
    ("20260421", "20260422", "20260424"),
    ("20260422", "20260423", "20260427"),
    ("20260424", "20260427", "20260429"),
    ("20260427", "20260428", "20260430"),
)


def _weights(submission: pd.DataFrame) -> pd.Series:
    sub = submission.copy()
    sub["stock_code"] = sub["stock_code"].astype(str).str.zfill(6)
    return sub.set_index("stock_code")["weight"].astype(float)


def _score(
    model: str,
    submission: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    weights = _weights(submission)
    result = score_window(weights, prices, index_df, start, end)
    return {
        "model": model,
        "as_of": as_of.date().isoformat(),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "n": int(len(weights)),
        "max_weight": float(weights.max()),
        "portfolio_return": result["portfolio_return"],
        "benchmark_return": result["benchmark_return"],
        "excess_return": result["excess_return"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lookback-days", type=int, default=520)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "submissions" / "stage1" / "backtests" / "stage1_lstm_vs_lgb_consensus_20260504"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(ROOT / "submissions" / "stage1" / "reports"),
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for as_of_s, start_s, end_s in DEFAULT_WINDOWS:
        as_of = pd.Timestamp(as_of_s)
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        print(f">> window as_of={as_of_s} eval={start_s}..{end_s}", flush=True)

        lstm_fit = fit_hybrid(
            prices,
            index_df,
            as_of,
            horizon=args.horizon,
            lookback_days=args.lookback_days,
            fixed_policy=AGGRESSIVE_LSTM_POLICY,
            confidence_mode="risk-balanced",
        )
        lstm_sub = generate_lstm_submission(lstm_fit, as_of, confidence_mode="risk-balanced")
        lstm_path = out_dir / f"risk_balanced_lstm_{as_of_s}.csv"
        lstm_sub.to_csv(lstm_path, index=False)
        rows.append(_score("risk_balanced_lstm", lstm_sub, prices, index_df, as_of, start, end))

        consensus_sub = generate_consensus_submission(
            prices,
            index_df,
            as_of=as_of,
            alpha_tuned_xgb=0.0,
            shape_horizon=args.horizon,
        )
        consensus_path = out_dir / f"lgb_consensus_previous_{as_of_s}.csv"
        consensus_sub.to_csv(consensus_path, index=False)
        rows.append(_score("lgb_consensus_previous", consensus_sub, prices, index_df, as_of, start, end))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("model")
        .agg(
            mean_excess=("excess_return", "mean"),
            min_excess=("excess_return", "min"),
            max_excess=("excess_return", "max"),
            std_excess=("excess_return", "std"),
            sum_excess=("excess_return", "sum"),
            mean_portfolio=("portfolio_return", "mean"),
            mean_benchmark=("benchmark_return", "mean"),
            mean_n=("n", "mean"),
            mean_max_weight=("max_weight", "mean"),
            count=("excess_return", "count"),
        )
        .reset_index()
        .sort_values(["mean_excess", "min_excess"], ascending=False)
    )
    detail_path = report_dir / "stage1_lstm_vs_lgb_consensus_multwindow_detail.csv"
    summary_path = report_dir / "stage1_lstm_vs_lgb_consensus_multwindow_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(">> detail")
    print(detail.to_string(index=False))
    print(">> summary")
    print(summary.to_string(index=False))
    print(f">> wrote {detail_path}")
    print(f">> wrote {summary_path}")


if __name__ == "__main__":
    main()
