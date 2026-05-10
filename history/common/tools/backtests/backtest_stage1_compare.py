"""
Backtest stage1-style 3-trading-day portfolios.

This runner mirrors the first evaluation window length (3 trading days) and
keeps each model in a fresh subprocess so Torch/XGBoost/LightGBM state cannot
leak across windows.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/mlcomp-sp26/bin/python")
ENV = os.environ.copy()
ENV["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:/opt/homebrew/opt/openssl@3/lib"
ENV.setdefault("MLCOMP_DEVICE", "cpu")
HORIZON = 3


def run(cmd: list[str]) -> str:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.check_output(cmd, cwd=ROOT, env=ENV, text=True, stderr=subprocess.STDOUT)


def parse_score(text: str) -> dict:
    row = {}
    for line in text.splitlines():
        if "portfolio return" in line:
            row["portfolio_return"] = float(line.split()[3].replace("%", "")) / 100
        elif "benchmark return" in line:
            row["benchmark_return"] = float(line.split()[3].replace("%", "")) / 100
        elif "excess return" in line:
            row["excess_return"] = float(line.split()[3].replace("%", "")) / 100
    return row


def score(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    text = run(
        [
            str(PYTHON),
            "score_submission.py",
            str(path),
            "--start",
            start.strftime("%Y%m%d"),
            "--end",
            end.strftime("%Y%m%d"),
        ]
    )
    return parse_score(text)


def window_for_asof(trading_dates: np.ndarray, as_of: pd.Timestamp):
    dates = [pd.Timestamp(d) for d in trading_dates]
    idx = dates.index(pd.Timestamp(as_of))
    return dates[idx + 1], dates[idx + HORIZON]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        nargs="+",
        default=["20260415", "20260420", "20260421", "20260422", "20260424"],
    )
    parser.add_argument("--out-dir", default="submissions/stage1/backtests/stage1_compare_20260504")
    parser.add_argument("--summary-out", default="submissions/stage1/reports/stage1_compare_20260504_summary.csv")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    args = parser.parse_args()

    prices = pd.read_parquet(ROOT / "data/prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    trading_dates = np.sort(prices["date"].unique())

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for as_of_text in args.as_of:
        as_of = pd.Timestamp(as_of_text)
        start, end = window_for_asof(trading_dates, as_of)
        print(f"\n## as_of={as_of.date()} window={start.date()}..{end.date()}", flush=True)
        suffix = as_of.strftime("%Y%m%d")
        models = {
            "baseline_xgb": [
                "baseline_xgboost.py",
                "--as-of",
                suffix,
                "--out",
                str(out_dir / f"baseline_xgb_{suffix}.csv"),
            ],
            "tuned_xgb_h3_shape": [
                "tuned_xgboost_portfolio.py",
                "--as-of",
                suffix,
                "--shape-horizon",
                "3",
                "--out",
                str(out_dir / f"tuned_xgb_h3_shape_{suffix}.csv"),
            ],
            "lightgbm_h3_shape": [
                "lightgbm_portfolio.py",
                "--as-of",
                suffix,
                "--shape-horizon",
                "3",
                "--out",
                str(out_dir / f"lightgbm_h3_shape_{suffix}.csv"),
            ],
            "gated_lstm_h3": [
                "gated_layered_lstm.py",
                "--as-of",
                suffix,
                "--target-horizon",
                "3",
                "--policy-horizon",
                "3",
                "--validation-horizon",
                "3",
                "--out",
                str(out_dir / f"gated_lstm_h3_{suffix}.csv"),
                "--table-out",
                str(out_dir / f"gated_lstm_h3_policy_{suffix}.csv"),
            ],
        }
        if args.skip_lstm:
            models.pop("gated_lstm_h3")
        for model_name, cmd in models.items():
            path = Path(cmd[cmd.index("--out") + 1])
            if not (args.skip_existing and path.exists()):
                run([str(PYTHON), "-u", *cmd])
            score_row = score(path, start, end)
            rows.append(
                {
                    "model": model_name,
                    "as_of": as_of.date().isoformat(),
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "submission": str(path.relative_to(ROOT)),
                    **score_row,
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("model")["excess_return"]
        .agg(["mean", "sum", "median", "min", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    summary_out = ROOT / args.summary_out
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(summary_out.with_name(summary_out.stem + "_detail.csv"), index=False)
    summary.to_csv(summary_out, index=False)
    print("\n>> summary")
    print(summary.to_string(index=False))
    print(f">> wrote {summary_out}")
    print(f">> wrote {summary_out.with_name(summary_out.stem + '_detail.csv')}")


if __name__ == "__main__":
    sys.exit(main())
