"""
Subprocess backtest for baseline, tuned XGB, LSTM, and layered LSTM.

Torch/MPS can be fragile when repeatedly training sequence models in one Python
process, so this runner launches model scripts in fresh subprocesses per
window.  It then scores all submissions with the canonical score_submission.py.
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
DEFAULT_LAYER_MEAN_THRESHOLD = 0.010
DEFAULT_LAYER_MIN_THRESHOLD = -0.020


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


def gated_choice(
    out_dir: Path,
    as_of: pd.Timestamp,
    layer_mean_threshold: float,
    layer_min_threshold: float,
) -> tuple[str, str, float, Path]:
    as_of_text = as_of.strftime("%Y%m%d")
    policy_path = out_dir / f"layered_policy_{as_of_text}.csv"
    policy = pd.read_csv(policy_path).iloc[0]
    layer_mean = float(policy["mean_excess_return"])
    layer_min = float(policy["min_excess_return"])
    if layer_mean >= layer_mean_threshold and layer_min >= layer_min_threshold:
        return "layered_lstm", str(policy["policy"]), layer_mean, out_dir / f"layered_lstm_{as_of_text}.csv"
    return "tuned_xgb_fallback", str(policy["policy"]), layer_mean, out_dir / f"tuned_xgb_{as_of_text}.csv"


def window_for_asof(trading_dates: np.ndarray, as_of: pd.Timestamp):
    dates = [pd.Timestamp(d) for d in trading_dates]
    idx = dates.index(pd.Timestamp(as_of))
    return dates[idx + 1], dates[idx + 5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        nargs="+",
        default=["20260324", "20260331", "20260408", "20260415", "20260421"],
    )
    parser.add_argument("--out-dir", default="submissions/stage2/backtests/archive_reference/layered_compare_20260502")
    parser.add_argument("--summary-out", default="submissions/legacy/historical_reports/layered_compare_5w_summary.csv")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--layer-mean-threshold", type=float, default=DEFAULT_LAYER_MEAN_THRESHOLD)
    parser.add_argument("--layer-min-threshold", type=float, default=DEFAULT_LAYER_MIN_THRESHOLD)
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
        models = {
            "baseline": (
                "baseline_xgboost.py",
                out_dir / f"baseline_{as_of.strftime('%Y%m%d')}.csv",
            ),
            "tuned_xgb": (
                "tuned_xgboost_portfolio.py",
                out_dir / f"tuned_xgb_{as_of.strftime('%Y%m%d')}.csv",
            ),
            "lightgbm": (
                "lightgbm_portfolio.py",
                out_dir / f"lightgbm_{as_of.strftime('%Y%m%d')}.csv",
            ),
            "lstm_rank_weight": (
                "lstm_rank_weight.py",
                out_dir / f"lstm_rank_weight_{as_of.strftime('%Y%m%d')}.csv",
            ),
            "layered_lstm": (
                "layered_lstm_portfolio.py",
                out_dir / f"layered_lstm_{as_of.strftime('%Y%m%d')}.csv",
            ),
        }
        for model_name, (script, path) in models.items():
            if not (args.skip_existing and path.exists()):
                cmd = [str(PYTHON), "-u", script, "--as-of", as_of.strftime("%Y%m%d"), "--out", str(path)]
                if model_name == "layered_lstm":
                    cmd += ["--table-out", str(out_dir / f"layered_policy_{as_of.strftime('%Y%m%d')}.csv")]
                run(cmd)
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
        selected, layer_policy, layer_val_mean, gated_path = gated_choice(
            out_dir,
            as_of,
            layer_mean_threshold=args.layer_mean_threshold,
            layer_min_threshold=args.layer_min_threshold,
        )
        score_row = score(gated_path, start, end)
        rows.append(
            {
                "model": "gated_layered_lstm",
                "as_of": as_of.date().isoformat(),
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "submission": str(gated_path.relative_to(ROOT)),
                "selected": selected,
                "layer_policy": layer_policy,
                "layer_val_mean": layer_val_mean,
                **score_row,
            }
        )

    result = pd.DataFrame(rows)
    summary = (
        result.groupby("model")["excess_return"]
        .agg(["mean", "sum", "median", "min", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    summary_out = ROOT / args.summary_out
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(summary_out.with_name(summary_out.stem + "_detail.csv"), index=False)
    summary.to_csv(summary_out, index=False)
    print("\n>> summary")
    print(summary.to_string(index=False))
    print(f">> wrote {summary_out}")
    print(f">> wrote {summary_out.with_name(summary_out.stem + '_detail.csv')}")


if __name__ == "__main__":
    sys.exit(main())
