"""Stage2 5-trading-day multi-window backtest runner.

This lean version only knows about active, report-relevant routes.  Historical
open-data probes, rejected deep models, and archived candidate scripts were
removed so the validation tool mirrors the final project layout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/mlcomp-sp26/bin/python")
FORWARD_HORIZON = 5


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'stage2_report' / 'scripts'}:{env.get('PYTHONPATH', '')}"
    env["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:/opt/homebrew/opt/openssl@3/lib"
    env["MLCOMP_DEVICE"] = "cpu"
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MLCOMP_TORCH_THREADS",
        "MLCOMP_TORCH_INTEROP_THREADS",
    ]:
        env[key] = "1"
    return env


def run(cmd: list[str], *, timeout: int | None = None) -> str:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.check_output(
        cmd,
        cwd=ROOT,
        env=_env(),
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def parse_score(text: str) -> dict[str, float]:
    row: dict[str, float] = {}
    for line in text.splitlines():
        if "portfolio return" in line:
            row["portfolio_return"] = float(line.split()[3].replace("%", "")) / 100.0
        elif "benchmark return" in line:
            row["benchmark_return"] = float(line.split()[3].replace("%", "")) / 100.0
        elif "excess return" in line:
            row["excess_return"] = float(line.split()[3].replace("%", "")) / 100.0
    return row


def score(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    text = run([
        str(PYTHON),
        "score_submission.py",
        str(path),
        "--start",
        start.strftime("%Y%m%d"),
        "--end",
        end.strftime("%Y%m%d"),
    ])
    return parse_score(text)


def rolling_windows(
    trading_dates: np.ndarray,
    windows: int,
    step: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    out = []
    max_asof_idx = len(trading_dates) - FORWARD_HORIZON - 1
    idx = max_asof_idx - step * (windows - 1)
    while idx <= max_asof_idx:
        if idx >= 120:
            as_of = pd.Timestamp(trading_dates[idx])
            out.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])))
        idx += step
    return out


def full_week_windows(
    trading_dates: np.ndarray,
    windows: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    out = []
    for idx in range(len(dates) - FORWARD_HORIZON - 1, 119, -1):
        as_of = dates[idx]
        start = dates[idx + 1]
        end = dates[idx + FORWARD_HORIZON]
        window = dates[idx + 1 : idx + FORWARD_HORIZON + 1]
        if start.weekday() == 0 and end.weekday() == 4 and (end - start).days == 4 and len(window) == 5:
            out.append((as_of, start, end))
        if len(out) >= windows:
            break
    return list(reversed(out))


def explicit_windows(
    trading_dates: np.ndarray,
    as_of_values: list[str],
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    out = []
    for value in as_of_values:
        as_of = pd.to_datetime(value, format="%Y%m%d")
        if as_of not in dates:
            raise ValueError(f"as_of {value} is not in trading dates")
        idx = dates.index(as_of)
        if idx + FORWARD_HORIZON >= len(dates):
            raise ValueError(f"as_of {value} does not have {FORWARD_HORIZON} future trading days in local data")
        out.append((as_of, dates[idx + 1], dates[idx + FORWARD_HORIZON]))
    return out


def model_commands(as_of: pd.Timestamp, out_dir: Path) -> dict[str, tuple[list[str], Path]]:
    stamp = as_of.strftime("%Y%m%d")
    return {
        "baseline_xgb": (
            [str(PYTHON), "-u", "baseline_xgboost.py", "--as-of", stamp, "--out", str(out_dir / f"baseline_xgb_{stamp}.csv")],
            out_dir / f"baseline_xgb_{stamp}.csv",
        ),
        "tree_consensus": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"tree_consensus_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_{stamp}.csv",
        ),
        "weekly_alpha_auto": (
            [
                str(PYTHON), "-u", "stage2_weekly_alpha_overlay.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--mode", "auto",
                "--out", str(out_dir / f"weekly_alpha_auto_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_alpha_auto_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_alpha_auto_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_alpha_auto_{stamp}.csv",
        ),
        "weekly_cycle_tree": (
            [
                str(PYTHON), "-u", "stage2_weekly_cycle_tree.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"weekly_cycle_tree_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_cycle_tree_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_cycle_tree_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_cycle_tree_{stamp}.csv",
        ),
        "weekly_consensus": (
            [
                str(PYTHON), "-u", "stage2_weekly_consensus_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--rank-power", "6.0",
                "--max-weight", "0.095",
                "--out", str(out_dir / f"weekly_consensus_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_consensus_meta_{stamp}.csv"),
            ],
            out_dir / f"weekly_consensus_{stamp}.csv",
        ),
        "baseline_guard_top30": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "30",
                "--out", str(out_dir / f"baseline_guard_top30_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_top30_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_top30_{stamp}.csv",
        ),
        "baseline_guard_top40": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "40",
                "--out", str(out_dir / f"baseline_guard_top40_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_top40_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_top40_{stamp}.csv",
        ),
        "baseline_guard_adaptive": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "0",
                "--out", str(out_dir / f"baseline_guard_adaptive_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_adaptive_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_adaptive_{stamp}.csv",
        ),
    }


def run_one(
    *,
    model: str,
    as_of: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    skip_existing: bool,
    timeout: int,
) -> dict:
    commands = model_commands(as_of, out_dir)
    if model not in commands:
        raise ValueError(f"Unknown model {model}. Valid: {sorted(commands)}")
    cmd, submission_path = commands[model]
    status = "ok"
    error = ""
    if not (skip_existing and submission_path.exists()):
        try:
            run(cmd, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = exc.output[-4000:]
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = str(exc)
    if status == "ok":
        try:
            score_row = score(submission_path, start, end)
            n_names = len(pd.read_csv(submission_path))
        except Exception as exc:
            status = "score_failed"
            error = str(exc)
            score_row = {"portfolio_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan}
            n_names = np.nan
    else:
        score_row = {"portfolio_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan}
        n_names = np.nan
    return {
        "model": model,
        "as_of": as_of.date().isoformat(),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "submission": str(submission_path.relative_to(ROOT)) if submission_path.exists() else "",
        "status": status,
        "error": error,
        "portfolio_return": score_row["portfolio_return"],
        "benchmark_return": score_row["benchmark_return"],
        "excess_return": score_row["excess_return"],
        "n_names": n_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--models", nargs="+", default=["baseline_xgb", "baseline_guard_adaptive"])
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--as-of", nargs="*", default=None)
    parser.add_argument(
        "--full-week-only",
        action="store_true",
        help="Use only complete Monday-Friday windows with no weekend/holiday gap inside the five trading days.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "stage2_report" / "backtests" / "stage2_5day_current"))
    parser.add_argument("--summary-out", default=str(ROOT / "stage2_report" / "final_report_materials" / "stage2_5day_current_summary.csv"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of window/model jobs to run in parallel. Each job is a subprocess with BLAS/Torch threads pinned to 1.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    trading_dates = np.sort(prices["date"].unique())
    if args.as_of:
        windows = explicit_windows(trading_dates, args.as_of)
    elif args.full_week_only:
        windows = full_week_windows(trading_dates, args.windows)
    else:
        windows = rolling_windows(trading_dates, args.windows, args.step)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [
        {
            "model": model,
            "as_of": as_of,
            "start": start,
            "end": end,
            "out_dir": out_dir,
            "skip_existing": args.skip_existing,
            "timeout": args.timeout,
        }
        for as_of, start, end in windows
        for model in args.models
    ]
    for as_of, start, end in windows:
        print(f"## window as_of={as_of.date()} score={start.date()}..{end.date()}", flush=True)
    if args.jobs <= 1:
        for job in jobs:
            row = run_one(**job)
            if row["status"] != "ok":
                print(f"[warn] {row['model']} {row['as_of']} {row['status']}: {row['error']}", flush=True)
            rows.append(row)
    else:
        print(f">> running {len(jobs)} jobs with --jobs={args.jobs}", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_job = {executor.submit(run_one, **job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "model": job["model"],
                        "as_of": job["as_of"].date().isoformat(),
                        "start": job["start"].date().isoformat(),
                        "end": job["end"].date().isoformat(),
                        "submission": "",
                        "status": "failed",
                        "error": str(exc),
                        "portfolio_return": np.nan,
                        "benchmark_return": np.nan,
                        "excess_return": np.nan,
                        "n_names": np.nan,
                    }
                if row["status"] != "ok":
                    print(f"[warn] {row['model']} {row['as_of']} {row['status']}: {row['error']}", flush=True)
                rows.append(row)

    detail = pd.DataFrame(rows).sort_values(["as_of", "model"])
    summary = (
        detail[detail["status"] == "ok"]
        .groupby("model")["excess_return"]
        .agg(["mean", "sum", "median", "min", "max", "count", lambda x: int((x < 0).sum())])
        .rename(columns={"<lambda_0>": "negative_windows"})
        .sort_values("mean", ascending=False)
    )
    summary_path = Path(args.summary_out)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = summary_path.with_name(summary_path.stem + "_detail.csv")
    summary.to_csv(summary_path)
    detail.to_csv(detail_path, index=False)
    print("\n>> summary")
    print(summary.to_string())
    print(f"\n>> wrote {summary_path}")
    print(f">> wrote {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
