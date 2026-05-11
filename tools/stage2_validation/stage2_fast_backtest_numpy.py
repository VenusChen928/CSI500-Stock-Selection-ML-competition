"""Fast parallel stage2 backtester with numpy scoring.

Model generation still happens in isolated subprocesses, but scoring no longer
calls ``score_submission.py`` for each window.  Instead this script precomputes
per-window stock returns and benchmark returns as numpy arrays, then scores each
CSV with one dot product.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_backtest_5day import (
    FORWARD_HORIZON,
    PYTHON,
    ROOT,
    explicit_windows,
    full_week_windows,
    model_commands,
    rolling_windows,
)


def env() -> dict[str, str]:
    out = os.environ.copy()
    out["PYTHONPATH"] = f"{ROOT}:{out.get('PYTHONPATH', '')}"
    out["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:/opt/homebrew/opt/openssl@3/lib"
    out["MLCOMP_DEVICE"] = "cpu"
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MLCOMP_TORCH_THREADS",
        "MLCOMP_TORCH_INTEROP_THREADS",
    ]:
        out[key] = "1"
    return out


class NumpyScorer:
    def __init__(self, prices_path: Path, index_path: Path, windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]):
        prices = pd.read_parquet(prices_path, columns=["date", "stock_code", "close"])
        prices["date"] = pd.to_datetime(prices["date"])
        prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
        matrix = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
        self.codes = matrix.columns.astype(str).to_numpy()
        self.code_to_idx = {code: idx for idx, code in enumerate(self.codes)}
        self.dates = pd.to_datetime(matrix.index).to_numpy()
        self.close = matrix.to_numpy(dtype=float)

        index_df = pd.read_parquet(index_path, columns=["date", "close"])
        index_df["date"] = pd.to_datetime(index_df["date"])
        index_df = index_df.sort_values("date")
        self.index_dates = index_df["date"].to_numpy()
        self.index_close = index_df["close"].to_numpy(dtype=float)
        self.window_returns: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        for _, start, end in windows:
            self.window_returns[(start.date().isoformat(), end.date().isoformat())] = self._window_return(start, end)

    def _pos(self, dates: np.ndarray, date: pd.Timestamp) -> int:
        matches = np.flatnonzero(dates == np.datetime64(date))
        if len(matches) == 0:
            raise ValueError(f"date {date.date()} missing")
        return int(matches[0])

    def _prev_pos(self, dates: np.ndarray, date: pd.Timestamp) -> int:
        pos = int(np.searchsorted(dates, np.datetime64(date), side="left")) - 1
        if pos < 0:
            raise ValueError(f"no prior date before {date.date()}")
        return pos

    def _window_return(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, float]:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        entry_pos = self._prev_pos(self.dates, start)
        exit_pos = self._pos(self.dates, end)
        entry = self.close[entry_pos]
        exit_ = self.close[exit_pos]
        returns = np.divide(exit_, entry, out=np.ones_like(exit_), where=np.isfinite(entry) & (entry > 0))
        returns = returns - 1.0
        returns[~np.isfinite(returns)] = 0.0

        idx_entry = self._prev_pos(self.index_dates, start)
        idx_exit = self._pos(self.index_dates, end)
        benchmark = float(self.index_close[idx_exit] / self.index_close[idx_entry] - 1.0)
        return returns.astype(float), benchmark

    def read_weights(self, path: Path) -> tuple[np.ndarray, int]:
        weights = np.zeros(len(self.codes), dtype=float)
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                code = str(row["stock_code"]).zfill(6)
                idx = self.code_to_idx.get(code)
                if idx is not None:
                    weights[idx] += float(row["weight"])
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"{path} has non-positive total weight")
        weights /= total
        return weights, int((weights > 0).sum())

    def score(self, path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float | int]:
        returns, benchmark = self.window_returns[(start.date().isoformat(), end.date().isoformat())]
        weights, n_names = self.read_weights(path)
        portfolio = float(np.dot(weights, returns))
        return {
            "portfolio_return": portfolio,
            "benchmark_return": benchmark,
            "excess_return": portfolio - benchmark,
            "n_names": n_names,
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
    scorer: NumpyScorer,
) -> dict:
    commands = model_commands(as_of, out_dir)
    if model not in commands:
        raise ValueError(f"Unknown model {model}. Valid: {sorted(commands)}")
    cmd, submission_path = commands[model]
    status = "ok"
    error = ""
    if not (skip_existing and submission_path.exists()):
        try:
            subprocess.check_output(cmd, cwd=ROOT, env=env(), stderr=subprocess.STDOUT, text=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = exc.output[-4000:]
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = str(exc)
    if status == "ok":
        try:
            score = scorer.score(submission_path, start, end)
        except Exception as exc:
            status = "score_failed"
            error = str(exc)
            score = {"portfolio_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan, "n_names": np.nan}
    else:
        score = {"portfolio_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan, "n_names": np.nan}
    return {
        "model": model,
        "as_of": as_of.date().isoformat(),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "submission": str(submission_path.relative_to(ROOT)) if submission_path.exists() else "",
        "status": status,
        "error": error,
        **score,
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"] == "ok"].copy()
    summary = (
        ok.groupby("model")["excess_return"]
        .agg(["mean", "sum", "median", "min", "max", "count"])
        .sort_values(["mean", "min"], ascending=False)
        .reset_index()
    )
    summary["negative_windows"] = summary["model"].map(
        ok.assign(neg=ok["excess_return"] < 0).groupby("model")["neg"].sum()
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--models", nargs="+", default=["baseline_xgb", "baseline_guard_adaptive"])
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--as-of", nargs="*", default=None)
    parser.add_argument("--full-week-only", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "stage2_report" / "backtests" / "fast_numpy"))
    parser.add_argument("--summary-out", default=str(ROOT / "stage2_report" / "final_report_materials" / "stage2_fast_numpy_summary.csv"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices, columns=["date"])
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
    scorer = NumpyScorer(Path(args.prices), Path(args.index), windows)
    jobs = [
        {
            "model": model,
            "as_of": as_of,
            "start": start,
            "end": end,
            "out_dir": out_dir,
            "skip_existing": args.skip_existing,
            "timeout": args.timeout,
            "scorer": scorer,
        }
        for as_of, start, end in windows
        for model in args.models
    ]
    for as_of, start, end in windows:
        print(f"## window as_of={as_of.date()} score={start.date()}..{end.date()}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
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
            rows.append(row)
            if row["status"] == "ok":
                print(f"[ok] {row['model']} {row['as_of']} excess={row['excess_return']:.4f}", flush=True)
            else:
                print(f"[warn] {row['model']} {row['as_of']} {row['status']}: {row['error']}", flush=True)

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    summary_out = Path(args.summary_out)
    if not summary_out.is_absolute():
        summary_out = ROOT / summary_out
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    detail_out = summary_out.with_name(summary_out.stem + "_detail.csv")
    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    print("\n>> summary")
    print(summary.to_string(index=False))
    print(f">> wrote {summary_out}")
    print(f">> wrote {detail_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
