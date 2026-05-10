"""Generate the current best Stage2 portfolios for audit and submission.

By convention we always keep two snapshots:

* as_of=20260428: last fully known five-day evaluation window in local data,
  useful as a rolling sanity check;
* as_of=20260508: latest local data snapshot, useful as the current submission
  candidate until the next data refresh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from score_submission import score_window
from stage2_multiroute_consensus import generate_submission
from validate_submission import validate

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def next_window(trading_dates: list[pd.Timestamp], as_of: pd.Timestamp, horizon: int = 5) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if as_of not in trading_dates:
        raise ValueError(f"as_of {as_of.date()} is not a trading date")
    idx = trading_dates.index(as_of)
    if idx + horizon >= len(trading_dates):
        return None
    return trading_dates[idx + 1], trading_dates[idx + horizon]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--out-dir", default=str(ROOT / "submissions" / "stage2" / "current_best"))
    parser.add_argument("--as-of", nargs="+", default=["20260428", "20260508"])
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = [pd.Timestamp(d) for d in sorted(prices["date"].unique())]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in args.as_of:
        as_of = pd.to_datetime(value, format="%Y%m%d")
        stamp = as_of.strftime("%Y%m%d")
        sub, meta = generate_submission(prices, index_df, as_of)
        out_path = out_dir / f"stage2_multiroute_consensus_{stamp}.csv"
        meta_path = out_dir / f"stage2_multiroute_consensus_{stamp}_meta.csv"
        sub.to_csv(out_path, index=False)
        meta.to_csv(meta_path, index=False)
        errors = validate(out_path, DATA_DIR / "constituents.csv")
        if errors:
            raise RuntimeError(f"{out_path} failed validation: {errors}")

        row = {
            "as_of": as_of.date().isoformat(),
            "csv": str(out_path.relative_to(ROOT)),
            "meta": str(meta_path.relative_to(ROOT)),
            "n_names": len(sub),
            "max_weight": float(sub["weight"].max()),
            "final_route": meta["model"].iloc[0] if "model" in meta else "",
        }
        window = next_window(trading_dates, as_of)
        if window is not None:
            start, end = window
            weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
            score = score_window(weights, prices, index_df, start, end)
            row.update(
                {
                    "start": score["start"],
                    "end": score["end"],
                    "portfolio_return": score["portfolio_return"],
                    "benchmark_return": score["benchmark_return"],
                    "excess_return": score["excess_return"],
                }
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "stage2_multiroute_consensus_pair_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f">> wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
