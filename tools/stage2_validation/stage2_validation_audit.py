"""Stage2 validation audit utilities.

This script checks two things that are easy to get wrong in this project:

1. as-of leakage: a cached portfolio generated with the full local dataset must
   match a portfolio generated after physically truncating prices/index to
   date <= as_of;
2. canonical scoring: every audited portfolio is scored through the same
   score_window logic as score_submission.py.
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

from baseline_xgboost import FORWARD_HORIZON
from score_submission import score_window
from stage2_hybrid_gate import generate_hybrid_submission
from stage2_meta_portfolio_ensemble import generate_meta_ensemble

DATA_DIR = ROOT / "data"


def rolling_windows(trading_dates: np.ndarray, windows: int, step: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    out = []
    max_asof_idx = len(trading_dates) - FORWARD_HORIZON - 1
    idx = max_asof_idx - step * (windows - 1)
    while idx <= max_asof_idx:
        if idx >= 120:
            as_of = pd.Timestamp(trading_dates[idx])
            out.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])))
        idx += step
    return out


def explicit_windows(trading_dates: np.ndarray, as_of_values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    out = []
    for value in as_of_values:
        as_of = pd.to_datetime(value, format="%Y%m%d")
        if as_of not in dates:
            raise ValueError(f"as_of {value} is not in trading dates")
        idx = dates.index(as_of)
        if idx + FORWARD_HORIZON >= len(dates):
            raise ValueError(f"as_of {value} does not have {FORWARD_HORIZON} future trading days")
        out.append((as_of, dates[idx + 1], dates[idx + FORWARD_HORIZON]))
    return out


def normalize_submission(sub: pd.DataFrame) -> pd.Series:
    df = sub.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df.groupby("stock_code")["weight"].sum().sort_index().astype(float)


def compare_submissions(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    a = normalize_submission(left)
    b = normalize_submission(right)
    all_codes = sorted(set(a.index) | set(b.index))
    av = a.reindex(all_codes).fillna(0.0)
    bv = b.reindex(all_codes).fillna(0.0)
    diff = (av - bv).abs()
    return {
        "same_codes": set(a.index) == set(b.index),
        "max_abs_weight_diff": float(diff.max()) if len(diff) else 0.0,
        "sum_abs_weight_diff": float(diff.sum()) if len(diff) else 0.0,
        "left_names": int((a > 0).sum()),
        "right_names": int((b > 0).sum()),
    }


def generate_submission_for_audit(
    generator: str,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if generator == "meta":
        return generate_meta_ensemble(prices, index_df, as_of)
    return generate_hybrid_submission(
        prices,
        index_df,
        as_of,
        alpha_mode=args.alpha_mode,
        factor_ic_filter=args.factor_ic_filter,
        factor_ic_dampen=args.factor_ic_dampen,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--cached-dir", default="")
    parser.add_argument("--cached-prefix", default="hybrid_gate")
    parser.add_argument("--generator", choices=["hybrid", "meta"], default="hybrid")
    parser.add_argument("--alpha-mode", default="full", choices=["full", "none", "no_regime", "no_liquidity", "no_secondary", "no_route", "no_final"])
    parser.add_argument("--factor-ic-filter", action="store_true")
    parser.add_argument("--factor-ic-dampen", action="store_true")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--as-of", nargs="*", default=None)
    parser.add_argument("--out", default="stage2_report/final_report_materials/stage2_validation_audit.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    windows = explicit_windows(trading_dates, args.as_of) if args.as_of else rolling_windows(trading_dates, args.windows, args.step)

    cached_dir = Path(args.cached_dir) if args.cached_dir else None
    rows = []
    for as_of, start, end in windows:
        stamp = as_of.strftime("%Y%m%d")
        if cached_dir:
            cached_path = cached_dir / f"{args.cached_prefix}_{stamp}.csv"
        else:
            cached_path = Path("")

        if cached_dir and cached_path.exists():
            full_sub = pd.read_csv(cached_path, dtype={"stock_code": str})
            source = str(cached_path)
        else:
            full_sub, _ = generate_submission_for_audit(args.generator, prices, index_df, as_of, args)
            source = "generated_from_full_data"

        trunc_prices = prices[prices["date"] <= as_of].copy()
        trunc_index = index_df[index_df["date"] <= as_of].copy()
        trunc_sub, trunc_meta = generate_submission_for_audit(args.generator, trunc_prices, trunc_index, as_of, args)

        cmp = compare_submissions(full_sub, trunc_sub)
        weights = normalize_submission(full_sub)
        score = score_window(weights, prices, index_df, start, end)
        rows.append(
            {
                "as_of": as_of.date().isoformat(),
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "cached_source": source,
                "future_price_rows_available": int((prices["date"] > as_of).sum()),
                "future_index_rows_available": int((index_df["date"] > as_of).sum()),
                "final_route": trunc_meta["final_route"].iloc[0] if "final_route" in trunc_meta else "",
                **cmp,
                "leak_check_pass": bool(cmp["same_codes"] and cmp["max_abs_weight_diff"] < 1e-12),
                "portfolio_return": score["portfolio_return"],
                "benchmark_return": score["benchmark_return"],
                "excess_return": score["excess_return"],
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f">> wrote {out_path}")
    if not out["leak_check_pass"].all():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
