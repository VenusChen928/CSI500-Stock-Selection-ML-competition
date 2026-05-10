"""Leakage audit for active stage2 routes.

The key dynamic check is:

    generate(full data, as_of=t) == generate(data truncated to <=t, as_of=t)

If a route accidentally reads future rows, these two outputs will diverge for a
historical ``as_of``.  Caches are disabled in this audit unless explicitly
requested, so the check also verifies that the live production path is
as-of-safe.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_weekly_alpha_overlay import generate_submission as generate_alpha
from stage2_baseline_guard_ensemble import generate_submission as generate_baseline_guard
from stage2_weekly_consensus_ensemble import generate_submission as generate_consensus
from stage2_weekly_cycle_tree import PortfolioShape, generate_submission as generate_cycle_tree

DATA_DIR = ROOT / "data"


def weight_series(sub: pd.DataFrame) -> pd.Series:
    out = sub.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    weights = out.groupby("stock_code")["weight"].sum().astype(float)
    return weights / weights.sum()


def compare_weights(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    index = sorted(set(left.index) | set(right.index))
    l = left.reindex(index).fillna(0.0)
    r = right.reindex(index).fillna(0.0)
    diff = (l - r).abs()
    return {
        "n_left": int((l > 0).sum()),
        "n_right": int((r > 0).sum()),
        "n_union": int(len(index)),
        "max_abs_diff": float(diff.max()),
        "l1_diff": float(diff.sum()),
        "changed_names": int((diff > 1e-10).sum()),
    }


def generate_model(
    model: str,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model == "weekly_alpha_auto":
        sub, meta, _ = generate_alpha(prices, index_df, as_of, mode="auto", meta_cache_dir=None)
        return sub, meta
    if model == "weekly_alpha_floor":
        sub, meta, _ = generate_alpha(prices, index_df, as_of, mode="floor", meta_cache_dir=None)
        return sub, meta
    if model == "weekly_cycle_tree":
        shape = PortfolioShape(
            top_k=40,
            score_temperature=0.80,
            rank_power=3.0,
            score_rank_blend=0.60,
            max_weight=0.08,
        )
        sub, meta, _ = generate_cycle_tree(
            prices,
            index_df,
            as_of,
            shape=shape,
            corr_threshold=0.90,
            half_life_days=180.0,
            fullweek_boost=0.20,
            model_set="lgb_xgb",
            alpha_blend=0.25,
        )
        return sub, meta
    if model == "weekly_consensus":
        sub, meta = generate_consensus(prices, index_df, as_of, cache_dir=None)
        return sub, meta
    if model == "baseline_guard_adaptive":
        sub, meta = generate_baseline_guard(prices, index_df, as_of, baseline_top_k=0, cache_dir=None)
        return sub, meta
    raise ValueError(f"unknown model {model}")


def static_source_scan() -> pd.DataFrame:
    rows = []
    active_files = [
        ROOT / "stage2_baseline_guard_ensemble.py",
        ROOT / "stage2_weekly_consensus_ensemble.py",
        ROOT / "stage2_weekly_cycle_tree.py",
        ROOT / "stage2_weekly_alpha_overlay.py",
        ROOT / "stage2_meta_portfolio_ensemble.py",
        ROOT / "stage2_hybrid_gate.py",
        ROOT / "stage2_tree_consensus.py",
        ROOT / "features.py",
    ]
    patterns = [
        "score_submission",
        "submissions/stage2/reports",
        "current_best",
        "archive_pre",
        "cache_dir",
        "meta_cache_dir",
        "shift(-",
        "target_excess",
    ]
    for path in active_files:
        text = path.read_text()
        for pattern in patterns:
            rows.append(
                {
                    "file": path.name,
                    "pattern": pattern,
                    "count": text.count(pattern),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", nargs="+", default=["20260410", "20260417"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["weekly_alpha_auto", "weekly_alpha_floor", "weekly_cycle_tree", "weekly_consensus", "baseline_guard_adaptive"],
    )
    parser.add_argument("--out", default="submissions/stage2/final_report_materials/05_final_leakage_audit_dynamic.csv")
    parser.add_argument("--static-out", default="submissions/stage2/final_report_materials/06_final_leakage_audit_static_scan.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    rows = []
    for as_of_text in args.as_of:
        as_of = pd.to_datetime(as_of_text, format="%Y%m%d")
        truncated_prices = prices[prices["date"] <= as_of].copy()
        truncated_index = index_df[index_df["date"] <= as_of].copy()
        for model in args.models:
            print(f">> audit model={model} as_of={as_of.date()}", flush=True)
            full_sub, full_meta = generate_model(model, prices, index_df, as_of)
            trunc_sub, trunc_meta = generate_model(model, truncated_prices, truncated_index, as_of)
            row = {
                "as_of": as_of.date().isoformat(),
                "model": model,
                **compare_weights(weight_series(full_sub), weight_series(trunc_sub)),
                "full_meta": full_meta.to_json(orient="records"),
                "truncated_meta": trunc_meta.to_json(orient="records"),
            }
            row["pass"] = bool(row["max_abs_diff"] <= 1e-10 and row["l1_diff"] <= 1e-8)
            rows.append(row)

    audit = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    static = static_source_scan()
    static_out = Path(args.static_out)
    static.to_csv(static_out, index=False)
    print("\n>> dynamic leakage audit")
    print(audit[["as_of", "model", "pass", "max_abs_diff", "l1_diff", "changed_names"]].to_string(index=False))
    print("\n>> static pattern scan")
    print(static.pivot(index="file", columns="pattern", values="count").fillna(0).astype(int).to_string())
    print(f">> wrote {out}")
    print(f">> wrote {static_out}")
    if not audit["pass"].all():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
