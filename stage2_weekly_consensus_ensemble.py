"""Unified weekly-cycle consensus ensemble for Stage2.

The validation lesson from the complete-week rebase is that no single route is
reliable enough:

* ``weekly_alpha_auto`` catches broad reversal / rebound weeks.
* ``weekly_alpha_floor`` is a slightly more defensive gap-flow route.
* ``weekly_cycle_tree`` adds learned weekly-calendar features, but is only
  useful as a vote, not as a standalone replacement.

This script aggregates those three as-of-safe portfolios and rebuilds a
non-equal-weight top-30 consensus portfolio.  It keeps one unified 5-day
workflow; complete-week behavior is handled by features and voting, not by a
separate full-week-only model family.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import MIN_STOCKS
from stage2_weekly_alpha_overlay import generate_submission as generate_alpha_submission
from stage2_weekly_cycle_tree import PortfolioShape, generate_submission as generate_cycle_submission

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAX_WEIGHT = 0.095


def read_submission(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        sub = pd.read_csv(path, dtype={"stock_code": str})
    except Exception:
        return None
    if "stock_code" not in sub or "weight" not in sub:
        return None
    sub["stock_code"] = sub["stock_code"].astype(str).str.zfill(6)
    weights = sub.groupby("stock_code")["weight"].sum().astype(float)
    weights = weights[weights > 0]
    if len(weights) < MIN_STOCKS or weights.sum() <= 0:
        return None
    return weights / weights.sum()


def cap_weights(raw: pd.Series, max_weight: float) -> pd.Series:
    weights = raw[raw > 0].astype(float).copy()
    if len(weights) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} names")
    weights = weights / weights.sum()
    for _ in range(100):
        over = weights > max_weight
        if not over.any():
            break
        excess = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        free = ~over
        if not free.any() or weights[free].sum() <= 0:
            break
        weights[free] += excess * weights[free] / weights[free].sum()
    return weights / weights.sum()


def ranked_consensus(
    candidates: list[tuple[str, pd.Series]],
    *,
    top_k: int,
    rank_power: float,
    max_weight: float,
) -> tuple[pd.DataFrame, pd.Series]:
    aggregate = pd.Series(dtype=float)
    for _, weights in candidates:
        aggregate = aggregate.add(weights, fill_value=0.0)
    aggregate = aggregate / len(candidates)
    selected = aggregate.sort_values(ascending=False).head(max(MIN_STOCKS, int(top_k)))
    raw = pd.Series(np.arange(len(selected), 0, -1, dtype=float) ** rank_power, index=selected.index)
    weights = cap_weights(raw, max_weight=max_weight)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values}), aggregate.sort_values(ascending=False)


def cached_candidate(cache_dir: Path | None, model: str, as_of: pd.Timestamp) -> pd.Series | None:
    if cache_dir is None:
        return None
    stamp = as_of.strftime("%Y%m%d")
    return read_submission(cache_dir / f"{model}_{stamp}.csv")


def live_candidates(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    cache_dir: Path | None,
) -> tuple[list[tuple[str, pd.Series]], list[str]]:
    candidates: list[tuple[str, pd.Series]] = []
    sources: list[str] = []

    for model_name, mode in [("weekly_alpha_auto", "auto"), ("weekly_alpha_floor", "floor")]:
        weights = cached_candidate(cache_dir, model_name, as_of)
        if weights is None:
            sub, _, _ = generate_alpha_submission(prices, index_df, as_of, mode=mode, meta_cache_dir=None)
            weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
            sources.append(f"live:{model_name}")
        else:
            sources.append(f"cache:{model_name}")
        candidates.append((model_name, weights / weights.sum()))

    weights = cached_candidate(cache_dir, "weekly_cycle_tree", as_of)
    if weights is None:
        shape = PortfolioShape(
            top_k=40,
            score_temperature=0.80,
            rank_power=3.0,
            score_rank_blend=0.60,
            max_weight=0.08,
        )
        sub, _, _ = generate_cycle_submission(
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
        weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
        sources.append("live:weekly_cycle_tree")
    else:
        sources.append("cache:weekly_cycle_tree")
    candidates.append(("weekly_cycle_tree", weights / weights.sum()))

    return candidates, sources


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    cache_dir: Path | None = None,
    top_k: int = 30,
    rank_power: float = 6.0,
    max_weight: float = MAX_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    candidates, sources = live_candidates(prices, index_df, as_of, cache_dir)
    sub, aggregate = ranked_consensus(candidates, top_k=top_k, rank_power=rank_power, max_weight=max_weight)
    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "model": "stage2_weekly_consensus_ensemble",
                "candidate_count": len(candidates),
                "top_k": top_k,
                "rank_power": rank_power,
                "max_weight": max_weight,
                "n_names": len(sub),
                "max_observed_weight": float(sub["weight"].max()),
                "effective_n": float(1.0 / np.square(sub["weight"].to_numpy()).sum()),
                "aggregate_top10": " | ".join(aggregate.head(10).index.astype(str).tolist()),
                "sources": " | ".join(sources),
            }
        ]
    )
    return sub, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rank-power", type=float, default=6.0)
    parser.add_argument("--max-weight", type=float, default=MAX_WEIGHT)
    parser.add_argument("--out", default="stage2_report/route_outputs/stage2_weekly_consensus_ensemble.csv")
    parser.add_argument("--meta-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    sub, meta = generate_submission(
        prices,
        index_df,
        as_of,
        cache_dir=cache_dir,
        top_k=args.top_k,
        rank_power=args.rank_power,
        max_weight=args.max_weight,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    print(f">> model=stage2_weekly_consensus_ensemble as_of={as_of.date()}")
    print(meta.drop(columns=["sources"], errors="ignore").to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
