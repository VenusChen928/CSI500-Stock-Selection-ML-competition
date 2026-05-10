"""Stage2 multi-route consensus portfolio.

This route is a controlled version of the "class ensemble" idea: collect
several as-of-safe route portfolios, aggregate their stock weights, then rebuild
a concentrated rank portfolio from the most agreed-upon names.

For historical validation windows, the script can reuse cached route CSVs from
prior no-leak backtests.  For a fresh as-of date, it falls back to live route
generation so the final submission does not require future data or realized
returns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from stage2_hybrid_gate import generate_hybrid_submission
from stage2_meta_portfolio_ensemble import COMPETITION_MAX_WEIGHT, generate_meta_ensemble

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE_DIR = ROOT / "submissions" / "stage2" / "backtests" / "archive_pre_12pct_20260510"
META_PROBE_DIR = ROOT / "submissions" / "stage2" / "backtests" / "probes" / "meta_portfolio_ensemble_adaptive_power_12w_20260510"
CURRENT_BEST_DIR = ROOT / "submissions" / "stage2" / "current_best"

BROAD_CACHE_DIRS = [
    "factor_route_12w_20260509",
    "factor_ic_dampen_12w_20260510",
    "factor_ic_gate_12w_20260510",
    "quality_gate_12w_20260509",
    "adaptive_cap_12w_20260509",
    "final_confidence_12w_20260509",
    "alpha_feature_route_v2_12w_20260510",
    "shapegrid_regularized_12w_20260509",
    "stage2_hybrid_gate_secondary_alpha_v2_12w_20260509",
    "stage2_hybrid_gate_regularized_top30_12w_20260509",
]

LIVE_HYBRID_VARIANTS: list[tuple[str, dict]] = [
    ("hybrid_default", {}),
    ("factor_ic_dampen", {"factor_ic_dampen": True}),
    ("factor_ic_filter", {"factor_ic_filter": True}),
    ("quality_features", {"tree_feature_set": "quality"}),
    ("core_features", {"tree_feature_set": "core"}),
    ("alpha_no_secondary", {"alpha_mode": "no_secondary"}),
    ("alpha_no_liquidity", {"alpha_mode": "no_liquidity"}),
    ("alpha_no_final", {"alpha_mode": "no_final"}),
    ("alpha_no_route", {"alpha_mode": "no_route"}),
    ("tree_top40", {"tree_top_k": 40}),
]


def read_weight_csv(path: Path) -> pd.Series | None:
    try:
        sub = pd.read_csv(path, dtype={"stock_code": str})
    except Exception:
        return None
    if "stock_code" not in sub.columns or "weight" not in sub.columns:
        return None
    sub["stock_code"] = sub["stock_code"].astype(str).str.zfill(6)
    weights = sub.groupby("stock_code")["weight"].sum().astype(float)
    weights = weights[weights > 0]
    if len(weights) < MIN_STOCKS or weights.sum() <= 0:
        return None
    return weights / weights.sum()


def current_meta_path(as_of: pd.Timestamp) -> Path | None:
    stamp = as_of.strftime("%Y%m%d")
    candidates = [
        META_PROBE_DIR / f"meta_portfolio_ensemble_{stamp}.csv",
        CURRENT_BEST_DIR / f"stage2_meta_portfolio_ensemble_{stamp}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def cached_candidates(as_of: pd.Timestamp, current_votes: int) -> tuple[list[tuple[str, pd.Series]], list[str]]:
    stamp = as_of.strftime("%Y%m%d")
    out: list[tuple[str, pd.Series]] = []
    sources: list[str] = []
    for directory in BROAD_CACHE_DIRS:
        root = ARCHIVE_DIR / directory
        if not root.exists():
            continue
        for path in root.glob(f"*{stamp}.csv"):
            name = path.name.lower()
            if any(skip in name for skip in ["meta", "shape", "feature", "policy", "summary", "detail", "table"]):
                continue
            weights = read_weight_csv(path)
            if weights is None:
                continue
            out.append((f"cache:{directory}:{path.name}", weights))
            sources.append(str(path.relative_to(ROOT)))

    meta_path = current_meta_path(as_of)
    if meta_path is not None:
        meta_weights = read_weight_csv(meta_path)
        if meta_weights is not None:
            for vote in range(current_votes):
                out.append((f"current_meta_vote_{vote + 1}", meta_weights))
            sources.append(str(meta_path.relative_to(ROOT)))
    return out, sources


def live_candidates(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    current_votes: int,
) -> tuple[list[tuple[str, pd.Series]], list[str]]:
    out: list[tuple[str, pd.Series]] = []
    sources: list[str] = []

    meta_sub, _ = generate_meta_ensemble(prices, index_df, as_of)
    meta_weights = meta_sub.set_index(meta_sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
    meta_weights = meta_weights / meta_weights.sum()
    for vote in range(current_votes):
        out.append((f"live_current_meta_vote_{vote + 1}", meta_weights))
    sources.append("live:stage2_meta_portfolio_ensemble")

    for name, kwargs in LIVE_HYBRID_VARIANTS:
        try:
            sub, _ = generate_hybrid_submission(prices, index_df, as_of, **kwargs)
        except Exception as exc:
            sources.append(f"live:{name}:skipped:{type(exc).__name__}")
            continue
        weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
        weights = weights[weights > 0]
        if len(weights) < MIN_STOCKS or weights.sum() <= 0:
            sources.append(f"live:{name}:skipped:too_few_names")
            continue
        out.append((f"live:{name}", weights / weights.sum()))
        sources.append(f"live:{name}")
    return out, sources


def cap_array(raw: np.ndarray, max_weight: float = COMPETITION_MAX_WEIGHT) -> np.ndarray:
    raw = raw.astype(float)
    if raw.sum() <= 0:
        raise ValueError("raw weights must sum positive")
    weights = np.zeros_like(raw, dtype=float)
    fixed = np.zeros(len(raw), dtype=bool)
    remaining_sum = 1.0
    for _ in range(len(raw) + 1):
        free = ~fixed
        if not free.any():
            break
        free_raw = raw[free]
        scaled = remaining_sum * free_raw / free_raw.sum()
        over = scaled > max_weight
        if not over.any():
            weights[free] = scaled
            break
        free_idx = np.flatnonzero(free)
        newly_fixed = free_idx[over]
        weights[newly_fixed] = max_weight
        fixed[newly_fixed] = True
        remaining_sum = 1.0 - float(weights[fixed].sum())
        if remaining_sum <= 0:
            break
    # Guard against tiny floating-point drift without renormalizing capped names
    # above the competition limit.
    weights = np.minimum(weights, max_weight)
    residual = 1.0 - float(weights.sum())
    if abs(residual) > 1e-12:
        free = weights < max_weight - 1e-12
        if free.any():
            weights[free] += residual * weights[free] / weights[free].sum()
    return weights


def consensus_portfolio(
    candidates: list[tuple[str, pd.Series]],
    top_k: int,
    rank_power: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(candidates) == 0:
        raise RuntimeError("No candidate portfolios available for consensus")
    aggregate = pd.Series(dtype=float)
    for _, weights in candidates:
        aggregate = aggregate.add(weights, fill_value=0.0)
    aggregate = (aggregate / len(candidates)).sort_values(ascending=False)
    effective_top_k = max(MIN_STOCKS, int(top_k))
    selected = aggregate.head(effective_top_k)
    if len(selected) < MIN_STOCKS:
        raise RuntimeError(f"Consensus produced only {len(selected)} names")
    raw = np.arange(len(selected), 0, -1, dtype=float) ** float(rank_power)
    weights = cap_array(raw, COMPETITION_MAX_WEIGHT)
    out = pd.DataFrame({"stock_code": selected.index.astype(str), "weight": weights})
    return out, aggregate


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    top_k: int = 30,
    rank_power: float = 128.0,
    current_votes: int = 5,
    cache_mode: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    candidates: list[tuple[str, pd.Series]]
    sources: list[str]
    source_mode = "cache"
    if cache_mode in {"auto", "cache"}:
        candidates, sources = cached_candidates(as_of, current_votes=current_votes)
        non_meta = [name for name, _ in candidates if not name.startswith("current_meta_vote")]
        if cache_mode == "cache" and len(candidates) == 0:
            raise RuntimeError(f"No cached candidates found for {as_of.date()}")
        if cache_mode == "auto" and len(non_meta) < 3:
            candidates, sources = live_candidates(prices, index_df, as_of, current_votes=current_votes)
            source_mode = "live"
    else:
        candidates, sources = live_candidates(prices, index_df, as_of, current_votes=current_votes)
        source_mode = "live"

    sub, aggregate = consensus_portfolio(candidates, top_k=top_k, rank_power=rank_power)
    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "model": "stage2_multiroute_consensus",
                "source_mode": source_mode,
                "candidate_count": len(candidates),
                "top_k": max(MIN_STOCKS, int(top_k)),
                "rank_power": float(rank_power),
                "current_votes": int(current_votes),
                "n_names": len(sub),
                "max_weight": float(sub["weight"].max()),
                "aggregate_top10": " | ".join(aggregate.head(10).index.astype(str).tolist()),
                "sources": " | ".join(sources[:80]),
            }
        ]
    )
    return sub, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rank-power", type=float, default=128.0)
    parser.add_argument("--current-votes", type=int, default=5)
    parser.add_argument("--cache-mode", choices=["auto", "cache", "live"], default="auto")
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_multiroute_consensus.csv")
    parser.add_argument("--meta-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, meta = generate_submission(
        prices,
        index_df,
        as_of,
        top_k=args.top_k,
        rank_power=args.rank_power,
        current_votes=args.current_votes,
        cache_mode=args.cache_mode,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)

    print(f">> model=stage2_multiroute_consensus as_of={as_of.date()}")
    print(meta.drop(columns=["sources"], errors="ignore").to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
