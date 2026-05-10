"""Stage2 meta-portfolio ensemble.

Generate several as-of-safe hybrid routes, aggregate their picked-stock weights,
then rebuild a concentrated rank portfolio from the consensus list.  This is
inspired by the class ensemble note, but uses only routes we can generate
locally from data <= as_of.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from stage2_hybrid_gate import factor_route_gate, generate_hybrid_submission
from stage2_tree_consensus import defensive_guard

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COMPETITION_MAX_WEIGHT = 0.10


def cap_weights(weights: pd.Series, max_weight: float = COMPETITION_MAX_WEIGHT) -> pd.Series:
    w = weights[weights > 0].astype(float).copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} names")
    w = w / w.sum()
    for _ in range(100):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def weight_series(sub: pd.DataFrame) -> pd.Series:
    df = sub.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    out = df.groupby("stock_code")["weight"].sum().astype(float)
    return out / out.sum()


def route_variants() -> list[tuple[str, dict]]:
    return [
        ("default", {}),
        ("factor_ic_dampen", {"factor_ic_dampen": True}),
        ("factor_ic_filter", {"factor_ic_filter": True}),
    ]


def meta_passthrough_reason(decision, factor_spec: tuple[str, str, int, float, float] | None) -> str | None:
    """Avoid meta concentration in regimes where it repeatedly reduced excess.

    This is intentionally narrow.  The meta layer helps most windows by
    concentrating consensus names, but two weak/flat tape shapes were hurt by
    that extra concentration in rolling validation.  In those cases the
    original hybrid route is already the safer confidence estimate.
    """
    if factor_spec is not None:
        factor, direction, *_ = factor_spec
        if (
            factor == "turnover_ma_20d"
            and direction == "low"
            and decision.idx_ret_5d < -0.03
            and -0.02 <= decision.idx_ret_20d <= 0.02
            and decision.breadth_ret_5d_pos < 0.30
        ):
            return "passthrough_defensive_low_turnover_factor"
    if (
        factor_spec is None
        and -0.02 <= decision.idx_ret_5d <= 0.0
        and -0.02 <= decision.idx_ret_20d <= 0.02
        and 0.40 <= decision.breadth_ret_5d_pos <= 0.60
    ):
        return "passthrough_weak_flat_tree_tape"
    return None


def adaptive_rank_power(decision, factor_spec: tuple[str, str, int, float, float] | None, base_power: float) -> float:
    """Choose concentration strength from validation-stable regime shape.

    The meta ensemble's selected names are already a confidence-ranked list.
    Rank power controls how much of that confidence is converted into weight.
    Fragile pullback-style factor routes keep a moderate power, while strong
    consensus / high-dispersion tapes can safely use a more concentrated tail.
    """
    if factor_spec is not None:
        factor, direction, *_ = factor_spec
        if factor == "intraday_mean_5d" and direction == "low":
            return max(base_power, 3.0)
        if factor == "obv_20d" and direction == "high":
            return max(base_power, 6.0)
        if factor == "overnight_ret" and direction == "low":
            return max(base_power, 8.0)
        if factor == "downside_vol_20d" and direction == "high":
            return max(base_power, 12.0)
        return base_power

    if decision.idx_ret_5d < 0 and abs(decision.idx_ret_20d) <= 0.02 and decision.breadth_ret_5d_pos < 0.40:
        return max(base_power, 4.8)
    if (
        abs(decision.idx_ret_20d) >= 0.04
        or abs(decision.idx_ret_5d) >= 0.04
        or decision.breadth_ret_5d_pos >= 0.75
        or decision.breadth_ret_5d_pos <= 0.15
    ):
        return max(base_power, 20.0)
    return base_power


def generate_meta_ensemble(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    top_k: int = 30,
    rank_power: float = 2.5,
    mix_agg: float = 0.0,
    use_passthrough_guards: bool = True,
    use_adaptive_rank_power: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    decision = defensive_guard(
        prices[prices["date"] <= as_of].copy(),
        index_df[index_df["date"] <= as_of].copy(),
        as_of,
    )
    factor_spec = factor_route_gate(decision)
    if use_passthrough_guards:
        passthrough = meta_passthrough_reason(decision, factor_spec)
        if passthrough is not None:
            sub, meta = generate_hybrid_submission(prices, index_df, as_of)
            meta = meta.copy()
            meta["as_of"] = as_of.date().isoformat()
            meta["meta_layer"] = "passthrough_guard"
            meta["meta_passthrough_reason"] = passthrough
            meta["top_k"] = top_k
            meta["rank_power"] = rank_power
            meta["base_rank_power"] = rank_power
            meta["adaptive_rank_power"] = use_adaptive_rank_power
            meta["mix_agg"] = mix_agg
            return sub, meta
    variants = route_variants() if factor_spec is not None else [("default", {})]
    aggregate = pd.Series(dtype=float)
    meta_rows = []
    for name, kwargs in variants:
        sub, meta = generate_hybrid_submission(prices, index_df, as_of, **kwargs)
        w = weight_series(sub)
        aggregate = aggregate.add(w, fill_value=0.0)
        row = meta.iloc[0].to_dict()
        meta_rows.append(
            {
                "variant": name,
                "variant_names": int((w > 0).sum()),
                "variant_max_weight": float(w.max()),
                "variant_route": row.get("final_route", ""),
                "variant_reason": row.get("route_reason", ""),
            }
        )
    aggregate = aggregate / len(variants)
    chosen = aggregate.sort_values(ascending=False).head(top_k)
    effective_rank_power = (
        adaptive_rank_power(decision, factor_spec, rank_power)
        if use_adaptive_rank_power
        else rank_power
    )
    ranks = pd.Series(np.arange(len(chosen), 0, -1, dtype=float) ** effective_rank_power, index=chosen.index)
    raw = mix_agg * chosen / chosen.sum() + (1.0 - mix_agg) * ranks / ranks.sum()
    weights = cap_weights(raw, COMPETITION_MAX_WEIGHT)
    sub = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "final_route": "meta_portfolio_ensemble",
                "n_variants": len(variants),
                "top_k": top_k,
                "rank_power": effective_rank_power,
                "base_rank_power": rank_power,
                "adaptive_rank_power": use_adaptive_rank_power,
                "mix_agg": mix_agg,
                "meta_layer": "portfolio_ensemble",
                "meta_passthrough_reason": "",
                "n_names": len(sub),
                "max_weight": float(sub["weight"].max()),
                "variant_routes": " | ".join(f"{r['variant']}={r['variant_route']}" for r in meta_rows),
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
    parser.add_argument("--rank-power", type=float, default=2.5)
    parser.add_argument("--mix-agg", type=float, default=0.0)
    parser.add_argument(
        "--disable-passthrough-guards",
        action="store_true",
        help="Always apply the meta concentration layer, even in weak/flat regimes where validation preferred the raw hybrid route.",
    )
    parser.add_argument(
        "--disable-adaptive-rank-power",
        action="store_true",
        help="Use the supplied --rank-power uniformly instead of regime-specific confidence concentration.",
    )
    parser.add_argument("--out", default="stage2_report/route_outputs/stage2_meta_portfolio_ensemble.csv")
    parser.add_argument("--meta-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, meta = generate_meta_ensemble(
        prices,
        index_df,
        as_of,
        top_k=args.top_k,
        rank_power=args.rank_power,
        mix_agg=args.mix_agg,
        use_passthrough_guards=not args.disable_passthrough_guards,
        use_adaptive_rank_power=not args.disable_adaptive_rank_power,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    print(f">> model=stage2_meta_portfolio_ensemble as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
