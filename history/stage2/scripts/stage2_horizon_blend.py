"""Blend short-horizon and stage2-horizon tree consensus portfolios.

The 3-day target is useful for short bursts such as the phase-1 window, while
the 5-day target is the actual stage2 objective.  This script blends both
signals at the portfolio level so we can test whether short-term momentum adds
alpha without letting it dominate the five-day risk profile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from features import FEATURE_COLUMNS, TARGET_3D_COLUMN, TARGET_COLUMN
from stage2_tree_consensus import ConsensusShape, _cap, generate_tree_consensus

DATA_DIR = Path(__file__).parent / "data"


def _weights(sub: pd.DataFrame) -> pd.Series:
    out = sub.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    return out.set_index("stock_code")["weight"].astype(float)


def blend_weights(long_w: pd.Series, short_w: pd.Series, short_weight: float, max_weight: float) -> pd.Series:
    codes = sorted(set(long_w.index) | set(short_w.index))
    raw = (
        (1.0 - short_weight) * long_w.reindex(codes).fillna(0.0)
        + short_weight * short_w.reindex(codes).fillna(0.0)
    )
    raw = raw[raw > 0]
    if len(raw) < MIN_STOCKS:
        raise ValueError(f"blend has only {len(raw)} names")
    return _cap(raw / raw.sum(), max_weight=max_weight)


def generate_horizon_blend(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    short_weight: float,
    max_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_shape = ConsensusShape(top_k=30, alpha_xgb=0.05, rank_power=1.6, equal_mix=0.0, max_weight=0.04)
    short_shape = ConsensusShape(top_k=30, alpha_xgb=0.0, rank_power=1.6, equal_mix=0.0, max_weight=0.04)

    long_sub, long_meta = generate_tree_consensus(
        prices,
        index_df,
        as_of,
        long_shape,
        defensive_equal_gate=True,
        features=FEATURE_COLUMNS,
        half_life=120,
        weight_floor=0.5,
        adaptive_time_decay=True,
        target_column=TARGET_COLUMN,
        target_horizon=FORWARD_HORIZON,
        shape_horizon=FORWARD_HORIZON,
    )
    short_sub, short_meta = generate_tree_consensus(
        prices,
        index_df,
        as_of,
        short_shape,
        defensive_equal_gate=True,
        features=FEATURE_COLUMNS,
        half_life=120,
        weight_floor=0.5,
        adaptive_time_decay=True,
        target_column=TARGET_3D_COLUMN,
        target_horizon=3,
        shape_horizon=3,
    )
    weights = blend_weights(_weights(long_sub), _weights(short_sub), short_weight=short_weight, max_weight=max_weight)
    meta = pd.DataFrame(
        [
            {
                "short_weight": short_weight,
                "max_weight": max_weight,
                "n_names": len(weights),
                "long_n": len(long_sub),
                "short_n": len(short_sub),
                "overlap_n": len(set(_weights(long_sub).index) & set(_weights(short_sub).index)),
                "long_route": long_meta["route"].iloc[0],
                "short_route": short_meta["route"].iloc[0],
                "long_decay_reason": long_meta["decay_reason"].iloc[0],
                "short_decay_reason": short_meta["decay_reason"].iloc[0],
            }
        ]
    )
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values}), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--short-weight", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=0.04)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_horizon_blend.csv")
    parser.add_argument("--meta-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, meta = generate_horizon_blend(
        prices,
        index_df,
        as_of=as_of,
        short_weight=args.short_weight,
        max_weight=args.max_weight,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    print(f">> model=stage2_horizon_blend as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
