"""Regime-gated Stage2 portfolio.

Most five-day windows are handled better by the regularized tree consensus.
However, after fixing the LSTM target-index compatibility issue, historical
checks showed two regimes where the sequence model adds real upside:

* post-rally with weak medium-term support, where the tree route is too muted;
* broad capitulation, where LSTM's sequence reversal signal beats tree ranking.

This script keeps the tree consensus as the default and only trains/uses the
LSTM route when the as-of regime matches one of those rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON
from lstm_rank_weight import fit_lstm, generate_submission as generate_lstm_submission
from stage2_tree_consensus import (
    ConsensusShape,
    defensive_guard,
    generate_tree_consensus,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def lstm_gate(decision) -> tuple[bool, str]:
    capitulation = (
        decision.idx_ret_5d < -0.07
        and decision.idx_ret_20d < -0.07
        and decision.breadth_ret_5d_pos < 0.15
    )
    post_rally_no_medium_support = (
        decision.idx_ret_5d > 0.03
        and 0.0 < decision.idx_ret_20d < 0.03
        and 0.55 < decision.breadth_ret_5d_pos < 0.75
    )
    if capitulation:
        return True, "lstm_capitulation_reversal"
    if post_rally_no_medium_support:
        return True, "lstm_post_rally_sequence"
    return False, "tree_default"


def generate_gated_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    force_route: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = defensive_guard(prices, index_df, as_of)
    use_lstm, route_reason = lstm_gate(decision)
    if force_route == "tree":
        use_lstm = False
        route_reason = "forced_tree"
    elif force_route == "lstm":
        use_lstm = True
        route_reason = "forced_lstm"

    if use_lstm:
        fit = fit_lstm(prices, index_df, as_of, policy_horizon=FORWARD_HORIZON, target_horizon=FORWARD_HORIZON)
        sub = generate_lstm_submission(fit, as_of)
        meta = pd.DataFrame(
            [
                {
                    **decision.__dict__,
                    "final_route": "lstm_rank_weight",
                    "route_reason": route_reason,
                    "lstm_top_k": fit["policy"].top_k,
                    "lstm_temperature": fit["policy"].temperature,
                    "lstm_rank_blend": fit["policy"].rank_blend,
                    "n_names": len(sub),
                    "max_weight": float(sub["weight"].max()),
                }
            ]
        )
        return sub, meta

    shape = ConsensusShape(top_k=30, alpha_xgb=0.05, rank_power=1.6, equal_mix=0.0, max_weight=0.04)
    sub, tree_meta = generate_tree_consensus(
        prices,
        index_df,
        as_of=as_of,
        shape=shape,
        defensive_equal_gate=True,
        half_life=120,
        weight_floor=0.5,
        adaptive_time_decay=True,
        reweight_factor="drawdown_20d",
        reweight_direction="low",
        reweight_gamma=1.5,
        reweight_power=1.0,
        reweight_gate="medium_move",
    )
    tree_meta = tree_meta.copy()
    tree_meta["final_route"] = "tree_consensus_drawdown_overlay"
    tree_meta["route_reason"] = route_reason
    tree_meta["n_names"] = len(sub)
    return sub, tree_meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_tree_lstm_gate.csv")
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--force-route", choices=["auto", "tree", "lstm"], default="auto")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, meta = generate_gated_submission(prices, index_df, as_of, force_route=args.force_route)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)

    print(f">> model=stage2_tree_lstm_gate as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
