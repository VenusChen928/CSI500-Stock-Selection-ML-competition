"""
Gated production portfolio: layered LSTM when validation evidence is strong,
otherwise tuned XGBoost.

The 5-window backtest showed layered LSTM has the best upside, but can inherit
LSTM's large drawdowns.  This script keeps the layered LSTM for windows where
its validation layer table is positive enough, and falls back to tuned XGB when
the sequence signal is weak.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import lstm_rank_weight as lstm
from features import FORWARD_HORIZON
from layered_lstm_portfolio import (
    generate_layered_submission,
    select_layer,
)
from open_data_features import add_open_data_features
from tuned_xgboost_portfolio import fit_tuned_model, generate_submission as generate_tuned_submission

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_LAYER_MEAN_THRESHOLD = 0.010
DEFAULT_LAYER_MIN_THRESHOLD = -0.020
DEFAULT_FALLBACK_MEAN_EDGE = 0.002
DEFAULT_FALLBACK_MIN_EDGE = 0.002


@dataclass(frozen=True)
class GateDecision:
    selected: str
    reason: str
    best_layer_mean: float
    best_layer_min: float
    fallback_mean: float | None = None
    fallback_min: float | None = None


def decide_gate(
    layer_table: pd.DataFrame,
    fallback_table: pd.DataFrame | None = None,
    layer_mean_threshold: float = DEFAULT_LAYER_MEAN_THRESHOLD,
    layer_min_threshold: float = DEFAULT_LAYER_MIN_THRESHOLD,
    fallback_mean_edge: float = DEFAULT_FALLBACK_MEAN_EDGE,
    fallback_min_edge: float = DEFAULT_FALLBACK_MIN_EDGE,
) -> GateDecision:
    """Decide whether the layered sequence signal is strong enough to use.

    The gate is intentionally simple and auditable: the best layered policy must
    clear both a validation mean hurdle and a validation worst-window hurdle.
    Tuned XGB is used as the low-variance fallback when the sequence signal is
    weak or unstable.
    """
    best = layer_table.iloc[0]
    best_mean = float(best["mean_excess_return"])
    best_min = float(best["min_excess_return"])
    fallback_mean = None
    fallback_min = None
    clears_fallback = True
    fallback_reason = ""
    if fallback_table is not None and not fallback_table.empty:
        fallback = fallback_table.iloc[0]
        fallback_mean = float(fallback["mean_excess_return"])
        fallback_min = float(fallback["min_excess_return"])
        clears_fallback = (
            best_mean >= fallback_mean + fallback_mean_edge
            and best_min >= fallback_min + fallback_min_edge
        )
        fallback_reason = (
            f", fallback_mean={fallback_mean:.6f}, fallback_min={fallback_min:.6f}, "
            f"required_mean_edge={fallback_mean_edge:.6f}, "
            f"required_min_edge={fallback_min_edge:.6f}"
        )
    if best_mean >= layer_mean_threshold and best_min >= layer_min_threshold and clears_fallback:
        return GateDecision(
            selected="layered_lstm",
            reason=(
                f"layer_mean {best_mean:.6f} >= {layer_mean_threshold:.6f} "
                f"and layer_min {best_min:.6f} >= {layer_min_threshold:.6f}"
                f"{fallback_reason}"
            ),
            best_layer_mean=best_mean,
            best_layer_min=best_min,
            fallback_mean=fallback_mean,
            fallback_min=fallback_min,
        )
    reasons = []
    if best_mean < layer_mean_threshold:
        reasons.append(f"layer_mean {best_mean:.6f} < {layer_mean_threshold:.6f}")
    if best_min < layer_min_threshold:
        reasons.append(f"layer_min {best_min:.6f} < {layer_min_threshold:.6f}")
    if not clears_fallback:
        reasons.append(f"layer does not clear fallback edge{fallback_reason}")
    return GateDecision(
        selected="tuned_xgb_fallback",
        reason="; ".join(reasons),
        best_layer_mean=best_mean,
        best_layer_min=best_min,
        fallback_mean=fallback_mean,
        fallback_min=fallback_min,
    )


def generate_gated_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    open_dir: str | Path,
    layer_mean_threshold: float = DEFAULT_LAYER_MEAN_THRESHOLD,
    layer_min_threshold: float = DEFAULT_LAYER_MIN_THRESHOLD,
    fallback_mean_edge: float = DEFAULT_FALLBACK_MEAN_EDGE,
    fallback_min_edge: float = DEFAULT_FALLBACK_MIN_EDGE,
    include_flow_policies: bool = False,
    policy_horizon: int = FORWARD_HORIZON,
    validation_horizon: int = FORWARD_HORIZON,
    target_horizon: int = FORWARD_HORIZON,
):
    lstm_fit = lstm.fit_lstm(
        prices,
        index_df,
        as_of,
        policy_horizon=policy_horizon,
        target_horizon=target_horizon,
    )
    panel_open, open_cols = add_open_data_features(lstm_fit["panel"], open_dir=open_dir)
    layer_policy, layer_table = select_layer(
        lstm_fit,
        panel_open,
        as_of,
        lstm_fit["prices"],
        lstm_fit["index_df"],
        include_flow_policies=include_flow_policies,
        validation_horizon=validation_horizon,
    )
    tuned_fit = fit_tuned_model(
        prices=prices,
        index_df=index_df,
        as_of=as_of,
        shape_horizon=validation_horizon,
    )
    decision = decide_gate(
        layer_table,
        fallback_table=tuned_fit["shape_table"],
        layer_mean_threshold=layer_mean_threshold,
        layer_min_threshold=layer_min_threshold,
        fallback_mean_edge=fallback_mean_edge,
        fallback_min_edge=fallback_min_edge,
    )
    if decision.selected == "layered_lstm":
        submission = generate_layered_submission(lstm_fit, panel_open, as_of, layer_policy)
        return submission, decision, layer_policy, layer_table, len(open_cols)

    submission = generate_tuned_submission(tuned_fit, as_of=as_of)
    return submission, decision, tuned_fit["shape"], layer_table, len(open_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DATA_DIR / "open"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="submissions/gated_layered_lstm.csv")
    parser.add_argument("--table-out", default=None)
    parser.add_argument("--layer-mean-threshold", type=float, default=DEFAULT_LAYER_MEAN_THRESHOLD)
    parser.add_argument("--layer-min-threshold", type=float, default=DEFAULT_LAYER_MIN_THRESHOLD)
    parser.add_argument(
        "--fallback-mean-edge",
        type=float,
        default=DEFAULT_FALLBACK_MEAN_EDGE,
        help="Require layered LSTM validation mean to exceed tuned-XGB fallback by this margin.",
    )
    parser.add_argument(
        "--fallback-min-edge",
        type=float,
        default=DEFAULT_FALLBACK_MIN_EDGE,
        help="Require layered LSTM worst validation window to exceed tuned-XGB fallback by this margin.",
    )
    parser.add_argument(
        "--policy-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon used to select the base LSTM portfolio policy.",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon for direct future-rank/future-weight training targets.",
    )
    parser.add_argument(
        "--validation-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon used to select the layered portfolio policy.",
    )
    parser.add_argument(
        "--include-flow-policies",
        action="store_true",
        help="Allow experimental fund-flow portfolio layers. Off by default to preserve the validated production route.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    submission, decision, selected_policy, layer_table, n_open = generate_gated_submission(
        prices,
        index_df,
        as_of,
        open_dir=args.open_dir,
        layer_mean_threshold=args.layer_mean_threshold,
        layer_min_threshold=args.layer_min_threshold,
        fallback_mean_edge=args.fallback_mean_edge,
        fallback_min_edge=args.fallback_min_edge,
        include_flow_policies=args.include_flow_policies,
        policy_horizon=args.policy_horizon,
        validation_horizon=args.validation_horizon,
        target_horizon=args.target_horizon,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if args.table_out:
        table_path = Path(args.table_out)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        layer_table.to_csv(table_path, index=False)
        print(f">> wrote layer table to {table_path}")
    print(f">> open_features={n_open}")
    print(f">> selected={decision.selected}")
    print(f">> gate_reason={decision.reason}")
    print(f">> selected_policy={selected_policy}")
    print(f">> best_layer_mean={decision.best_layer_mean:.6f}")
    print(f">> best_layer_min={decision.best_layer_min:.6f}")
    if decision.fallback_mean is not None:
        print(f">> fallback_mean={decision.fallback_mean:.6f}")
        print(f">> fallback_min={decision.fallback_min:.6f}")
        print(f">> fallback_mean_edge={args.fallback_mean_edge:.6f}")
        print(f">> fallback_min_edge={args.fallback_min_edge:.6f}")
    print(f">> mean_threshold={args.layer_mean_threshold:.6f}")
    print(f">> min_threshold={args.layer_min_threshold:.6f}")
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
