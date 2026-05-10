"""
Layered open-data portfolio construction on top of the LSTM rank/weight model.

The previous open-data experiment showed that valuation/size features can hurt
when injected directly into a short-horizon LSTM.  This script instead uses
open data as a portfolio layer:

  1. Train the original LSTM rank/weight model.
  2. Build the original LSTM candidate weights.
  3. Validate small, interpretable exposure adjustments using only past windows:
     - flatten or concentrate the LSTM weights,
     - tilt away/toward high PB names,
     - tilt toward mid-size names,
     - optional broad size/PB filters.

The selected layer is then applied to the requested as-of date.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import lstm_rank_weight as lstm
from features import FORWARD_HORIZON
from open_data_features import add_open_data_features
from score_submission import score_window

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class LayerPolicy:
    name: str
    gamma: float = 1.0
    value_tilt: float = 0.0
    mid_size_tilt: float = 0.0
    min_size_rank: float = 0.0
    max_size_rank: float = 1.0
    min_pb_rank: float = 0.0
    max_pb_rank: float = 1.0
    min_flow_rank: float = 0.0
    max_flow_rank: float = 1.0
    flow_tilt: float = 0.0


def candidate_policies(include_flow: bool = False) -> list[LayerPolicy]:
    policies = [
        LayerPolicy("base"),
        LayerPolicy("flatten_075", gamma=0.75),
        LayerPolicy("flatten_085", gamma=0.85),
        LayerPolicy("concentrate_115", gamma=1.15),
        LayerPolicy("concentrate_130", gamma=1.30),
        LayerPolicy("value_tilt_025", value_tilt=0.25),
        LayerPolicy("value_tilt_050", value_tilt=0.50),
        LayerPolicy("growth_tilt_025", value_tilt=-0.25),
        LayerPolicy("mid_size_025", mid_size_tilt=0.25),
        LayerPolicy("mid_size_050", mid_size_tilt=0.50),
        LayerPolicy("value_mid_size", value_tilt=0.25, mid_size_tilt=0.25),
        LayerPolicy("drop_biggest_10pct", max_size_rank=0.90),
        LayerPolicy("drop_smallest_10pct", min_size_rank=0.10),
        LayerPolicy("drop_high_pb_10pct", max_pb_rank=0.90),
        LayerPolicy("drop_low_pb_10pct", min_pb_rank=0.10),
    ]
    if include_flow:
        policies += [
        LayerPolicy("drop_low_flow_20pct", min_flow_rank=0.20),
        LayerPolicy("drop_low_flow_30pct", min_flow_rank=0.30),
        LayerPolicy("drop_high_flow_20pct", max_flow_rank=0.80),
        LayerPolicy("positive_flow_tilt_025", flow_tilt=0.25),
        LayerPolicy("positive_flow_tilt_050", flow_tilt=0.50),
        LayerPolicy("negative_flow_tilt_025", flow_tilt=-0.25),
        LayerPolicy("value_positive_flow", value_tilt=0.25, flow_tilt=0.25),
        LayerPolicy("mid_size_positive_flow", mid_size_tilt=0.25, flow_tilt=0.25),
        LayerPolicy("drop_smallest_positive_flow", min_size_rank=0.10, flow_tilt=0.25),
        ]
    return policies


def _open_slice(panel_open: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    cols = ["date", "stock_code", "float_mv_rank", "pb_rank", "main_net_pct_ma5_rank"]
    available = [c for c in cols if c in panel_open.columns]
    d = panel_open[panel_open["date"] == pd.Timestamp(as_of)][available].copy()
    d["stock_code"] = d["stock_code"].astype(str).str.zfill(6)
    if "float_mv_rank" not in d.columns:
        d["float_mv_rank"] = 0.5
    if "pb_rank" not in d.columns:
        d["pb_rank"] = 0.5
    if "main_net_pct_ma5_rank" not in d.columns:
        d["main_net_pct_ma5_rank"] = 0.5
    d["float_mv_rank"] = d["float_mv_rank"].fillna(0.5)
    d["pb_rank"] = d["pb_rank"].fillna(0.5)
    d["main_net_pct_ma5_rank"] = d["main_net_pct_ma5_rank"].fillna(0.5)
    return d.set_index("stock_code")


def apply_layer(
    base_weights: pd.Series,
    panel_open: pd.DataFrame,
    as_of: pd.Timestamp,
    policy: LayerPolicy,
) -> pd.Series:
    meta = _open_slice(panel_open, as_of)
    w = base_weights.copy()
    w.index = w.index.astype(str).str.zfill(6)
    frame = pd.DataFrame({"weight": w}).join(meta, how="left")
    frame["float_mv_rank"] = frame["float_mv_rank"].fillna(0.5)
    frame["pb_rank"] = frame["pb_rank"].fillna(0.5)
    frame["main_net_pct_ma5_rank"] = frame["main_net_pct_ma5_rank"].fillna(0.5)

    keep = (
        frame["float_mv_rank"].between(policy.min_size_rank, policy.max_size_rank)
        & frame["pb_rank"].between(policy.min_pb_rank, policy.max_pb_rank)
        & frame["main_net_pct_ma5_rank"].between(policy.min_flow_rank, policy.max_flow_rank)
    )
    frame = frame[keep].copy()
    if len(frame) < lstm.MIN_STOCKS:
        raise ValueError(f"policy {policy.name} leaves only {len(frame)} names")

    adjusted = np.power(frame["weight"].clip(lower=1e-12), policy.gamma)
    if policy.value_tilt:
        adjusted *= np.exp(policy.value_tilt * (0.5 - frame["pb_rank"]))
    if policy.mid_size_tilt:
        adjusted *= np.exp(-policy.mid_size_tilt * np.abs(frame["float_mv_rank"] - 0.5))
    if policy.flow_tilt:
        adjusted *= np.exp(policy.flow_tilt * (frame["main_net_pct_ma5_rank"] - 0.5))
    adjusted = pd.Series(adjusted.to_numpy(), index=frame.index)
    return lstm.apply_cap(adjusted)


def validation_windows(
    val_dates: list[pd.Timestamp],
    trading_dates: np.ndarray,
    horizon: int = FORWARD_HORIZON,
):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    out = []
    for d in val_dates[::horizon]:
        d = pd.Timestamp(d)
        idx = date_to_idx.get(d)
        if idx is None or idx + horizon >= len(trading_dates):
            continue
        out.append((d, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + horizon])))
    return out


def refit_validation_predictions(fit_result: dict, as_of: pd.Timestamp):
    panel = fit_result["panel"]
    trading_dates = fit_result["trading_dates"]
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    target_horizon = fit_result.get("target_horizon", FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - target_horizon)])
    _, _, rank_col, weight_col = lstm._target_columns(target_horizon)
    usable = panel.dropna(subset=lstm.SEQUENCE_FEATURES + [rank_col, weight_col])
    train_pool = usable[usable["date"] <= train_cutoff].copy()
    _, val_dates, _, _ = lstm.date_splits(train_pool)
    val_records = lstm.build_records(panel, val_dates, target_horizon=target_horizon)
    val_records = fit_result["normalizer"].apply(val_records)
    val_pred = lstm.predict_records(fit_result["model"], val_records)
    return val_pred, val_dates


def select_layer(
    fit_result: dict,
    panel_open: pd.DataFrame,
    as_of: pd.Timestamp,
    prices,
    index_df,
    include_flow_policies: bool = False,
    validation_horizon: int = FORWARD_HORIZON,
):
    val_pred, val_dates = refit_validation_predictions(fit_result, as_of)
    windows = validation_windows(
        val_dates,
        fit_result["trading_dates"],
        horizon=validation_horizon,
    )
    min_windows = max(1, int(np.ceil(len(windows) * 0.8)))
    policies = candidate_policies(include_flow=include_flow_policies)
    rows = []
    for policy in policies:
        scores = []
        for d, start, end in windows:
            try:
                base_w = lstm.weights_from_predictions(val_pred, d, fit_result["policy"])
                w = apply_layer(base_w, panel_open, d, policy)
            except ValueError:
                continue
            scores.append(score_window(w, prices, index_df, start, end)["excess_return"])
        if scores:
            mean_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            std_score = float(np.std(scores))
            rows.append(
                {
                    "policy": policy.name,
                    "n_windows": len(scores),
                    "required_windows": min_windows,
                    "gamma": policy.gamma,
                    "value_tilt": policy.value_tilt,
                    "mid_size_tilt": policy.mid_size_tilt,
                    "min_size_rank": policy.min_size_rank,
                    "max_size_rank": policy.max_size_rank,
                    "min_pb_rank": policy.min_pb_rank,
                    "max_pb_rank": policy.max_pb_rank,
                    "min_flow_rank": policy.min_flow_rank,
                    "max_flow_rank": policy.max_flow_rank,
                    "flow_tilt": policy.flow_tilt,
                    "mean_excess_return": mean_score,
                    "sum_excess_return": float(np.sum(scores)),
                    "min_excess_return": min_score,
                    "std_excess_return": std_score,
                    "utility_score": mean_score + min_score - 0.5 * std_score,
                }
            )
    table = pd.DataFrame(rows)
    eligible = table[table["n_windows"] >= min_windows].copy()
    if eligible.empty:
        eligible = table
    table = eligible.sort_values(
        ["utility_score", "mean_excess_return", "min_excess_return", "sum_excess_return"],
        ascending=False,
    )
    best = table.iloc[0]
    policy = next(p for p in policies if p.name == best["policy"])
    return policy, table


def generate_layered_submission(fit_result: dict, panel_open: pd.DataFrame, as_of: pd.Timestamp, policy: LayerPolicy):
    records = lstm.build_records(
        fit_result["panel"],
        [as_of],
        target_horizon=fit_result.get("target_horizon", FORWARD_HORIZON),
        require_targets=False,
    )
    records = fit_result["normalizer"].apply(records)
    pred = lstm.predict_records(fit_result["model"], records)
    base_w = lstm.weights_from_predictions(pred, as_of, fit_result["policy"])
    weights = apply_layer(base_w, panel_open, as_of, policy)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DATA_DIR / "open"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="submissions/experiments/layered_lstm_20260421.csv")
    parser.add_argument("--table-out", default="submissions/reports/layered_lstm_policy_table_20260421.csv")
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
        help="Trading-day horizon used to select the open-data portfolio layer.",
    )
    parser.add_argument(
        "--include-flow-policies",
        action="store_true",
        help="Allow fund-flow portfolio layers. Off by default to preserve the validated production route.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = lstm.fit_lstm(
        prices,
        index_df,
        as_of,
        policy_horizon=args.policy_horizon,
        target_horizon=args.target_horizon,
    )
    panel_open, open_cols = add_open_data_features(fit["panel"], open_dir=args.open_dir)
    policy, table = select_layer(
        fit,
        panel_open,
        as_of,
        fit["prices"],
        fit["index_df"],
        include_flow_policies=args.include_flow_policies,
        validation_horizon=args.validation_horizon,
    )
    submission = generate_layered_submission(fit, panel_open, as_of, policy)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    table_path = Path(args.table_out)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    print(f">> open_features={len(open_cols)}")
    print(f">> selected layer: {policy}")
    print(">> validation layer table")
    print(table.head(12).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(f">> wrote policy table to {table_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
