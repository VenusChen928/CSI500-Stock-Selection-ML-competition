"""
Validation-selected fund-flow gate for the 5-day stage2 baseline.

The gate is a post-portfolio risk filter: train the standard baseline scorer,
build baseline portfolios on validation windows, choose a simple fund-flow
policy using only those validation windows, then apply the chosen policy to the
requested as-of portfolio.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import (
    EMBARGO_DAYS,
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    MAX_WEIGHT,
    MIN_STOCKS,
    VAL_DAYS,
    build_features,
    build_portfolio,
    prediction_frame,
    train_model,
    training_frame,
)
from score_submission import score_window

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OPEN_DIR = ROOT / "archive" / "data_unused" / "open"


@dataclass(frozen=True)
class FlowPolicy:
    name: str
    min_flow_rank: float = 0.0
    max_flow_rank: float = 1.0
    flow_tilt: float = 0.0
    super_tilt: float = 0.0


POLICIES = [
    FlowPolicy("base"),
    FlowPolicy("drop_low_10", min_flow_rank=0.10),
    FlowPolicy("drop_low_20", min_flow_rank=0.20),
    FlowPolicy("drop_low_30", min_flow_rank=0.30),
    FlowPolicy("drop_high_10", max_flow_rank=0.90),
    FlowPolicy("tilt_pos_25", flow_tilt=0.25),
    FlowPolicy("tilt_pos_50", flow_tilt=0.50),
    FlowPolicy("tilt_pos_100", flow_tilt=1.00),
    FlowPolicy("tilt_super_50", super_tilt=0.50),
]


def _apply_cap(weights: pd.Series) -> pd.Series:
    w = weights[weights > 0].copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"portfolio has only {len(w)} names after flow gate")
    w = w / w.sum()
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def load_flow(open_dir: str | Path) -> pd.DataFrame:
    path = Path(open_dir) / "stock_fund_flow.parquet"
    if not path.exists():
        return pd.DataFrame()
    flow = pd.read_parquet(path)
    if flow.empty:
        return flow
    flow["date"] = pd.to_datetime(flow["date"])
    flow["stock_code"] = flow["stock_code"].astype(str).str.zfill(6)
    for col in ["main_net_pct", "super_net_pct"]:
        flow[col] = pd.to_numeric(flow[col], errors="coerce")
    flow = flow.sort_values(["stock_code", "date"])
    flow["main_net_pct_ma5"] = flow.groupby("stock_code")["main_net_pct"].transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    flow["super_net_pct_ma5"] = flow.groupby("stock_code")["super_net_pct"].transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    return flow


def flow_meta(flow: pd.DataFrame, as_of: pd.Timestamp, max_lag_days: int = 7) -> pd.DataFrame:
    if flow.empty:
        return pd.DataFrame(columns=["flow_rank", "super_rank"])
    as_of = pd.Timestamp(as_of)
    chunks = []
    for _, group in flow[flow["date"] <= as_of].groupby("stock_code", sort=False):
        row = group.tail(1)
        if not row.empty and (as_of - row["date"].iloc[0]).days <= max_lag_days:
            chunks.append(row)
    if not chunks:
        return pd.DataFrame(columns=["flow_rank", "super_rank"])
    meta = pd.concat(chunks, ignore_index=True)
    meta["flow_rank"] = meta["main_net_pct_ma5"].rank(method="average", pct=True)
    meta["super_rank"] = meta["super_net_pct_ma5"].rank(method="average", pct=True)
    return meta.set_index("stock_code")[["flow_rank", "super_rank"]].fillna(0.5)


def apply_flow_policy(weights: pd.Series, meta: pd.DataFrame, policy: FlowPolicy) -> pd.Series:
    w = weights.copy()
    w.index = w.index.astype(str).str.zfill(6)
    frame = pd.DataFrame({"weight": w}).join(meta, how="left")
    frame["flow_rank"] = frame["flow_rank"].fillna(0.5)
    frame["super_rank"] = frame["super_rank"].fillna(0.5)
    frame = frame[frame["flow_rank"].between(policy.min_flow_rank, policy.max_flow_rank)].copy()
    adjusted = frame["weight"].copy()
    if policy.flow_tilt:
        adjusted *= np.exp(policy.flow_tilt * (frame["flow_rank"] - 0.5))
    if policy.super_tilt:
        adjusted *= np.exp(policy.super_tilt * (frame["super_rank"] - 0.5))
    return _apply_cap(pd.Series(adjusted.to_numpy(), index=frame.index))


def split_train_val(df: pd.DataFrame, val_days: int = VAL_DAYS, embargo_days: int = EMBARGO_DAYS):
    dates = np.sort(df["date"].unique())
    if len(dates) < val_days + embargo_days + 20:
        raise RuntimeError("Not enough dates for validation split.")
    val_start = pd.Timestamp(dates[-val_days])
    train_end = pd.Timestamp(dates[-(val_days + embargo_days + 1)])
    return df[df["date"] <= train_end].copy(), df[df["date"] >= val_start].copy(), train_end, val_start


def validation_windows(val_dates: np.ndarray, trading_dates: np.ndarray):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for offset in range(0, len(val_dates), FORWARD_HORIZON):
        as_of = pd.Timestamp(val_dates[offset])
        idx = date_to_idx.get(as_of)
        if idx is None or idx + FORWARD_HORIZON >= len(trading_dates):
            continue
        windows.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])))
    return windows


def fit_baseline(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp):
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(prices, index_df)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    train_pool = training_frame(panel, max_date=train_cutoff)
    train_df, val_df, train_end, val_start = split_train_val(train_pool)
    model = train_model(train_df, val_df)
    return {
        "prices": prices,
        "index_df": index_df,
        "panel": panel,
        "trading_dates": trading_dates,
        "train_df": train_df,
        "val_df": val_df,
        "train_end": train_end,
        "val_start": val_start,
        "model": model,
    }


def baseline_weights(fit: dict, as_of: pd.Timestamp) -> pd.Series:
    pred = prediction_frame(fit["panel"], as_of=as_of).copy()
    pred["score"] = fit["model"].predict(pred[FEATURE_COLUMNS])
    return build_portfolio(pred.set_index("stock_code")["score"])


def select_policy(fit: dict, flow: pd.DataFrame) -> tuple[FlowPolicy, pd.DataFrame]:
    windows = validation_windows(np.sort(fit["val_df"]["date"].unique()), fit["trading_dates"])
    rows = []
    for policy in POLICIES:
        scores = []
        names = []
        for as_of, start, end in windows:
            try:
                base_w = baseline_weights(fit, as_of)
                gated_w = apply_flow_policy(base_w, flow_meta(flow, as_of), policy)
            except ValueError:
                continue
            result = score_window(gated_w, fit["prices"], fit["index_df"], start, end)
            scores.append(result["excess_return"])
            names.append(len(gated_w))
        if scores:
            mean_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            std_score = float(np.std(scores))
            rows.append({
                "policy": policy.name,
                "n_windows": len(scores),
                "mean_excess_return": mean_score,
                "sum_excess_return": float(np.sum(scores)),
                "min_excess_return": min_score,
                "std_excess_return": std_score,
                "avg_names": float(np.mean(names)),
                "utility_score": mean_score + min_score - 0.25 * std_score,
            })
    table = pd.DataFrame(rows).sort_values(
        ["utility_score", "mean_excess_return", "min_excess_return"],
        ascending=False,
    )
    best_name = str(table.iloc[0]["policy"]) if not table.empty else "base"
    return next(policy for policy in POLICIES if policy.name == best_name), table


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    open_dir: str | Path,
) -> tuple[pd.DataFrame, FlowPolicy, pd.DataFrame, dict]:
    fit = fit_baseline(prices, index_df, as_of)
    flow = load_flow(open_dir)
    policy, table = select_policy(fit, flow)
    weights = baseline_weights(fit, as_of)
    gated = apply_flow_policy(weights, flow_meta(flow, as_of), policy)
    return pd.DataFrame({"stock_code": gated.index, "weight": gated.values}), policy, table, fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DEFAULT_OPEN_DIR))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/experiments/stage2_fund_flow_gate.csv")
    parser.add_argument("--table-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, policy, table, fit = generate_submission(prices, index_df, as_of, args.open_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.table_out:
        table_path = Path(args.table_out)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
    print(f">> as_of={as_of.date()} selected_policy={policy.name}")
    print(f">> train: {len(fit['train_df']):,} rows up to {fit['train_end'].date()}")
    print(f">> val:   {len(fit['val_df']):,} rows from {fit['val_start'].date()}")
    print(">> policy table")
    print(table.head(10).to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
