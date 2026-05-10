"""Final Stage2 portfolio generator.

This is the compact submission path for ``submissions/portfolio.csv``.  It uses
only data available up to ``as_of``:

1. build OHLCV features;
2. train the official-style XGBoost baseline with an embargoed validation split;
3. choose the baseline top-k from observable market-regime statistics;
4. write a rank-weighted, capped portfolio.

Historical ensemble/LSTM/tree experiments are archived under ``history/`` and
are not required to reproduce the final submission.
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
    TARGET_COLUMN,
    VAL_DAYS,
    build_portfolio,
    prediction_frame,
    rank_ic,
    train_model,
    training_frame,
)
from features import build_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class GuardDecision:
    route: str
    reason: str
    idx_ret_5d: float
    idx_ret_20d: float
    breadth_ret_5d_pos: float
    median_ret_5d: float


def _ret_at(close: pd.Series, pos: int, days: int) -> float:
    if pos < days:
        return 0.0
    return float(close.iloc[pos] / close.iloc[pos - days] - 1.0)


def defensive_guard(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp) -> GuardDecision:
    idx = index_df.sort_values("date").set_index("date")
    close = idx["close"].astype(float)
    pos = idx.index.get_loc(as_of)
    idx_ret_5d = _ret_at(close, pos, 5)
    idx_ret_20d = _ret_at(close, pos, 20)

    px = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    if as_of not in px.index:
        raise ValueError(f"as_of {as_of.date()} is not in price data")
    asof_pos = px.index.get_loc(as_of)
    if asof_pos < 5:
        breadth = 0.5
        median5 = 0.0
    else:
        ret5 = (px.iloc[asof_pos] / px.iloc[asof_pos - 5] - 1.0).replace([np.inf, -np.inf], np.nan)
        breadth = float((ret5 > 0).mean())
        median5 = float(ret5.median())

    return GuardDecision(
        route="baseline_xgb",
        reason="asof_observable_baseline_guard",
        idx_ret_5d=idx_ret_5d,
        idx_ret_20d=idx_ret_20d,
        breadth_ret_5d_pos=breadth,
        median_ret_5d=median5,
    )


def guard_reason(decision: GuardDecision) -> str | None:
    overheated_breadth = decision.idx_ret_5d > 0.04 and decision.breadth_ret_5d_pos > 0.70
    mild_positive_flat = (
        0.02 < decision.idx_ret_5d < 0.04
        and 0.02 < decision.idx_ret_20d < 0.04
        and decision.breadth_ret_5d_pos > 0.55
    )
    weak_flat_chop = (
        -0.02 < decision.idx_ret_5d < 0.0
        and -0.02 < decision.idx_ret_20d < 0.02
        and 0.35 < decision.breadth_ret_5d_pos < 0.50
    )
    severe_broad_selloff = (
        decision.idx_ret_5d < -0.045
        and decision.idx_ret_20d < -0.05
        and decision.breadth_ret_5d_pos < 0.20
    )
    if overheated_breadth:
        return "baseline_guard_overheated_high_breadth"
    if mild_positive_flat:
        return "baseline_guard_mild_positive_flat_tape"
    if weak_flat_chop:
        return "baseline_guard_weak_flat_chop"
    if severe_broad_selloff:
        return "baseline_guard_severe_broad_selloff"
    return None


def resolve_baseline_top_k(reason: str | None, decision: GuardDecision, requested_top_k: int) -> int:
    if requested_top_k > 0:
        return requested_top_k
    if reason == "baseline_guard_overheated_high_breadth":
        if decision.idx_ret_5d > 0.07:
            return 60
        return 35 if decision.idx_ret_20d < 0 else 30
    if reason == "baseline_guard_mild_positive_flat_tape":
        return 120
    if reason == "baseline_guard_weak_flat_chop":
        return 30
    if reason == "baseline_guard_severe_broad_selloff":
        return 40
    return 50


def generate_baseline_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_prices = prices[prices["date"] <= as_of].copy()
    fit_index = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(fit_prices, fit_index)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train baseline guard.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    model = train_model(train_df, val_df)
    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())

    pred_df = prediction_frame(panel, as_of=as_of)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={as_of.date()}")
    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index(pred_df["stock_code"].astype(str).str.zfill(6))["score"]
    weights = build_portfolio(scores, top_k=top_k)
    sub = pd.DataFrame({"stock_code": weights.index.astype(str), "weight": weights.values})
    meta = pd.DataFrame(
        [
            {
                "baseline_top_k": top_k,
                "baseline_train_rows": len(train_df),
                "baseline_train_end": train_end.date().isoformat(),
                "baseline_val_start": val_start.date().isoformat(),
                "baseline_val_rank_ic": ic,
                "baseline_n_names": len(sub),
                "baseline_max_weight": float(sub["weight"].max()),
            }
        ]
    )
    return sub, meta


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    baseline_top_k: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    decision = defensive_guard(prices, index_df, as_of)
    reason = guard_reason(decision)
    effective_top_k = resolve_baseline_top_k(reason, decision, baseline_top_k)
    sub, route_meta = generate_baseline_submission(prices, index_df, as_of, top_k=effective_top_k)

    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "model": "stage2_baseline_guard_ensemble",
                "selected_route": "baseline_xgb",
                "guard_reason": reason or "baseline_guard_default_baseline",
                "baseline_top_k": baseline_top_k,
                "effective_baseline_top_k": effective_top_k,
                "n_names": len(sub),
                "max_weight": float(sub["weight"].max()),
                "effective_n": float(1.0 / np.square(sub["weight"].to_numpy()).sum()),
                **decision.__dict__,
            }
        ]
    )
    route_cols = route_meta.add_prefix("route_")
    meta = pd.concat([meta.reset_index(drop=True), route_cols.reset_index(drop=True)], axis=1)
    return sub, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data.")
    parser.add_argument("--baseline-top-k", type=int, default=0, help="Use 0 for regime-adaptive top-k.")
    parser.add_argument("--out", default="submissions/portfolio.csv")
    parser.add_argument("--meta-out", default="submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])

    sub, meta = generate_submission(prices, index_df, as_of, baseline_top_k=args.baseline_top_k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)

    print(f">> model=stage2_baseline_guard_ensemble as_of={as_of.date()}")
    print(meta[["as_of", "selected_route", "guard_reason", "n_names", "max_weight", "effective_n"]].to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
