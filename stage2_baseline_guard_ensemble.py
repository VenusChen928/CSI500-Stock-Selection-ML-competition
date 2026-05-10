"""Baseline-guarded Stage2 ensemble.

The current weekly consensus improves mean excess, but several complete-week
windows still underperform the original XGBoost baseline.  This route keeps the
weekly consensus in regimes where it has a clear edge and falls back to the
original baseline in as-of-observable regimes where the consensus is fragile:

* broad overheated rebound / high breadth,
* mild positive tape with flat medium trend,
* weak flat chop,
* severe broad selloff.

The guard uses only prices and index data up to ``as_of``.  It does not inspect
realized window returns or cached scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import (
    EMBARGO_DAYS,
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    MIN_STOCKS,
    TARGET_COLUMN,
    VAL_DAYS,
    build_portfolio,
    prediction_frame,
    rank_ic,
    train_model,
    training_frame,
)
from features import build_features
from stage2_tree_consensus import defensive_guard
from stage2_weekly_alpha_overlay import generate_submission as generate_alpha_submission
from stage2_weekly_consensus_ensemble import generate_submission as generate_weekly_consensus
from stage2_weekly_cycle_tree import PortfolioShape, generate_submission as generate_cycle_submission

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def generate_baseline_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
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


def generate_equal_universe_submission(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = (
        prices.loc[prices["date"] == pd.Timestamp(as_of), "stock_code"]
        .astype(str)
        .str.zfill(6)
        .drop_duplicates()
        .sort_values()
    )
    if len(today) < MIN_STOCKS:
        raise RuntimeError(f"Only {len(today)} names available for {as_of.date()}")
    weight = 1.0 / float(len(today))
    sub = pd.DataFrame({"stock_code": today.to_numpy(), "weight": weight})
    meta = pd.DataFrame(
        [
            {
                "defensive_route": "equal_universe",
                "defensive_n_names": len(sub),
                "defensive_weight": weight,
            }
        ]
    )
    return sub, meta


def cap_raw_weights(raw: pd.Series, max_weight: float) -> pd.Series:
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


def rank_term(frame: pd.DataFrame, column: str, direction: str) -> pd.Series:
    ranks = frame[column].astype(float).replace([np.inf, -np.inf], np.nan).rank(method="average", pct=True)
    ranks = ranks.fillna(0.5)
    if direction == "low":
        return 1.0 - ranks
    if direction == "high":
        return ranks
    raise ValueError(f"unknown rank direction {direction}")


def generate_defensive_broad_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    temperature: float = 2.0,
    max_weight: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_features(prices[prices["date"] <= as_of].copy(), index_df[index_df["date"] <= as_of].copy())
    frame = panel[panel["date"] == pd.Timestamp(as_of)].copy()
    if len(frame) < MIN_STOCKS:
        raise RuntimeError(f"Only {len(frame)} names available for {as_of.date()}")
    terms = [
        ("beta_60d", "low", 0.25),
        ("market_corr_60d", "low", 0.15),
        ("downside_vol_20d", "low", 0.20),
        ("drawdown_20d", "high", 0.15),
        ("obv_20d", "high", 0.15),
        ("amount_z_20d", "high", 0.10),
    ]
    score = pd.Series(0.0, index=frame.index)
    total = 0.0
    used: list[str] = []
    for column, direction, weight in terms:
        if column not in frame.columns:
            continue
        score = score.add(weight * rank_term(frame, column, direction), fill_value=0.0)
        total += abs(weight)
        used.append(f"{column}:{direction}:{weight:g}")
    if total <= 0:
        return generate_equal_universe_submission(prices, as_of)
    score = score / total
    scale = float((score - score.median()).abs().median()) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(score.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raw = pd.Series(1.0, index=frame["stock_code"].astype(str).str.zfill(6))
    else:
        z = ((score - score.median()) / scale).clip(-3, 3)
        raw = pd.Series(np.exp(temperature * z.to_numpy()), index=frame["stock_code"].astype(str).str.zfill(6))
    weights = cap_raw_weights(raw, max_weight=max_weight)
    sub = pd.DataFrame({"stock_code": weights.index.astype(str), "weight": weights.values})
    meta = pd.DataFrame(
        [
            {
                "defensive_route": "broad_defensive_tilt",
                "defensive_temperature": temperature,
                "defensive_max_weight": max_weight,
                "defensive_n_names": len(sub),
                "defensive_effective_n": float(1.0 / np.square(sub["weight"].to_numpy()).sum()),
                "defensive_terms": " | ".join(used),
            }
        ]
    )
    return sub, meta


def generate_weekly_cycle_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape = PortfolioShape(
        top_k=40,
        score_temperature=0.80,
        rank_power=3.0,
        score_rank_blend=0.60,
        max_weight=0.08,
    )
    sub, meta, _ = generate_cycle_submission(
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


def generate_weekly_alpha_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub, meta, _ = generate_alpha_submission(prices, index_df, as_of, mode=mode, meta_cache_dir=None)
    return sub, meta


def guard_reason(decision) -> str | None:
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


def resolve_baseline_top_k(reason: str, decision, requested_top_k: int) -> int:
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


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    baseline_top_k: int = 0,
    cache_dir: Path | None = None,
    weekly_top_k: int = 30,
    weekly_rank_power: float = 6.0,
    weekly_max_weight: float = 0.095,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    decision = defensive_guard(prices, index_df, as_of)
    reason = guard_reason(decision)
    if (
        reason == "baseline_guard_overheated_high_breadth"
        and decision.idx_ret_5d > 0.07
        and decision.idx_ret_20d > 0.08
        and decision.breadth_ret_5d_pos > 0.80
    ):
        sub, route_meta = generate_weekly_cycle_submission(prices, index_df, as_of)
        selected_route = "weekly_cycle_tree"
        reason = "weekly_cycle_extreme_broad_rebound"
        effective_baseline_top_k = 0
    elif (
        reason == "baseline_guard_overheated_high_breadth"
        and decision.idx_ret_20d < 0.0
        and decision.breadth_ret_5d_pos > 0.65
    ):
        sub, route_meta = generate_weekly_alpha_submission(prices, index_df, as_of, mode="auto")
        selected_route = "weekly_alpha_auto"
        reason = "weekly_alpha_overheated_negative_medium_trend"
        effective_baseline_top_k = 0
    elif (
        reason is None
        and decision.idx_ret_20d < -0.08
        and -0.015 < decision.idx_ret_5d < 0.01
        and 0.30 < decision.breadth_ret_5d_pos < 0.45
    ):
        sub, route_meta = generate_weekly_cycle_submission(prices, index_df, as_of)
        selected_route = "weekly_cycle_tree"
        reason = "weekly_cycle_long_selloff_stabilization"
        effective_baseline_top_k = 0
    elif (
        reason == "baseline_guard_weak_flat_chop"
        and decision.idx_ret_20d > 0.0
        and 0.35 < decision.breadth_ret_5d_pos < 0.50
    ):
        sub, route_meta = generate_weekly_alpha_submission(prices, index_df, as_of, mode="current_regime")
        selected_route = "weekly_alpha_current"
        reason = "weekly_alpha_weak_flat_positive_medium_trend"
        effective_baseline_top_k = 0
    elif reason is None:
        sub, route_meta = generate_weekly_consensus(
            prices,
            index_df,
            as_of,
            cache_dir=cache_dir,
            top_k=weekly_top_k,
            rank_power=weekly_rank_power,
            max_weight=weekly_max_weight,
        )
        selected_route = "weekly_consensus"
        effective_baseline_top_k = baseline_top_k
    elif reason == "baseline_guard_mild_positive_flat_tape":
        sub, route_meta = generate_defensive_broad_submission(prices, index_df, as_of)
        selected_route = "broad_defensive_tilt"
        effective_baseline_top_k = 0
    else:
        effective_baseline_top_k = resolve_baseline_top_k(reason, decision, baseline_top_k)
        sub, route_meta = generate_baseline_submission(prices, index_df, as_of, top_k=effective_baseline_top_k)
        selected_route = "baseline_xgb"

    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "model": "stage2_baseline_guard_ensemble",
                "selected_route": selected_route,
                "guard_reason": reason or "weekly_consensus_allowed",
                "baseline_top_k": baseline_top_k,
                "effective_baseline_top_k": effective_baseline_top_k,
                "weekly_top_k": weekly_top_k,
                "weekly_rank_power": weekly_rank_power,
                "weekly_max_weight": weekly_max_weight,
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
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--baseline-top-k", type=int, default=0, help="Use 0 for regime-adaptive baseline top-k.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--weekly-top-k", type=int, default=30)
    parser.add_argument("--weekly-rank-power", type=float, default=6.0)
    parser.add_argument("--weekly-max-weight", type=float, default=0.095)
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
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    sub, meta = generate_submission(
        prices,
        index_df,
        as_of,
        baseline_top_k=args.baseline_top_k,
        cache_dir=cache_dir,
        weekly_top_k=args.weekly_top_k,
        weekly_rank_power=args.weekly_rank_power,
        weekly_max_weight=args.weekly_max_weight,
    )
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
