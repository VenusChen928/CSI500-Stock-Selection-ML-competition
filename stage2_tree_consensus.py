"""Stage2 tree portfolio consensus.

Inspired by the phase-1 class ensemble result: aggregate independent portfolio
weights, then keep the most agreed-upon names.  This is intentionally simpler
than the LSTM regime route and serves as a regularized stage2 challenger.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from features import (
    FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    QUALITY_FEATURE_COLUMNS,
    REFERENCE_FEATURE_COLUMNS,
    TARGET_3D_COLUMN,
    TARGET_COLUMN,
    TARGET_EXCESS_3D_COLUMN,
    TARGET_EXCESS_COLUMN,
    prediction_frame,
)
from lightgbm_portfolio import fit_lightgbm_model, generate_submission as generate_lgb_submission
from tuned_xgboost_portfolio import fit_tuned_model, generate_submission as generate_xgb_submission

DATA_DIR = ROOT / "data"
DEFAULT_TOP_K = 30
DEFAULT_ALPHA_XGB = 0.05
DEFAULT_RANK_POWER = 1.6
DEFAULT_EQUAL_MIX = 0.0
DEFAULT_MAX_WEIGHT = 0.04


@dataclass(frozen=True)
class ConsensusShape:
    top_k: int
    alpha_xgb: float
    rank_power: float
    equal_mix: float
    max_weight: float


@dataclass(frozen=True)
class GuardDecision:
    route: str
    reason: str
    idx_ret_5d: float
    idx_ret_20d: float
    breadth_ret_5d_pos: float
    median_ret_5d: float


def _weights(sub: pd.DataFrame) -> pd.Series:
    out = sub.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    return out.set_index("stock_code")["weight"].astype(float)


def _cap(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights[weights > 0].astype(float).copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} names")
    w = w / w.sum()
    for _ in range(80):
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

    if idx_ret_5d < -0.03 and breadth < 0.25 and idx_ret_20d > -0.05:
        return GuardDecision(
            route="equal_universe",
            reason="weak_short_tape_low_breadth_avoid_active_selection_noise",
            idx_ret_5d=idx_ret_5d,
            idx_ret_20d=idx_ret_20d,
            breadth_ret_5d_pos=breadth,
            median_ret_5d=median5,
        )
    return GuardDecision(
        route="tree_consensus",
        reason="normal_tree_consensus",
        idx_ret_5d=idx_ret_5d,
        idx_ret_20d=idx_ret_20d,
        breadth_ret_5d_pos=breadth,
        median_ret_5d=median5,
    )


def equal_universe_submission(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    today = prices[prices["date"] == as_of]["stock_code"].astype(str).str.zfill(6).drop_duplicates().sort_values()
    weights = pd.Series(1.0 / len(today), index=today)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def consensus_weights(xgb_w: pd.Series, lgb_w: pd.Series, shape: ConsensusShape) -> pd.Series:
    xgb_w = xgb_w[xgb_w > 0]
    lgb_w = lgb_w[lgb_w > 0]
    all_codes = sorted(set(xgb_w.index) | set(lgb_w.index))
    blended = shape.alpha_xgb * xgb_w.reindex(all_codes).fillna(0.0) + (1.0 - shape.alpha_xgb) * lgb_w.reindex(all_codes).fillna(0.0)

    xgb_rank = xgb_w.rank(ascending=False, method="first")
    lgb_rank = lgb_w.rank(ascending=False, method="first")
    rank_sum = pd.Series(
        {
            code: xgb_rank.get(code, len(xgb_w) + 1) + lgb_rank.get(code, len(lgb_w) + 1)
            for code in all_codes
        }
    ).sort_values()

    intersection = set(xgb_w.index) & set(lgb_w.index)
    chosen = [code for code in rank_sum.index if code in intersection and blended.get(code, 0.0) > 0]
    for code in rank_sum.index:
        if len(chosen) >= shape.top_k:
            break
        if code not in chosen and blended.get(code, 0.0) > 0:
            chosen.append(code)
    chosen = chosen[: shape.top_k]

    selected = blended.reindex(chosen).fillna(0.0)
    ranks = pd.Series(np.arange(len(selected), 0, -1, dtype=float), index=selected.index)
    rank_component = ranks ** shape.rank_power
    raw = 0.50 * selected / selected.sum() + 0.50 * rank_component / rank_component.sum()
    if shape.equal_mix > 0:
        raw = (1.0 - shape.equal_mix) * raw + shape.equal_mix * pd.Series(1.0 / len(raw), index=raw.index)
    return _cap(raw, max_weight=shape.max_weight)


def _rank_pct(scores: pd.Series) -> pd.Series:
    return scores.rank(method="average", pct=True)


def reweight_gate_enabled(decision: GuardDecision, gate: str) -> bool:
    if gate == "none":
        return True
    if gate == "not_calm":
        return abs(decision.idx_ret_20d) > 0.02 or abs(decision.idx_ret_5d) > 0.04
    if gate == "medium_move":
        return abs(decision.idx_ret_20d) > 0.04 or decision.idx_ret_5d > 0.06
    if gate == "post_drawdown":
        return decision.idx_ret_20d < -0.02
    if gate == "trend_dislocation":
        return decision.idx_ret_20d < -0.02 or decision.idx_ret_20d > 0.04 or decision.idx_ret_5d > 0.06
    raise ValueError(f"unknown reweight gate: {gate}")


def factor_reweight(
    weights: pd.Series,
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    factor: str,
    direction: str,
    gamma: float,
    power: float,
    max_weight: float,
) -> pd.Series:
    """Tilt an already-selected portfolio by an as-of factor without changing names."""
    if not factor or gamma <= 0:
        return weights
    frame = panel[panel["date"] == as_of].dropna(subset=[factor]).copy()
    if frame.empty:
        return weights
    scores = frame.set_index(frame["stock_code"].astype(str).str.zfill(6))[factor].astype(float)
    if direction == "low":
        # Low raw value receives high confidence, e.g. deeper recent drawdown.
        confidence = 1.0 - scores.rank(ascending=True, pct=True)
    elif direction == "high":
        confidence = scores.rank(ascending=True, pct=True)
    else:
        raise ValueError(f"unknown reweight direction: {direction}")
    adj = 1.0 + gamma * confidence.reindex(weights.index).fillna(0.5).clip(0.0, 1.0).pow(power)
    return _cap(weights * adj, max_weight=max_weight)


def _positive_score_weights(scores: pd.Series, top_k: int, max_weight: float, floor: float = 1e-6) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k)
    shifted = chosen - float(chosen.min()) + floor
    if not np.isfinite(shifted).all() or shifted.sum() <= 0:
        shifted = pd.Series(np.ones(len(chosen), dtype=float), index=chosen.index)
    return _cap(shifted / shifted.sum(), max_weight=max_weight)


def score_prop_consensus_weights(
    xgb_fit: dict,
    lgb_fit: dict,
    as_of: pd.Timestamp,
    features: list[str],
    shape: ConsensusShape,
    score_rank_blend: float = 0.35,
) -> pd.Series:
    """Blend raw model score magnitude with rank confidence before weighting."""
    xpred = prediction_frame(xgb_fit["panel"], as_of=as_of).dropna(subset=features).copy()
    lpred = prediction_frame(lgb_fit["panel"], as_of=as_of).dropna(subset=features).copy()
    x_scores = pd.Series(
        xgb_fit["model"].predict(xpred[features]),
        index=xpred["stock_code"].astype(str).str.zfill(6),
    )
    l_scores = pd.Series(
        lgb_fit["model"].predict(lpred[features]),
        index=lpred["stock_code"].astype(str).str.zfill(6),
    )
    all_codes = sorted(set(x_scores.index) & set(l_scores.index))
    x = x_scores.reindex(all_codes).astype(float)
    l = l_scores.reindex(all_codes).astype(float)
    x_rank = _rank_pct(x)
    l_rank = _rank_pct(l)
    rank_conf = shape.alpha_xgb * x_rank + (1.0 - shape.alpha_xgb) * l_rank

    def robust_z(s: pd.Series) -> pd.Series:
        med = float(s.median())
        mad = float((s - med).abs().median())
        scale = mad * 1.4826 if mad > 1e-12 else float(s.std(ddof=0))
        if scale <= 1e-12 or not np.isfinite(scale):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return ((s - med) / scale).clip(-4, 4)

    mag = shape.alpha_xgb * robust_z(x) + (1.0 - shape.alpha_xgb) * robust_z(l)
    mag_rank = _rank_pct(mag)
    blended = score_rank_blend * mag_rank + (1.0 - score_rank_blend) * rank_conf
    return _positive_score_weights(blended, top_k=shape.top_k, max_weight=shape.max_weight)


def generate_tree_consensus(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    shape: ConsensusShape,
    defensive_equal_gate: bool = False,
    features: list[str] | None = None,
    half_life: int | None = None,
    weight_floor: float = 0.0,
    adaptive_time_decay: bool = False,
    score_prop: bool = False,
    score_rank_blend: float = 0.35,
    target_column: str = TARGET_COLUMN,
    target_horizon: int = FORWARD_HORIZON,
    shape_horizon: int = FORWARD_HORIZON,
    reweight_factor: str = "",
    reweight_direction: str = "low",
    reweight_gamma: float = 0.0,
    reweight_power: float = 1.0,
    reweight_gate: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    # Hard as-of boundary: downstream train cutoffs already prevent target
    # leakage, but trimming here keeps future rows out of feature construction
    # and makes the pipeline easier to audit.
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    features = features or FEATURE_COLUMNS
    decision = defensive_guard(prices, index_df, as_of)
    model_half_life = half_life
    model_weight_floor = weight_floor
    decay_reason = "fixed_decay" if half_life is not None else "no_decay"
    if adaptive_time_decay and half_life is not None:
        post_rally_no_medium_support = (
            decision.idx_ret_5d > 0.03
            and 0.0 < decision.idx_ret_20d < 0.03
            and 0.55 < decision.breadth_ret_5d_pos < 0.75
        )
        pullback_in_medium_uptrend = (
            decision.idx_ret_5d < 0.0
            and decision.idx_ret_20d > 0.04
            and decision.breadth_ret_5d_pos < 0.45
        )
        if post_rally_no_medium_support or pullback_in_medium_uptrend:
            model_half_life = None
            model_weight_floor = 0.0
            decay_reason = (
                "core_post_rally_no_medium_support"
                if post_rally_no_medium_support
                else "core_pullback_in_medium_uptrend"
            )
        else:
            decay_reason = "adaptive_decay_on"
    if defensive_equal_gate and decision.route == "equal_universe":
        sub = equal_universe_submission(prices, as_of)
        meta = pd.DataFrame(
            [
                {
                    **shape.__dict__,
                    **decision.__dict__,
                    "feature_count": len(features),
                    "time_decay_half_life": half_life if half_life is not None else 0,
                    "time_decay_floor": weight_floor,
                    "selected_time_decay_half_life": 0,
                    "selected_time_decay_floor": 0.0,
                    "decay_reason": "equal_universe_no_model",
                    "portfolio_mode": "equal_universe",
                    "score_rank_blend": 0.0,
                    "reweight_factor": reweight_factor,
                    "reweight_direction": reweight_direction,
                    "reweight_gamma": reweight_gamma,
                    "reweight_power": reweight_power,
                    "reweight_gate": reweight_gate,
                    "reweight_applied": False,
                    "target_column": target_column,
                    "target_horizon": target_horizon,
                    "shape_horizon": shape_horizon,
                    "xgb_n": 0,
                    "lgb_n": 0,
                    "intersection_n": 0,
                }
            ]
        )
        return sub, meta

    xgb_fit = fit_tuned_model(
        prices,
        index_df,
        as_of=as_of,
        shape_horizon=shape_horizon,
        allow_equal_weight=False,
        features=features,
        half_life=model_half_life,
        weight_floor=model_weight_floor,
        target_column=target_column,
        target_horizon=target_horizon,
    )
    lgb_fit = fit_lightgbm_model(
        prices,
        index_df,
        as_of=as_of,
        shape_horizon=shape_horizon,
        features=features,
        half_life=model_half_life,
        weight_floor=model_weight_floor,
        target_column=target_column,
        target_horizon=target_horizon,
    )
    xgb_w = _weights(generate_xgb_submission(xgb_fit, as_of=as_of))
    lgb_w = _weights(generate_lgb_submission(lgb_fit, as_of=as_of))
    if score_prop:
        weights = score_prop_consensus_weights(
            xgb_fit,
            lgb_fit,
            as_of=as_of,
            features=features,
            shape=shape,
            score_rank_blend=score_rank_blend,
        )
    else:
        weights = consensus_weights(xgb_w, lgb_w, shape)
    reweight_applied = False
    if reweight_factor and reweight_gamma > 0 and reweight_gate_enabled(decision, reweight_gate):
        weights = factor_reweight(
            weights=weights,
            panel=lgb_fit["panel"],
            as_of=as_of,
            factor=reweight_factor,
            direction=reweight_direction,
            gamma=reweight_gamma,
            power=reweight_power,
            max_weight=shape.max_weight,
        )
        reweight_applied = True
    meta = pd.DataFrame(
        [
            {
                **shape.__dict__,
                **decision.__dict__,
                "feature_count": len(features),
                "time_decay_half_life": half_life if half_life is not None else 0,
                "time_decay_floor": weight_floor,
                "selected_time_decay_half_life": model_half_life if model_half_life is not None else 0,
                "selected_time_decay_floor": model_weight_floor,
                "decay_reason": decay_reason,
                "portfolio_mode": "score_prop" if score_prop else "rank_consensus",
                "score_rank_blend": score_rank_blend if score_prop else 0.0,
                "reweight_factor": reweight_factor,
                "reweight_direction": reweight_direction,
                "reweight_gamma": reweight_gamma,
                "reweight_power": reweight_power,
                "reweight_gate": reweight_gate,
                "reweight_applied": reweight_applied,
                "target_column": target_column,
                "target_horizon": target_horizon,
                "shape_horizon": shape_horizon,
                "xgb_n": int((xgb_w > 0).sum()),
                "lgb_n": int((lgb_w > 0).sum()),
                "intersection_n": int(len(set(xgb_w.index) & set(lgb_w.index))),
                "xgb_top_k": getattr(xgb_fit["shape"], "top_k", np.nan),
                "xgb_rank_power": getattr(xgb_fit["shape"], "rank_power", np.nan),
                "lgb_top_k": getattr(lgb_fit["shape"], "top_k", np.nan),
                "lgb_rank_power": getattr(lgb_fit["shape"], "rank_power", np.nan),
            }
        ]
    )
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values}), meta


def shape_from_args(args: argparse.Namespace) -> ConsensusShape:
    return ConsensusShape(
        top_k=args.top_k,
        alpha_xgb=args.alpha_xgb,
        rank_power=args.rank_power,
        equal_mix=args.equal_mix,
        max_weight=args.max_weight,
    )


def feature_set_from_args(args: argparse.Namespace) -> list[str]:
    if args.feature_set == "momentum":
        return MOMENTUM_FEATURE_COLUMNS
    if args.feature_set == "quality":
        return QUALITY_FEATURE_COLUMNS
    if args.feature_set == "reference":
        return REFERENCE_FEATURE_COLUMNS
    return FEATURE_COLUMNS


def target_from_args(args: argparse.Namespace) -> tuple[str, int]:
    if args.target_horizon == 3:
        return (TARGET_EXCESS_3D_COLUMN if args.target_mode == "excess" else TARGET_3D_COLUMN), 3
    return (TARGET_EXCESS_COLUMN if args.target_mode == "excess" else TARGET_COLUMN), FORWARD_HORIZON


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="stage2_report/route_outputs/stage2_tree_consensus.csv")
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha-xgb", type=float, default=DEFAULT_ALPHA_XGB)
    parser.add_argument("--rank-power", type=float, default=DEFAULT_RANK_POWER)
    parser.add_argument("--equal-mix", type=float, default=DEFAULT_EQUAL_MIX)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--defensive-equal-gate", action="store_true")
    parser.add_argument("--feature-set", choices=["core", "reference", "momentum", "quality"], default="core")
    parser.add_argument("--time-decay-half-life", type=int, default=0)
    parser.add_argument("--time-decay-floor", type=float, default=0.0)
    parser.add_argument("--adaptive-time-decay", action="store_true")
    parser.add_argument("--score-prop", action="store_true")
    parser.add_argument("--score-rank-blend", type=float, default=0.35)
    parser.add_argument("--target-horizon", type=int, choices=[3, 5], default=5)
    parser.add_argument("--target-mode", choices=["return", "excess"], default="return")
    parser.add_argument("--reweight-factor", default="")
    parser.add_argument("--reweight-direction", choices=["low", "high"], default="low")
    parser.add_argument("--reweight-gamma", type=float, default=0.0)
    parser.add_argument("--reweight-power", type=float, default=1.0)
    parser.add_argument(
        "--reweight-gate",
        choices=["none", "not_calm", "medium_move", "post_drawdown", "trend_dislocation"],
        default="none",
    )
    parser.add_argument(
        "--shape-horizon",
        type=int,
        choices=[0, 3, 5],
        default=0,
        help="Validation horizon for selecting internal top_k/rank_power; 0 follows target horizon.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])
    shape = shape_from_args(args)
    features = feature_set_from_args(args)
    half_life = args.time_decay_half_life if args.time_decay_half_life > 0 else None
    target_column, target_horizon = target_from_args(args)
    shape_horizon = target_horizon if args.shape_horizon == 0 else args.shape_horizon
    sub, meta = generate_tree_consensus(
        prices,
        index_df,
        as_of,
        shape,
        defensive_equal_gate=args.defensive_equal_gate,
        features=features,
        half_life=half_life,
        weight_floor=args.time_decay_floor,
        adaptive_time_decay=args.adaptive_time_decay,
        score_prop=args.score_prop,
        score_rank_blend=args.score_rank_blend,
        target_column=target_column,
        target_horizon=target_horizon,
        shape_horizon=shape_horizon,
        reweight_factor=args.reweight_factor,
        reweight_direction=args.reweight_direction,
        reweight_gamma=args.reweight_gamma,
        reweight_power=args.reweight_power,
        reweight_gate=args.reweight_gate,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    print(f">> model=stage2_tree_consensus as_of={as_of.date()} shape={shape}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
