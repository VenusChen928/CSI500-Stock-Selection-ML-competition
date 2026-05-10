"""Hybrid gated Stage2 portfolio.

This route keeps the strongest stage2 idea so far: use simple, regularized
tree consensus as the default, and only switch models when the market regime
clearly favors a different error profile.

Routes:
* regularized consensus for defensive/equal-universe tape and mild post-rally
  setups, where the default tree route is too flat or too exposed to reversal;
* LSTM rank-weight for broad capitulation, where sequence reversal has shown
  useful upside;
* tree consensus with drawdown overlay everywhere else.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON
from features import FEATURE_COLUMNS, QUALITY_FEATURE_COLUMNS, TARGET_EXCESS_COLUMN, build_features
from lstm_rank_weight import fit_lstm, generate_submission as generate_lstm_submission
from stage2_regularized_consensus import (
    fit_regularized_consensus,
    generate_submission as generate_regularized_submission,
)
from stage2_tree_consensus import (
    ConsensusShape,
    defensive_guard,
    factor_reweight,
    generate_tree_consensus,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REGIME_ALPHA_MAX_WEIGHT = 0.08
LIQUIDITY_ALPHA_MAX_WEIGHT = 0.10
MIN_PORTFOLIO_NAMES = 30
COMPETITION_MAX_WEIGHT = 0.10
FINAL_CONFIDENCE_MAX_WEIGHT = 0.10


def tree_features(feature_set: str) -> list[str]:
    if feature_set == "quality":
        return QUALITY_FEATURE_COLUMNS
    if feature_set == "core":
        return FEATURE_COLUMNS
    raise ValueError(f"unknown tree feature set: {feature_set}")


def quality_tree_gate(decision) -> bool:
    """Narrow challenger route for shallow selloffs on flat medium tape."""
    return (
        -0.04 < decision.idx_ret_5d < -0.015
        and 0.0 < decision.idx_ret_20d < 0.03
        and 0.25 < decision.breadth_ret_5d_pos < 0.35
    )


def resolve_tree_feature_set(feature_set: str, decision) -> str:
    if feature_set == "auto":
        return "quality" if quality_tree_gate(decision) else "core"
    tree_features(feature_set)
    return feature_set


def adaptive_tree_base_max_weight(decision) -> float:
    """Use a more concentrated first-stage tree portfolio only in strong tape."""
    if decision.idx_ret_20d > 0.08 and decision.idx_ret_5d > -0.04:
        return 0.08
    return 0.04


def regularized_gate(decision) -> tuple[bool, str]:
    defensive_equal = decision.route == "equal_universe"
    mild_post_rally = (
        decision.idx_ret_5d > 0.03
        and 0.0 < decision.idx_ret_20d < 0.03
        and 0.55 < decision.breadth_ret_5d_pos < 0.75
    )
    if defensive_equal:
        return True, "regularized_defensive_equal_tape"
    if mild_post_rally:
        return True, "regularized_mild_post_rally"
    return False, "regularized_off"


def lstm_gate(decision) -> tuple[bool, str]:
    capitulation = (
        decision.idx_ret_5d < -0.07
        and decision.idx_ret_20d < -0.07
        and decision.breadth_ret_5d_pos < 0.15
    )
    if capitulation:
        return True, "lstm_capitulation_reversal"
    return False, "lstm_off"


def tree_route(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    route_reason: str,
    feature_set: str = "core",
    top_k: int = 30,
    rank_power: float = 1.6,
    base_max_weight: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape = ConsensusShape(top_k=top_k, alpha_xgb=0.05, rank_power=rank_power, equal_mix=0.0, max_weight=base_max_weight)
    sub, meta = generate_tree_consensus(
        prices,
        index_df,
        as_of=as_of,
        shape=shape,
        defensive_equal_gate=True,
        half_life=120,
        weight_floor=0.5,
        adaptive_time_decay=True,
        features=tree_features(feature_set),
        reweight_factor="drawdown_20d",
        reweight_direction="low",
        reweight_gamma=1.5,
        reweight_power=1.0,
        reweight_gate="medium_move",
    )
    meta = meta.copy()
    meta["final_route"] = "tree_consensus_drawdown_overlay"
    meta["route_reason"] = route_reason
    meta["tree_feature_set"] = feature_set
    meta["n_names"] = len(sub)
    return sub, meta


def factor_route_gate(decision) -> tuple[str, str, int, float, float] | None:
    """Small set of interpretable factor routes for regimes where trees lag.

    These routes are intentionally narrow and only use as-of features.  They are
    different from post-selection alpha: the factor route is allowed to choose a
    different stock pool when the learned tree pool appears mismatched to the
    market setup.
    """
    mild_flat_rebound = (
        0.0 < decision.idx_ret_5d < 0.03
        and -0.02 < decision.idx_ret_20d < 0.02
        and 0.40 < decision.breadth_ret_5d_pos < 0.60
    )
    strong_medium_pullback = (
        -0.04 < decision.idx_ret_5d < 0.0
        and decision.idx_ret_20d > 0.08
        and decision.breadth_ret_5d_pos < 0.35
    )
    defensive_weak_flat_tape = (
        decision.idx_ret_5d < -0.03
        and -0.02 < decision.idx_ret_20d < 0.02
        and decision.breadth_ret_5d_pos < 0.25
    )
    strong_rebound_weak_medium = (
        decision.idx_ret_5d > 0.055
        and decision.idx_ret_20d < 0.02
        and decision.breadth_ret_5d_pos > 0.70
    )
    mild_post_rally = (
        decision.idx_ret_5d > 0.03
        and 0.0 < decision.idx_ret_20d < 0.03
        and 0.55 < decision.breadth_ret_5d_pos < 0.75
    )
    if mild_flat_rebound:
        return ("overnight_ret", "low", 30, 2.0, 0.10)
    if strong_medium_pullback:
        return ("intraday_mean_5d", "low", 30, 2.0, 0.10)
    if defensive_weak_flat_tape:
        return ("turnover_ma_20d", "low", 40, 0.7, 0.10)
    if strong_rebound_weak_medium:
        return ("downside_vol_20d", "high", 30, 2.0, 0.10)
    if mild_post_rally:
        return ("obv_20d", "high", 30, 2.0, 0.10)
    return None


def factor_route(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    decision,
    spec: tuple[str, str, int, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor, direction, top_k, rank_power, max_weight = spec
    panel = build_features(prices, index_df)
    today = panel[panel["date"] == pd.Timestamp(as_of)].dropna(subset=[factor]).copy()
    if len(today) < top_k:
        raise ValueError(f"not enough names for factor route {factor}: {len(today)}")
    scores = today.set_index(today["stock_code"].astype(str).str.zfill(6))[factor].astype(float)
    ranks = scores.rank(method="average", pct=True)
    confidence = ranks if direction == "high" else 1.0 - ranks
    chosen = confidence.sort_values(ascending=False).head(top_k)
    raw = pd.Series(np.arange(len(chosen), 0, -1, dtype=float) ** rank_power, index=chosen.index)
    weights = cap_weights(raw / raw.sum(), max_weight=max_weight)
    sub = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    meta = pd.DataFrame(
        [
            {
                **decision.__dict__,
                "final_route": "factor_route",
                "route_reason": f"factor_{factor}_{direction}",
                "factor_route_factor": factor,
                "factor_route_direction": direction,
                "factor_route_top_k": top_k,
                "factor_route_rank_power": rank_power,
                "n_names": len(sub),
                "max_weight": float(sub["weight"].max()),
            }
        ]
    )
    return sub, meta


def _rank_ic(frame: pd.DataFrame, factor: str, direction: str, target: str = TARGET_EXCESS_COLUMN) -> float:
    tmp = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 80:
        return np.nan
    score = tmp[factor].astype(float)
    if direction == "low":
        score = -score
    if score.nunique(dropna=True) < 5 or tmp[target].nunique(dropna=True) < 5:
        return np.nan
    return float(score.rank(method="average").corr(tmp[target].rank(method="average")))


def _top_spread(frame: pd.DataFrame, factor: str, direction: str, top_k: int, target: str = TARGET_EXCESS_COLUMN) -> float:
    tmp = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(tmp) < max(80, top_k):
        return np.nan
    score = tmp[factor].astype(float)
    tmp["_score"] = score if direction == "high" else -score
    top = tmp.nlargest(top_k, "_score")
    return float(top[target].mean() - tmp[target].mean())


def factor_route_ic_support(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    spec: tuple[str, str, int, float, float],
    lookback_days: int = 60,
    min_ic: float = 0.0,
    min_pos_rate: float = 0.45,
) -> tuple[bool, dict]:
    """Use trailing validation IC to filter fragile one-window factor routes.

    The panel is built on data already truncated to as_of by the caller.  The
    most recent 5 dates have unknown targets and are therefore excluded by the
    ``dropna`` logic below, keeping this check production-safe.
    """
    factor, direction, top_k, _, _ = spec
    panel = build_features(prices, index_df)
    panel["date"] = pd.to_datetime(panel["date"])
    known = panel.dropna(subset=[factor, TARGET_EXCESS_COLUMN]).copy()
    dates = sorted(known["date"].dropna().unique())
    dates = [pd.Timestamp(date) for date in dates if pd.Timestamp(date) < pd.Timestamp(as_of)]
    dates = dates[-lookback_days:]
    ics = []
    spreads = []
    for date in dates:
        day = known[known["date"] == date]
        ic = _rank_ic(day, factor, direction)
        if not np.isnan(ic):
            ics.append(ic)
        spread = _top_spread(day, factor, direction, top_k)
        if not np.isnan(spread):
            spreads.append(spread)
    if not ics:
        stats = {
            "factor_ic_filter_enabled": True,
            "factor_ic_supported": False,
            "factor_ic_mean": np.nan,
            "factor_ic_pos_rate": np.nan,
            "factor_ic_days": 0,
            "factor_ic_top_spread_mean": np.nan,
        }
        return False, stats
    ic_arr = np.asarray(ics, dtype=float)
    spread_arr = np.asarray(spreads, dtype=float)
    ic_mean = float(ic_arr.mean())
    pos_rate = float((ic_arr > 0).mean())
    spread_mean = float(spread_arr.mean()) if len(spread_arr) else np.nan
    supported = bool(ic_mean >= min_ic and pos_rate >= min_pos_rate)
    stats = {
        "factor_ic_filter_enabled": True,
        "factor_ic_supported": supported,
        "factor_ic_mean": ic_mean,
        "factor_ic_pos_rate": pos_rate,
        "factor_ic_days": int(len(ic_arr)),
        "factor_ic_top_spread_mean": spread_mean,
    }
    return supported, stats


def dampen_factor_spec(spec: tuple[str, str, int, float, float]) -> tuple[str, str, int, float, float]:
    factor, direction, top_k, rank_power, max_weight = spec
    return (factor, direction, top_k, min(rank_power, 1.0), max_weight)


def tree_regime_alpha(decision) -> tuple[str, str, float, float, float] | None:
    """Return a post-selection factor tilt for specific tree-route regimes.

    These alphas are intentionally applied after the tree models have selected
    names.  That keeps the learned ranking stable while allowing regime-specific
    confidence weighting:
    - weak short tape without medium support: favor positive overnight demand;
    - strong rebound with still-weak medium tape: favor one-day pullback names;
    - medium uptrend pullback: favor longer-horizon laggards/pullbacks;
    - mild flat rebound: favor 10-day pullback catch-up.
    """
    weak_short_tape = (
        decision.idx_ret_5d < 0.0
        and decision.idx_ret_20d < 0.02
        and decision.breadth_ret_5d_pos < 0.55
    )
    rebound_after_weak_medium = (
        decision.idx_ret_5d > 0.055
        and decision.idx_ret_20d < 0.02
        and decision.breadth_ret_5d_pos > 0.70
    )
    medium_uptrend_pullback = (
        decision.idx_ret_5d < 0.0
        and decision.idx_ret_20d > 0.05
        and decision.breadth_ret_5d_pos < 0.45
    )
    mild_flat_rebound = (
        0.0 < decision.idx_ret_5d < 0.03
        and -0.02 < decision.idx_ret_20d < 0.02
        and 0.40 < decision.breadth_ret_5d_pos < 0.60
    )
    if weak_short_tape:
        return ("overnight_ret", "high", 5.0, 3.0, 0.10)
    if rebound_after_weak_medium:
        return ("ret_1d", "low", 5.0, 1.0, 0.10)
    if medium_uptrend_pullback:
        return ("ret_60d", "low", 5.0, 1.0, 0.10)
    if mild_flat_rebound:
        return ("ret_10d", "low", 3.0, 2.0, REGIME_ALPHA_MAX_WEIGHT)
    return None


def apply_tree_regime_alpha(
    sub: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    decision,
) -> tuple[pd.DataFrame, dict]:
    alpha = tree_regime_alpha(decision)
    if alpha is None:
        return sub, {
            "regime_alpha_applied": False,
            "regime_alpha_factor": "",
            "regime_alpha_direction": "",
            "regime_alpha_gamma": 0.0,
            "regime_alpha_power": 0.0,
            "regime_alpha_max_weight": 0.0,
        }
    factor, direction, gamma, power, max_weight = alpha
    panel = build_features(prices, index_df)
    weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
    tilted = factor_reweight(
        weights=weights,
        panel=panel,
        as_of=as_of,
        factor=factor,
        direction=direction,
        gamma=gamma,
        power=power,
        max_weight=max_weight,
    )
    out = pd.DataFrame({"stock_code": tilted.index, "weight": tilted.values})
    return out, {
        "regime_alpha_applied": True,
        "regime_alpha_factor": factor,
        "regime_alpha_direction": direction,
        "regime_alpha_gamma": gamma,
        "regime_alpha_power": power,
        "regime_alpha_max_weight": max_weight,
    }


def tree_liquidity_alpha(decision) -> tuple[str, str, float, float, float] | None:
    """Secondary confidence tilt for regimes where volume spikes hurt stability.

    This is deliberately post-selection: the tree ensemble still chooses the
    stocks, while this layer avoids giving too much size to names with unusual
    recent trading-amount spikes in regimes where that signal was unstable.
    """
    weak_low_breadth = decision.idx_ret_5d < -0.02 and decision.breadth_ret_5d_pos < 0.35
    mild_flat_rebound = (
        0.0 < decision.idx_ret_5d < 0.03
        and -0.02 < decision.idx_ret_20d < 0.02
        and 0.40 < decision.breadth_ret_5d_pos < 0.60
    )
    bear_market_snapback = (
        decision.idx_ret_5d > 0.03
        and decision.idx_ret_20d < -0.05
        and decision.breadth_ret_5d_pos > 0.75
    )
    strong_medium_trend_follow_through = (
        decision.idx_ret_5d > 0.0
        and decision.idx_ret_20d > 0.08
        and decision.breadth_ret_5d_pos > 0.50
    )
    if (
        weak_low_breadth
        or mild_flat_rebound
        or bear_market_snapback
        or strong_medium_trend_follow_through
    ):
        return ("amount_z_20d", "low", 5.0, 1.0, LIQUIDITY_ALPHA_MAX_WEIGHT)
    return None


def apply_tree_liquidity_alpha(
    sub: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    decision,
) -> tuple[pd.DataFrame, dict]:
    alpha = tree_liquidity_alpha(decision)
    if alpha is None:
        return sub, {
            "liquidity_alpha_applied": False,
            "liquidity_alpha_factor": "",
            "liquidity_alpha_direction": "",
            "liquidity_alpha_gamma": 0.0,
            "liquidity_alpha_power": 0.0,
            "liquidity_alpha_max_weight": 0.0,
        }
    factor, direction, gamma, power, max_weight = alpha
    panel = build_features(prices, index_df)
    weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
    tilted = factor_reweight(
        weights=weights,
        panel=panel,
        as_of=as_of,
        factor=factor,
        direction=direction,
        gamma=gamma,
        power=power,
        max_weight=max_weight,
    )
    out = pd.DataFrame({"stock_code": tilted.index, "weight": tilted.values})
    return out, {
        "liquidity_alpha_applied": True,
        "liquidity_alpha_factor": factor,
        "liquidity_alpha_direction": direction,
        "liquidity_alpha_gamma": gamma,
        "liquidity_alpha_power": power,
        "liquidity_alpha_max_weight": max_weight,
    }


def tree_secondary_alpha(decision) -> tuple[str, str, float, float, float] | None:
    weak_short_flat_medium = (
        decision.idx_ret_5d < -0.02
        and -0.02 < decision.idx_ret_20d < 0.03
        and decision.breadth_ret_5d_pos < 0.35
    )
    weak_short_strong_medium = (
        decision.idx_ret_5d < -0.02
        and decision.idx_ret_20d > 0.08
        and decision.breadth_ret_5d_pos < 0.35
    )
    bear_market_snapback = (
        decision.idx_ret_5d > 0.03
        and decision.idx_ret_20d < -0.05
        and decision.breadth_ret_5d_pos > 0.75
    )
    strong_medium_trend_follow_through = (
        decision.idx_ret_5d > 0.0
        and decision.idx_ret_20d > 0.08
        and decision.breadth_ret_5d_pos > 0.50
    )
    rebound_after_weak_medium = (
        decision.idx_ret_5d > 0.055
        and decision.idx_ret_20d < 0.02
        and decision.breadth_ret_5d_pos > 0.70
    )
    moderate_medium_pullback = (
        decision.idx_ret_5d < 0.0
        and 0.04 < decision.idx_ret_20d < 0.08
        and decision.breadth_ret_5d_pos > 0.35
    )
    if weak_short_flat_medium:
        return ("intraday_ret", "low", 5.0, 3.0, 0.10)
    if weak_short_strong_medium:
        return ("close_over_ma60", "low", 5.0, 1.5, 0.10)
    if bear_market_snapback:
        return ("intraday_ret", "high", 5.0, 3.0, 0.10)
    if strong_medium_trend_follow_through:
        return ("drawdown_20d", "low", 5.0, 1.5, 0.10)
    if rebound_after_weak_medium:
        return ("ret_1d", "low", 5.0, 1.0, 0.10)
    if moderate_medium_pullback:
        return ("overnight_ret", "low", 5.0, 3.0, 0.10)
    return None


def apply_route_alpha(
    sub: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    alpha: tuple[str, str, float, float, float] | None,
    prefix: str,
) -> tuple[pd.DataFrame, dict]:
    if alpha is None:
        return sub, {
            f"{prefix}_applied": False,
            f"{prefix}_factor": "",
            f"{prefix}_direction": "",
            f"{prefix}_gamma": 0.0,
            f"{prefix}_power": 0.0,
            f"{prefix}_max_weight": 0.0,
        }
    factor, direction, gamma, power, max_weight = alpha
    panel = build_features(prices, index_df)
    weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
    tilted = factor_reweight(
        weights=weights,
        panel=panel,
        as_of=as_of,
        factor=factor,
        direction=direction,
        gamma=gamma,
        power=power,
        max_weight=max_weight,
    )
    out = pd.DataFrame({"stock_code": tilted.index, "weight": tilted.values})
    return out, {
        f"{prefix}_applied": True,
        f"{prefix}_factor": factor,
        f"{prefix}_direction": direction,
        f"{prefix}_gamma": gamma,
        f"{prefix}_power": power,
        f"{prefix}_max_weight": max_weight,
    }


def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights[weights > 0].astype(float).copy()
    if len(w) < MIN_PORTFOLIO_NAMES:
        raise ValueError(f"portfolio must contain at least {MIN_PORTFOLIO_NAMES} names")
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


def attach_meta(meta: pd.DataFrame, extra: dict | None) -> pd.DataFrame:
    if not extra:
        return meta
    out = meta.copy()
    for key, value in extra.items():
        out[key] = value
    return out


def concentrate_regularized_portfolio(sub: pd.DataFrame, top_k: int = MIN_PORTFOLIO_NAMES) -> pd.DataFrame:
    weights = sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"].astype(float)
    selected = weights.sort_values(ascending=False).head(top_k)
    concentrated = cap_weights(selected, COMPETITION_MAX_WEIGHT)
    return pd.DataFrame({"stock_code": concentrated.index, "weight": concentrated.values})


def regularized_route_alpha(route_reason: str) -> tuple[str, str, float, float, float] | None:
    if route_reason == "regularized_mild_post_rally":
        return ("amount_z_20d", "high", 5.0, 3.0, 0.06)
    if route_reason == "regularized_defensive_equal_tape":
        return ("intraday_ret", "low", 5.0, 3.0, 0.06)
    return None


def lstm_route_alpha(route_reason: str) -> tuple[str, str, float, float, float] | None:
    if route_reason == "lstm_capitulation_reversal":
        return ("close_over_ma60", "high", 5.0, 3.0, 0.10)
    return None


def final_confidence_alpha(decision, final_route: str) -> tuple[str, str, float, float, float] | None:
    """Light momentum confirmation layer for weak/flat medium-tape regimes.

    This is intentionally mild.  In validation it helped the weakest
    regularized windows without changing the portfolio names.  We skip LSTM
    capitulation and mild flat rebounds because those routes already rely on
    different reversal/rebound signals.
    """
    if "lstm" in final_route:
        return None
    mild_flat_rebound = (
        0.0 < decision.idx_ret_5d < 0.03
        and -0.02 < decision.idx_ret_20d < 0.02
        and 0.40 < decision.breadth_ret_5d_pos < 0.60
    )
    if not mild_flat_rebound and decision.idx_ret_20d < 0.06:
        return ("ret_20d", "high", 0.5, 1.0, FINAL_CONFIDENCE_MAX_WEIGHT)
    return None


def apply_final_confidence_alpha(
    sub: pd.DataFrame,
    meta: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    decision,
    enable_alpha: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_route = str(meta["final_route"].iloc[0]) if "final_route" in meta else ""
    alpha = final_confidence_alpha(decision, final_route) if enable_alpha else None
    sub, final_meta = apply_route_alpha(
        sub,
        prices,
        index_df,
        as_of,
        alpha,
        "final_confidence_alpha",
    )
    meta = meta.copy()
    for key, value in final_meta.items():
        meta[key] = value
    if final_meta["final_confidence_alpha_applied"]:
        meta["final_route"] = meta["final_route"].astype(str) + "_final_confidence_alpha"
    meta["n_names"] = len(sub)
    meta["max_weight"] = float(sub["weight"].max())
    return sub, meta


def regularized_route(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    decision,
    route_reason: str,
    enable_alpha: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = fit_regularized_consensus(prices, index_df, as_of=as_of)
    sub = generate_regularized_submission(fit, as_of)
    sub, route_alpha_meta = apply_route_alpha(
        sub,
        prices,
        index_df,
        as_of,
        regularized_route_alpha(route_reason) if enable_alpha else None,
        "regularized_alpha",
    )
    original_n_names = len(sub)
    if enable_alpha:
        sub = concentrate_regularized_portfolio(sub)
    split = fit["split_dates"]
    meta = pd.DataFrame(
        [
            {
                **decision.__dict__,
                "final_route": "regularized_consensus",
                "route_reason": route_reason,
                "n_names": len(sub),
                "max_weight": float(sub["weight"].max()),
                "regularized_original_n_names": original_n_names,
                "regularized_concentrated_top_k": len(sub) if enable_alpha else 0,
                "regularized_concentration_max_weight": COMPETITION_MAX_WEIGHT if enable_alpha else 0.0,
                "selected_features": len(fit["features"]),
                "shape": str(fit["shape"]),
                "model_weights": fit["model_weights"].round(4).to_dict(),
                "train_end": split.train_end.date().isoformat(),
                "shape_start": split.shape_start.date().isoformat(),
                **route_alpha_meta,
            }
        ]
    )
    if route_alpha_meta["regularized_alpha_applied"]:
        meta["final_route"] = meta["final_route"].astype(str) + "_route_alpha"
    return sub, meta


def lstm_route(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    decision,
    route_reason: str,
    enable_alpha: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = fit_lstm(prices, index_df, as_of, policy_horizon=FORWARD_HORIZON, target_horizon=FORWARD_HORIZON)
    sub = generate_lstm_submission(fit, as_of)
    sub, route_alpha_meta = apply_route_alpha(
        sub,
        prices,
        index_df,
        as_of,
        lstm_route_alpha(route_reason) if enable_alpha else None,
        "lstm_alpha",
    )
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
                **route_alpha_meta,
            }
        ]
    )
    if route_alpha_meta["lstm_alpha_applied"]:
        meta["final_route"] = meta["final_route"].astype(str) + "_route_alpha"
    return sub, meta


def generate_hybrid_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    force_route: str = "auto",
    alpha_mode: str = "full",
    tree_feature_set: str = "auto",
    tree_top_k: int = 30,
    tree_rank_power: float = 1.6,
    tree_base_max_weight: float = -1.0,
    factor_ic_filter: bool = False,
    factor_ic_dampen: bool = False,
    disable_factor_route: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if alpha_mode not in {"full", "none", "no_regime", "no_liquidity", "no_secondary", "no_route", "no_final"}:
        raise ValueError(f"unknown alpha_mode: {alpha_mode}")
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    decision = defensive_guard(prices, index_df, as_of)
    resolved_tree_feature_set = resolve_tree_feature_set(tree_feature_set, decision)
    use_regularized, regularized_reason = regularized_gate(decision)
    use_lstm, lstm_reason = lstm_gate(decision)
    factor_spec = None if disable_factor_route else factor_route_gate(decision)
    factor_ic_meta: dict | None = None
    if factor_spec is not None and (factor_ic_filter or factor_ic_dampen):
        supported, factor_ic_meta = factor_route_ic_support(prices, index_df, as_of, factor_spec)
        if not supported:
            if factor_ic_dampen:
                old_factor, old_direction, old_top_k, old_rank_power, old_max_weight = factor_spec
                factor_spec = dampen_factor_spec(factor_spec)
                factor_ic_meta = {
                    **factor_ic_meta,
                    "factor_ic_route_skipped": False,
                    "factor_ic_route_dampened": True,
                    "factor_ic_original_factor": old_factor,
                    "factor_ic_original_direction": old_direction,
                    "factor_ic_original_top_k": old_top_k,
                    "factor_ic_original_rank_power": old_rank_power,
                    "factor_ic_original_max_weight": old_max_weight,
                }
            else:
                factor_ic_meta = {**factor_ic_meta, "factor_ic_route_skipped": True, "factor_ic_route_dampened": False}
                factor_spec = None
        else:
            factor_ic_meta = {**factor_ic_meta, "factor_ic_route_skipped": False, "factor_ic_route_dampened": False}
    route_alpha_on = alpha_mode not in {"none", "no_route"}
    final_alpha_on = alpha_mode not in {"none", "no_final"}
    resolved_tree_base_max_weight = (
        adaptive_tree_base_max_weight(decision)
        if tree_base_max_weight < 0
        else tree_base_max_weight
    )

    if force_route == "regularized":
        sub, meta = regularized_route(prices, index_df, as_of, decision, "forced_regularized", enable_alpha=route_alpha_on)
        return apply_final_confidence_alpha(sub, meta, prices, index_df, as_of, decision, enable_alpha=final_alpha_on)
    if force_route == "lstm":
        sub, meta = lstm_route(prices, index_df, as_of, decision, "forced_lstm", enable_alpha=route_alpha_on)
        return apply_final_confidence_alpha(sub, meta, prices, index_df, as_of, decision, enable_alpha=final_alpha_on)
    if force_route == "tree":
        sub, meta = tree_route(
            prices,
            index_df,
            as_of,
            "forced_tree",
            feature_set=resolved_tree_feature_set,
            top_k=tree_top_k,
            rank_power=tree_rank_power,
            base_max_weight=resolved_tree_base_max_weight,
        )
        return apply_final_confidence_alpha(sub, meta, prices, index_df, as_of, decision, enable_alpha=final_alpha_on)

    if factor_spec is not None:
        sub, meta = factor_route(prices, index_df, as_of, decision, factor_spec)
        return sub, attach_meta(meta, factor_ic_meta)
    if use_regularized:
        sub, meta = regularized_route(prices, index_df, as_of, decision, regularized_reason, enable_alpha=route_alpha_on)
        sub, meta = apply_final_confidence_alpha(sub, meta, prices, index_df, as_of, decision, enable_alpha=final_alpha_on)
        return sub, attach_meta(meta, factor_ic_meta)
    if use_lstm:
        sub, meta = lstm_route(prices, index_df, as_of, decision, lstm_reason, enable_alpha=route_alpha_on)
        sub, meta = apply_final_confidence_alpha(sub, meta, prices, index_df, as_of, decision, enable_alpha=final_alpha_on)
        return sub, attach_meta(meta, factor_ic_meta)
    sub, meta = tree_route(
        prices,
        index_df,
        as_of,
        "tree_default",
        feature_set=resolved_tree_feature_set,
        top_k=tree_top_k,
        rank_power=tree_rank_power,
        base_max_weight=resolved_tree_base_max_weight,
    )
    if alpha_mode in {"none", "no_regime"}:
        alpha_meta = {
            "regime_alpha_applied": False,
            "regime_alpha_factor": "",
            "regime_alpha_direction": "",
            "regime_alpha_gamma": 0.0,
            "regime_alpha_power": 0.0,
            "regime_alpha_max_weight": 0.0,
        }
    else:
        sub, alpha_meta = apply_tree_regime_alpha(sub, prices, index_df, as_of, decision)
    if alpha_mode in {"none", "no_liquidity"}:
        liquidity_meta = {
            "liquidity_alpha_applied": False,
            "liquidity_alpha_factor": "",
            "liquidity_alpha_direction": "",
            "liquidity_alpha_gamma": 0.0,
            "liquidity_alpha_power": 0.0,
            "liquidity_alpha_max_weight": 0.0,
        }
    else:
        sub, liquidity_meta = apply_tree_liquidity_alpha(sub, prices, index_df, as_of, decision)
    sub, secondary_meta = apply_route_alpha(
        sub,
        prices,
        index_df,
        as_of,
        tree_secondary_alpha(decision) if alpha_mode not in {"none", "no_secondary"} else None,
        "tree_secondary_alpha",
    )
    meta = meta.copy()
    for key, value in alpha_meta.items():
        meta[key] = value
    for key, value in liquidity_meta.items():
        meta[key] = value
    for key, value in secondary_meta.items():
        meta[key] = value
    if alpha_meta["regime_alpha_applied"]:
        meta["final_route"] = meta["final_route"].astype(str) + "_regime_alpha"
    if liquidity_meta["liquidity_alpha_applied"]:
        meta["final_route"] = meta["final_route"].astype(str) + "_liquidity_alpha"
    if secondary_meta["tree_secondary_alpha_applied"]:
        meta["final_route"] = meta["final_route"].astype(str) + "_secondary_alpha"
    meta["n_names"] = len(sub)
    sub, meta = apply_final_confidence_alpha(sub, meta, prices, index_df, as_of, decision, enable_alpha=final_alpha_on)
    return sub, attach_meta(meta, factor_ic_meta)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/route_outputs/stage2_hybrid_gate.csv")
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--force-route", choices=["auto", "tree", "lstm", "regularized"], default="auto")
    parser.add_argument("--tree-feature-set", choices=["auto", "core", "quality"], default="auto")
    parser.add_argument("--tree-top-k", type=int, default=30)
    parser.add_argument("--tree-rank-power", type=float, default=1.6)
    parser.add_argument("--tree-base-max-weight", type=float, default=-1.0, help="Use a negative value for the adaptive cap.")
    parser.add_argument(
        "--factor-ic-filter",
        action="store_true",
        help="Require trailing validation IC support before enabling narrow factor routes.",
    )
    parser.add_argument(
        "--factor-ic-dampen",
        action="store_true",
        help="Keep IC-weak factor routes but reduce their rank-power concentration.",
    )
    parser.add_argument(
        "--disable-factor-route",
        action="store_true",
        help="Skip direct factor stock-pool routes and fall back to model routes.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["full", "none", "no_regime", "no_liquidity", "no_secondary", "no_route", "no_final"],
        default="full",
        help="Ablation switch for post-selection confidence layers; default keeps production behavior.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, meta = generate_hybrid_submission(
        prices,
        index_df,
        as_of,
        force_route=args.force_route,
        alpha_mode=args.alpha_mode,
        tree_feature_set=args.tree_feature_set,
        tree_top_k=args.tree_top_k,
        tree_rank_power=args.tree_rank_power,
        tree_base_max_weight=args.tree_base_max_weight,
        factor_ic_filter=args.factor_ic_filter,
        factor_ic_dampen=args.factor_ic_dampen,
        disable_factor_route=args.disable_factor_route,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)

    print(f">> model=stage2_hybrid_gate as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
