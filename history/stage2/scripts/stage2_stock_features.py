"""
Stage2 stock-only feature enhancement.

The goal is to extract cleaner information from the original OHLCV panel before
trying more data sources.  Features are past-only and model-agnostic:
  - volatility-adjusted momentum,
  - trend quality / efficiency,
  - downside-risk and gap-risk measures,
  - liquidity and volume-price confirmation,
  - daily cross-sectional robust z-scores and ranks,
  - IC + correlation based feature selection on the train split only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import TARGET_COLUMN


RAW_STOCK_FEATURES = [
    "s2_ret_5d_vol_adj",
    "s2_ret_20d_vol_adj",
    "s2_ret_60d_vol_adj",
    "s2_trend_eff_20d",
    "s2_trend_eff_60d",
    "s2_up_day_ratio_20d",
    "s2_downside_vol_20d",
    "s2_upside_downside_20d",
    "s2_gap_risk_20d",
    "s2_intraday_stability_20d",
    "s2_vol_compression_20d",
    "s2_vol_expansion_5d",
    "s2_volume_price_confirm_5d",
    "s2_amount_price_confirm_5d",
    "s2_turnover_price_confirm_5d",
    "s2_liquidity_trend_20d",
    "s2_excess_5d_vol_adj",
    "s2_excess_20d_vol_adj",
    "s2_reversal_risk_5d",
    "s2_pullback_quality_20d",
    "s2_high_low_stress_5d",
    "s2_close_position_change_5d",
]


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected: list[str]
    report: pd.DataFrame


def _safe_div(num, den):
    den = den.replace(0, np.nan) if isinstance(den, pd.Series) else den
    return num / den


def _per_stock_enhanced(group: pd.DataFrame) -> pd.DataFrame:
    stock_code = getattr(group, "name", None)
    g = group.sort_values("date").copy()
    if "stock_code" not in g.columns and stock_code is not None:
        g["stock_code"] = str(stock_code).zfill(6)
    close = g["close"].astype(float)
    high = g["high"].astype(float) if "high" in g else close
    low = g["low"].astype(float) if "low" in g else close
    ret = g["ret_1d"].astype(float)
    abs_ret = ret.abs()

    vol_5 = ret.rolling(5).std()
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()
    path_20 = abs_ret.rolling(20).sum()
    path_60 = abs_ret.rolling(60).sum()

    g["s2_ret_5d_vol_adj"] = _safe_div(g["ret_5d"], vol_20)
    g["s2_ret_20d_vol_adj"] = _safe_div(g["ret_20d"], vol_60)
    g["s2_ret_60d_vol_adj"] = _safe_div(g["ret_60d"], vol_60)
    g["s2_trend_eff_20d"] = _safe_div(g["ret_20d"].abs(), path_20)
    g["s2_trend_eff_60d"] = _safe_div(g["ret_60d"].abs(), path_60)
    g["s2_up_day_ratio_20d"] = (ret > 0).rolling(20).mean()
    g["s2_downside_vol_20d"] = ret.where(ret < 0, 0.0).rolling(20).std()
    upside = ret.clip(lower=0).rolling(20).sum()
    downside = (-ret.clip(upper=0)).rolling(20).sum()
    g["s2_upside_downside_20d"] = _safe_div(upside, downside)
    g["s2_gap_risk_20d"] = g.get("overnight_ret", pd.Series(0.0, index=g.index)).rolling(20).std()
    g["s2_intraday_stability_20d"] = -g.get("intraday_ret", pd.Series(0.0, index=g.index)).abs().rolling(20).mean()
    g["s2_vol_compression_20d"] = -_safe_div(vol_20, vol_60)
    g["s2_vol_expansion_5d"] = _safe_div(vol_5, vol_20)

    volume_z = g.get("volume_z_20d", pd.Series(np.nan, index=g.index))
    amount_z = g.get("amount_z_20d", pd.Series(np.nan, index=g.index))
    turnover_z = g.get("turnover_z_20d", pd.Series(np.nan, index=g.index))
    g["s2_volume_price_confirm_5d"] = g["ret_5d"] * volume_z.clip(-4, 4)
    g["s2_amount_price_confirm_5d"] = g["ret_5d"] * amount_z.clip(-4, 4)
    g["s2_turnover_price_confirm_5d"] = g["ret_5d"] * turnover_z.clip(-4, 4)
    g["s2_liquidity_trend_20d"] = amount_z.rolling(5).mean() - amount_z.rolling(20).mean()

    g["s2_excess_5d_vol_adj"] = _safe_div(g.get("excess_ret_5d", g["ret_5d"]), vol_20)
    g["s2_excess_20d_vol_adj"] = _safe_div(g.get("excess_ret_20d", g["ret_20d"]), vol_60)
    g["s2_reversal_risk_5d"] = -g["ret_5d"] * g["s2_vol_expansion_5d"]
    g["s2_pullback_quality_20d"] = g["ret_20d"] - g["drawdown_20d"].abs()
    g["s2_high_low_stress_5d"] = (high / low.replace(0, np.nan) - 1.0).rolling(5).mean()
    if "close_pos_20d" in g:
        g["s2_close_position_change_5d"] = g["close_pos_20d"].diff(5)
    else:
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        close_pos = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        g["s2_close_position_change_5d"] = close_pos.diff(5)
    return g


def _robust_z_by_date(frame: pd.DataFrame, cols: list[str], clip: float = 4.0) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        median = out.groupby("date")[col].transform("median")
        mad = out.groupby("date")[col].transform(lambda s: (s - s.median()).abs().median())
        scale = (1.4826 * mad).replace(0, np.nan)
        out[f"{col}_z"] = ((out[col] - median) / scale).clip(-clip, clip)
    return out


def _rank_by_date(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[f"{col}_rank"] = out.groupby("date")[col].rank(method="average", pct=True)
    return out


def add_stage2_stock_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add enhanced stock-only features and return candidate model columns."""
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    try:
        out = out.groupby("stock_code", group_keys=False).apply(_per_stock_enhanced, include_groups=False).reset_index(drop=True)
    except TypeError:
        out = out.groupby("stock_code", group_keys=False).apply(_per_stock_enhanced).reset_index(drop=True)

    for col in RAW_STOCK_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lo = out.groupby("date")[col].transform(lambda s: s.quantile(0.01))
        hi = out.groupby("date")[col].transform(lambda s: s.quantile(0.99))
        out[col] = out[col].clip(lo, hi)

    out = _robust_z_by_date(out, RAW_STOCK_FEATURES)
    out = _rank_by_date(out, RAW_STOCK_FEATURES)
    candidates = []
    for col in RAW_STOCK_FEATURES:
        candidates.extend([f"{col}_z", f"{col}_rank"])

    for col in candidates:
        if col.endswith("_rank"):
            out[col] = out[col].fillna(0.5)
        else:
            out[col] = out[col].fillna(0.0)
    return out, candidates


def _daily_spearman_ic(df: pd.DataFrame, feature: str, target: str = TARGET_COLUMN) -> tuple[float, float, int]:
    vals = []
    for _, g in df[["date", feature, target]].dropna().groupby("date"):
        if len(g) < 30 or g[feature].nunique() < 3:
            continue
        rho, _ = spearmanr(g[feature], g[target])
        if np.isfinite(rho):
            vals.append(float(rho))
    if not vals:
        return 0.0, 0.0, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def select_stage2_stock_features(
    train_df: pd.DataFrame,
    candidate_cols: list[str],
    *,
    target_col: str = TARGET_COLUMN,
    max_features: int = 24,
    min_abs_ic: float = 0.003,
    corr_threshold: float = 0.90,
) -> FeatureSelectionResult:
    """Select features using train-only IC ranking and greedy correlation pruning."""
    rows = []
    for col in candidate_cols:
        coverage = float(train_df[col].notna().mean()) if col in train_df else 0.0
        std = float(train_df[col].std(skipna=True)) if col in train_df else 0.0
        mean_ic, std_ic, n_days = _daily_spearman_ic(train_df, col, target=target_col)
        rows.append({
            "feature": col,
            "coverage": coverage,
            "std": std,
            "mean_ic": mean_ic,
            "abs_mean_ic": abs(mean_ic),
            "std_ic": std_ic,
            "n_days": n_days,
            "selected": False,
            "drop_reason": "",
        })
    report = pd.DataFrame(rows).sort_values(["abs_mean_ic", "coverage"], ascending=False)
    eligible = report[
        (report["coverage"] >= 0.80)
        & (report["std"] > 1e-8)
        & (report["n_days"] >= 20)
        & (report["abs_mean_ic"] >= min_abs_ic)
    ].copy()

    selected: list[str] = []
    drop_reasons: dict[str, str] = {}
    sample = train_df[[c for c in eligible["feature"] if c in train_df]].copy()
    for feature in eligible["feature"]:
        if len(selected) >= max_features:
            drop_reasons[feature] = "max_features_reached"
            continue
        keep = True
        for prev in selected:
            corr = sample[[feature, prev]].corr(method="spearman").iloc[0, 1]
            if np.isfinite(corr) and abs(float(corr)) >= corr_threshold:
                drop_reasons[feature] = f"corr_with:{prev}:{corr:.3f}"
                keep = False
                break
        if keep:
            selected.append(feature)

    report["selected"] = report["feature"].isin(selected)
    report["drop_reason"] = report.apply(
        lambda r: "" if r["selected"] else drop_reasons.get(r["feature"], "low_ic_or_coverage"),
        axis=1,
    )
    return FeatureSelectionResult(selected=selected, report=report)
