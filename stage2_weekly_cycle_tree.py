"""Unified 5-day tree ensemble with weekly-cycle features.

This route is not a separate full-week-only model.  It trains on every
available five-trading-day target, while adding known-in-advance calendar and
weekly-cycle features so the learner can handle complete Monday-Friday weeks
more intelligently than a pure short-window momentum model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from features import (
    CALENDAR_FEATURE_COLUMNS,
    TARGET_EXCESS_COLUMN,
    WEEKLY_CYCLE_FEATURE_COLUMNS,
    build_features,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAX_WEIGHT = 0.10


@dataclass(frozen=True)
class PortfolioShape:
    top_k: int
    score_temperature: float
    rank_power: float
    score_rank_blend: float
    max_weight: float


def usable_features(panel: pd.DataFrame) -> list[str]:
    features = [c for c in WEEKLY_CYCLE_FEATURE_COLUMNS if c in panel.columns]
    # Remove target/diagnostic leakage-like columns if they were ever added to a
    # feature group by accident.
    blocked = {
        "target_3d",
        "target_5d",
        "target_excess_3d",
        "target_excess_5d",
        "idx_target_3d",
        "idx_target_5d",
    }
    return [c for c in dict.fromkeys(features) if c not in blocked]


def normalize_feature_panel(panel: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Cross-sectionally normalize stock features, preserving calendar state."""
    out = panel.copy()
    calendar = set(CALENDAR_FEATURE_COLUMNS)
    calendar.update(
        {
            "breadth_ret_5d_pos",
            "dispersion_ret_5d",
            "idx_ret_1d",
            "idx_ret_5d",
            "idx_ret_20d",
            "idx_vol_20d",
            "idx_ret_1d_mean_60",
            "idx_ret_1d_var_60",
        }
    )
    normalized: list[str] = []
    for col in features:
        if col not in out.columns:
            continue
        if col in calendar or col.endswith("_rank"):
            out[col] = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            out[col] = (
                out.groupby("date", sort=False)[col]
                .rank(method="average", pct=True)
                .sub(0.5)
            )
        normalized.append(col)
    return out, normalized


def correlation_filter(train: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    if not (0.0 < threshold < 1.0) or len(features) <= 1:
        return features
    numeric = train[features].replace([np.inf, -np.inf], np.nan)
    std = numeric.std(axis=0, ddof=0)
    features = [c for c in features if np.isfinite(std.get(c, np.nan)) and std.get(c, 0.0) > 1e-10]
    if len(features) <= 1:
        return features
    corr = numeric[features].corr(method="spearman").abs().fillna(0.0)
    keep: list[str] = []
    dropped: set[str] = set()
    for col in features:
        if col in dropped:
            continue
        keep.append(col)
        high = corr.index[(corr[col] > threshold) & (corr.index != col)]
        dropped.update(high)
    return keep


def sample_weights(train: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float, fullweek_boost: float) -> np.ndarray:
    age = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.astype(float)
    weights = np.power(0.5, age / max(float(half_life_days), 1.0))
    if "eval_is_full_workweek" in train:
        weights = weights * (1.0 + fullweek_boost * train["eval_is_full_workweek"].astype(float).clip(0, 1))
    if "eval_month_start" in train:
        weights = weights * (1.0 + 0.05 * train["eval_month_start"].astype(float).clip(0, 1))
    if "eval_month_end" in train:
        weights = weights * (1.0 + 0.05 * train["eval_month_end"].astype(float).clip(0, 1))
    return np.asarray(weights / max(float(weights.mean()), 1e-12), dtype=float)


def robust_target(y: pd.Series) -> pd.Series:
    lo, hi = y.quantile([0.01, 0.99])
    return y.clip(lo, hi).astype(float)


def train_models(
    train: pd.DataFrame,
    features: list[str],
    y: pd.Series,
    weights: np.ndarray,
    model_set: str,
) -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = []
    if model_set in {"lgb_xgb", "all"}:
        models.append(
            (
                "lgb",
                lgb.LGBMRegressor(
                    objective="huber",
                    alpha=0.85,
                    n_estimators=650,
                    learning_rate=0.022,
                    num_leaves=47,
                    min_child_samples=70,
                    subsample=0.82,
                    subsample_freq=1,
                    colsample_bytree=0.70,
                    reg_alpha=0.35,
                    reg_lambda=5.0,
                    random_state=2026,
                    verbosity=-1,
                    n_jobs=1,
                ),
            )
        )
        models.append(
            (
                "xgb",
                XGBRegressor(
                    objective="reg:pseudohubererror",
                    n_estimators=520,
                    learning_rate=0.022,
                    max_depth=3,
                    min_child_weight=45,
                    subsample=0.82,
                    colsample_bytree=0.70,
                    reg_alpha=0.20,
                    reg_lambda=6.0,
                    tree_method="hist",
                    random_state=2027,
                    n_jobs=1,
                ),
            )
        )
    if model_set in {"cat", "all"}:
        models.append(
            (
                "cat",
                CatBoostRegressor(
                    loss_function="Quantile:alpha=0.55",
                    iterations=520,
                    depth=5,
                    learning_rate=0.025,
                    l2_leaf_reg=10.0,
                    random_strength=1.0,
                    bagging_temperature=0.35,
                    random_seed=2028,
                    allow_writing_files=False,
                    thread_count=1,
                    verbose=False,
                ),
            )
        )
    if not models:
        raise ValueError(f"unknown model_set={model_set}")
    fitted: list[tuple[str, object]] = []
    for name, model in models:
        model.fit(train[features], y, sample_weight=weights)
        fitted.append((name, model))
    return fitted


def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def calendar_alpha(today: pd.DataFrame) -> pd.Series:
    codes = today["stock_code"].astype(str).str.zfill(6)
    frame = today.set_index(codes)
    score = pd.Series(0.0, index=frame.index)
    total = 0.0

    def add(col: str, direction: str, weight: float) -> None:
        nonlocal score, total
        if col not in frame.columns:
            return
        ranks = frame[col].astype(float).rank(method="average", pct=True)
        if direction == "low":
            ranks = 1.0 - ranks
        score = score.add(weight * ranks, fill_value=0.0)
        total += abs(weight)

    add("weekly_carry_quality_rank", "high", 0.24)
    add("obv_20d_rank", "high", 0.18)
    add("vol_ratio_5_20_rank", "low", 0.16)
    add("gap_mean_20d_rank", "high", 0.14)
    add("amount_z_20d_rank", "high", 0.10)
    add("trend_quality_20d_rank", "high", 0.08)
    if float(frame["eval_month_start"].iloc[0]) > 0:
        add("month_start_flow_rank", "high", 0.10)
    elif float(frame["eval_month_end"].iloc[0]) > 0:
        add("month_end_defensive_rank", "high", 0.10)
    else:
        add("post_gap_reopen_flow_rank", "high", 0.10)
    if total <= 0:
        return pd.Series(0.5, index=frame.index)
    return score / total


def cap_weights(raw: pd.Series, max_weight: float) -> pd.Series:
    w = raw[raw > 0].astype(float).copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} names")
    w = w / w.sum()
    for _ in range(100):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def portfolio_from_confidence(confidence: pd.Series, shape: PortfolioShape) -> pd.Series:
    chosen = confidence.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(shape.top_k)
    if len(chosen) < MIN_STOCKS:
        raise ValueError(f"only {len(chosen)} stocks available")
    ranks = pd.Series(np.arange(len(chosen), 0, -1, dtype=float), index=chosen.index)
    rank_component = ranks.pow(shape.rank_power)
    z = chosen - float(chosen.median())
    scale = float((chosen - chosen.median()).abs().median()) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(chosen.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        score_component = pd.Series(np.ones(len(chosen)), index=chosen.index)
    else:
        score_component = np.exp((z / scale).clip(-3, 3) * shape.score_temperature)
    raw = (
        shape.score_rank_blend * score_component / score_component.sum()
        + (1.0 - shape.score_rank_blend) * rank_component / rank_component.sum()
    )
    return cap_weights(raw, max_weight=shape.max_weight)


def fit_predict(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    corr_threshold: float,
    half_life_days: float,
    fullweek_boost: float,
    model_set: str,
) -> tuple[pd.Series, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(prices, index_df)
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    features = usable_features(panel)
    panel, features = normalize_feature_panel(panel, features)

    train = panel.dropna(subset=features + [TARGET_EXCESS_COLUMN]).copy()
    train = train[train["date"] < as_of].copy()
    pred = panel[panel["date"] == as_of].dropna(subset=features).copy()
    if train.empty or pred.empty:
        raise RuntimeError(f"empty train/pred frame for {as_of.date()}")

    features = correlation_filter(train, features, corr_threshold)
    train = train.dropna(subset=features + [TARGET_EXCESS_COLUMN]).copy()
    pred = pred.dropna(subset=features).copy()

    y = robust_target(train[TARGET_EXCESS_COLUMN])
    weights = sample_weights(train, as_of=as_of, half_life_days=half_life_days, fullweek_boost=fullweek_boost)
    models = train_models(train, features, y, weights, model_set=model_set)

    codes = pred["stock_code"].astype(str).str.zfill(6)
    score = pd.Series(0.0, index=codes)
    model_rows = []
    for name, model in models:
        pred_score = pd.Series(model.predict(pred[features]), index=codes)
        score = score.add(rank_pct(pred_score), fill_value=0.0)
        model_rows.append({"model": name, "features": len(features), "train_rows": len(train)})
    score = score / len(models)
    meta = pd.DataFrame(model_rows)
    return score, meta


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    shape: PortfolioShape,
    corr_threshold: float = 0.90,
    half_life_days: float = 180.0,
    fullweek_boost: float = 0.20,
    model_set: str = "lgb_xgb",
    alpha_blend: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score, model_meta = fit_predict(
        prices,
        index_df,
        as_of,
        corr_threshold=corr_threshold,
        half_life_days=half_life_days,
        fullweek_boost=fullweek_boost,
        model_set=model_set,
    )
    panel = build_features(prices[prices["date"] <= pd.Timestamp(as_of)].copy(), index_df[index_df["date"] <= pd.Timestamp(as_of)].copy())
    today = panel[panel["date"] == pd.Timestamp(as_of)].copy()
    alpha = calendar_alpha(today)
    common = sorted(set(score.index) & set(alpha.index))
    confidence = (1.0 - alpha_blend) * score.reindex(common) + alpha_blend * alpha.reindex(common)
    weights = portfolio_from_confidence(confidence, shape)
    sub = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    diagnostics = pd.DataFrame(
        {
            "stock_code": confidence.sort_values(ascending=False).head(80).index,
            "confidence": confidence.sort_values(ascending=False).head(80).values,
            "model_rank_score": score.reindex(confidence.sort_values(ascending=False).head(80).index).values,
            "calendar_alpha": alpha.reindex(confidence.sort_values(ascending=False).head(80).index).values,
        }
    )
    meta = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                "model": "stage2_weekly_cycle_tree",
                **shape.__dict__,
                "corr_threshold": corr_threshold,
                "half_life_days": half_life_days,
                "fullweek_boost": fullweek_boost,
                "model_set": model_set,
                "alpha_blend": alpha_blend,
                "base_models": "|".join(model_meta["model"].tolist()),
                "train_rows": int(model_meta["train_rows"].max()),
                "features": int(model_meta["features"].max()),
                "n_names": len(sub),
                "max_observed_weight": float(sub["weight"].max()),
                "effective_n": float(1.0 / np.square(sub["weight"].to_numpy()).sum()),
            }
        ]
    )
    return sub, meta, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--score-temperature", type=float, default=0.80)
    parser.add_argument("--rank-power", type=float, default=3.0)
    parser.add_argument("--score-rank-blend", type=float, default=0.60)
    parser.add_argument("--max-weight", type=float, default=0.08)
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--fullweek-boost", type=float, default=0.20)
    parser.add_argument("--model-set", choices=["lgb_xgb", "cat", "all"], default="lgb_xgb")
    parser.add_argument("--alpha-blend", type=float, default=0.25)
    parser.add_argument("--out", default="submissions/stage2/route_outputs/stage2_weekly_cycle_tree.csv")
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--diagnostics-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])
    shape = PortfolioShape(
        top_k=args.top_k,
        score_temperature=args.score_temperature,
        rank_power=args.rank_power,
        score_rank_blend=args.score_rank_blend,
        max_weight=args.max_weight,
    )
    sub, meta, diagnostics = generate_submission(
        prices,
        index_df,
        as_of,
        shape=shape,
        corr_threshold=args.corr_threshold,
        half_life_days=args.half_life_days,
        fullweek_boost=args.fullweek_boost,
        model_set=args.model_set,
        alpha_blend=args.alpha_blend,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    if args.diagnostics_out:
        diag_path = Path(args.diagnostics_out)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(diag_path, index=False)
    print(f">> model=stage2_weekly_cycle_tree as_of={pd.Timestamp(as_of).date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
