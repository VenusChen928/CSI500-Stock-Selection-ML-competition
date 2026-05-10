"""Full-workweek Stage2 tree portfolio.

This challenger is trained only on as-of dates whose next five trading days are
a complete Monday-Friday week.  It avoids mixing ordinary weekly market moves
with holiday/weekend-split five-trading-day windows during model fitting.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from features import (
    ALPHA_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    QUALITY_FEATURE_COLUMNS,
    REFERENCE_FEATURE_COLUMNS,
    TARGET_EXCESS_COLUMN,
    build_features,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAX_WEIGHT = 0.10


@dataclass(frozen=True)
class FullWeekWindow:
    as_of: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


def complete_week_windows(dates: list[pd.Timestamp]) -> list[FullWeekWindow]:
    out: list[FullWeekWindow] = []
    for idx in range(len(dates) - FORWARD_HORIZON):
        eval_dates = dates[idx + 1 : idx + FORWARD_HORIZON + 1]
        if len(eval_dates) != FORWARD_HORIZON:
            continue
        if eval_dates[0].weekday() != 0 or eval_dates[-1].weekday() != 4:
            continue
        if (eval_dates[-1] - eval_dates[0]).days != 4:
            continue
        out.append(FullWeekWindow(dates[idx], eval_dates[0], eval_dates[-1]))
    return out


def feature_pool(panel: pd.DataFrame, feature_set: str) -> list[str]:
    groups = {
        "core": FEATURE_COLUMNS,
        "quality": QUALITY_FEATURE_COLUMNS,
        "reference": REFERENCE_FEATURE_COLUMNS,
        "momentum": MOMENTUM_FEATURE_COLUMNS,
        "alpha": ALPHA_FEATURE_COLUMNS,
        "all": list(
            dict.fromkeys(
                FEATURE_COLUMNS
                + REFERENCE_FEATURE_COLUMNS
                + MOMENTUM_FEATURE_COLUMNS
                + QUALITY_FEATURE_COLUMNS
                + ALPHA_FEATURE_COLUMNS
            )
        ),
    }
    if feature_set not in groups:
        raise ValueError(f"unknown feature_set={feature_set}")
    return [c for c in groups[feature_set] if c in panel.columns]


def rank_normalize(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = panel.copy()
    ranked = out.groupby("date", sort=False)[features].rank(pct=True)
    out[features] = ranked.sub(0.5).fillna(0.0)
    return out


def correlation_filter(frame: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    if threshold <= 0 or threshold >= 1 or len(features) <= 1:
        return features
    corr = frame[features].corr(method="spearman").abs()
    keep: list[str] = []
    dropped: set[str] = set()
    for col in features:
        if col in dropped:
            continue
        keep.append(col)
        high = corr.index[(corr[col] > threshold) & (corr.index != col)]
        dropped.update(high)
    return keep


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


def portfolio_from_scores(
    scores: pd.Series,
    *,
    top_k: int,
    rank_power: float,
    score_blend: float,
    max_weight: float,
) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(max(MIN_STOCKS, int(top_k)))
    ranks = pd.Series(np.arange(len(chosen), 0, -1, dtype=float), index=chosen.index)
    rank_component = ranks.pow(rank_power)
    shifted = chosen - float(chosen.min()) + 1e-6
    score_component = shifted.clip(lower=1e-6)
    raw = (1.0 - score_blend) * rank_component / rank_component.sum()
    raw = raw + score_blend * score_component / score_component.sum()
    return cap_weights(raw, max_weight=max_weight)


def fit_predict_scores(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    feature_set: str,
    model_type: str,
    corr_threshold: float,
    half_life_weeks: float,
) -> tuple[pd.Series, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(prices, index_df)
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    dates = [pd.Timestamp(d) for d in np.sort(panel["date"].dropna().unique())]
    windows = complete_week_windows(dates)
    known_asofs = [w.as_of for w in windows if w.end <= as_of and w.as_of < as_of]
    if len(known_asofs) < 20:
        raise RuntimeError(f"not enough complete-week training windows before {as_of.date()}: {len(known_asofs)}")

    features = feature_pool(panel, feature_set)
    panel = rank_normalize(panel, features)
    train = panel[panel["date"].isin(known_asofs)].dropna(subset=features + [TARGET_EXCESS_COLUMN]).copy()
    pred = panel[panel["date"] == as_of].dropna(subset=features).copy()
    if train.empty or pred.empty:
        raise RuntimeError(f"empty train/pred frame for {as_of.date()}")

    features = correlation_filter(train, features, corr_threshold)
    train = train.dropna(subset=features + [TARGET_EXCESS_COLUMN]).copy()
    pred = pred.dropna(subset=features).copy()

    y = train[TARGET_EXCESS_COLUMN].astype(float)
    lo, hi = y.quantile([0.01, 0.99])
    y = y.clip(lo, hi)
    week_rank = train["date"].rank(method="dense").astype(float)
    age = week_rank.max() - week_rank
    sample_weight = np.power(0.5, age / max(float(half_life_weeks), 1.0))

    models: list[tuple[str, object]] = []
    if model_type in {"lgb", "blend"}:
        models.append(
            (
                "lgb",
                lgb.LGBMRegressor(
                    objective="huber",
                    alpha=0.85,
                    n_estimators=520,
                    learning_rate=0.025,
                    num_leaves=63,
                    min_child_samples=45,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.75,
                    reg_alpha=0.15,
                    reg_lambda=2.5,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=1,
                ),
            )
        )
    if model_type in {"xgb", "blend"}:
        models.append(
            (
                "xgb",
                XGBRegressor(
                    objective="reg:pseudohubererror",
                    n_estimators=420,
                    learning_rate=0.025,
                    max_depth=4,
                    min_child_weight=30,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_alpha=0.05,
                    reg_lambda=3.0,
                    tree_method="hist",
                    random_state=7,
                    n_jobs=1,
                ),
            )
        )
    if not models:
        raise ValueError(f"unknown model_type={model_type}")

    score = pd.Series(0.0, index=pred["stock_code"].astype(str).str.zfill(6))
    meta_rows = []
    for name, model in models:
        model.fit(train[features], y, sample_weight=sample_weight)
        pred_score = pd.Series(model.predict(pred[features]), index=score.index)
        score = score.add(pred_score.rank(pct=True), fill_value=0.0)
        meta_rows.append({"model": name, "train_rows": len(train), "features": len(features)})
    score = score / len(models)
    meta = pd.DataFrame(meta_rows)
    return score, meta


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    feature_set: str = "all",
    model_type: str = "blend",
    top_k: int = 30,
    rank_power: float = 2.0,
    score_blend: float = 0.5,
    max_weight: float = 0.08,
    corr_threshold: float = 0.94,
    half_life_weeks: float = 52.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores, model_meta = fit_predict_scores(
        prices,
        index_df,
        as_of,
        feature_set=feature_set,
        model_type=model_type,
        corr_threshold=corr_threshold,
        half_life_weeks=half_life_weeks,
    )
    weights = portfolio_from_scores(
        scores,
        top_k=top_k,
        rank_power=rank_power,
        score_blend=score_blend,
        max_weight=max_weight,
    )
    sub = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    meta = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                "model": "stage2_fullweek_tree_portfolio",
                "model_type": model_type,
                "feature_set": feature_set,
                "top_k": top_k,
                "rank_power": rank_power,
                "score_blend": score_blend,
                "max_weight": max_weight,
                "corr_threshold": corr_threshold,
                "half_life_weeks": half_life_weeks,
                "n_names": len(sub),
                "max_observed_weight": float(sub["weight"].max()),
                "effective_n": float(1.0 / np.square(sub["weight"].to_numpy()).sum()),
                "train_rows": int(model_meta["train_rows"].max()),
                "features": int(model_meta["features"].max()),
                "base_models": "|".join(model_meta["model"].tolist()),
            }
        ]
    )
    return sub, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--feature-set", choices=["core", "quality", "reference", "momentum", "alpha", "all"], default="all")
    parser.add_argument("--model-type", choices=["lgb", "xgb", "blend"], default="blend")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rank-power", type=float, default=2.0)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=0.08)
    parser.add_argument("--corr-threshold", type=float, default=0.94)
    parser.add_argument("--half-life-weeks", type=float, default=52.0)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_fullweek_tree_portfolio.csv")
    parser.add_argument("--meta-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = [pd.Timestamp(d) for d in np.sort(prices["date"].unique())]
    windows = complete_week_windows(trading_dates)
    as_of = pd.Timestamp(args.as_of) if args.as_of else windows[-1].as_of

    sub, meta = generate_submission(
        prices,
        index_df,
        as_of,
        feature_set=args.feature_set,
        model_type=args.model_type,
        top_k=args.top_k,
        rank_power=args.rank_power,
        score_blend=args.score_blend,
        max_weight=args.max_weight,
        corr_threshold=args.corr_threshold,
        half_life_weeks=args.half_life_weeks,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    print(f">> model=stage2_fullweek_tree_portfolio as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
