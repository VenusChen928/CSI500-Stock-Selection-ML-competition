"""Regularized forest rank portfolio for stage2.

This is a deliberately different challenger from the LightGBM/XGBoost-heavy
route: selected rank-normalized features feed shallow bagged trees trained on
cross-sectional future-return ranks.  Portfolio shape is chosen only on
already-realized validation dates before `as_of`, then the model is refit on all
as-of-safe labels.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from baseline_xgboost import EMBARGO_DAYS, FORWARD_HORIZON, MIN_STOCKS
from features import build_features, prediction_frame, training_frame
from score_submission import score_window
from stage2_regularized_consensus import (
    add_rank_target,
    candidate_features,
    cross_sectional_normalize,
    select_features,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAX_WEIGHT = 0.10


@dataclass(frozen=True)
class Shape:
    top_k: int
    rank_power: float
    max_weight: float
    blend_extra: float


def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights[weights > 0].astype(float).copy()
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
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def split_train_shape(pool: pd.DataFrame, shape_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = np.sort(pool["date"].unique())
    needed = shape_days + EMBARGO_DAYS + 80
    if len(dates) < needed:
        raise RuntimeError(f"not enough dates for forest split: {len(dates)} < {needed}")
    shape_start = pd.Timestamp(dates[-shape_days])
    train_end = pd.Timestamp(dates[-(shape_days + EMBARGO_DAYS + 1)])
    train_df = pool[pool["date"] <= train_end].copy()
    shape_df = pool[pool["date"] >= shape_start].copy()
    return train_df, shape_df, train_end, shape_start


def make_models() -> dict[str, object]:
    return {
        "extra": ExtraTreesRegressor(
            n_estimators=96,
            max_depth=6,
            min_samples_leaf=120,
            max_features=0.65,
            bootstrap=False,
            random_state=1618,
            n_jobs=1,
        ),
        "rf": RandomForestRegressor(
            n_estimators=72,
            max_depth=5,
            min_samples_leaf=140,
            max_features=0.70,
            bootstrap=True,
            random_state=2718,
            n_jobs=1,
        ),
    }


def fit_models(train_df: pd.DataFrame, features: list[str]) -> dict[str, object]:
    models = make_models()
    y = train_df["target_rank_5d"].astype(float)
    for model in models.values():
        model.fit(train_df[features], y)
    return models


def predict_scores(models: dict[str, object], df: pd.DataFrame, features: list[str], blend_extra: float) -> pd.Series:
    extra = pd.Series(models["extra"].predict(df[features]), index=df["stock_code"].astype(str).str.zfill(6))
    rf = pd.Series(models["rf"].predict(df[features]), index=df["stock_code"].astype(str).str.zfill(6))
    # Blend daily ranks instead of raw values so tree scale differences cannot
    # dominate the portfolio.
    extra_rank = extra.rank(method="average", pct=True)
    rf_rank = rf.rank(method="average", pct=True)
    return blend_extra * extra_rank + (1.0 - blend_extra) * rf_rank


def build_portfolio(scores: pd.Series, shape: Shape) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(shape.top_k)
    ranks = pd.Series(np.arange(len(chosen), 0, -1, dtype=float) ** shape.rank_power, index=chosen.index)
    return cap_weights(ranks, shape.max_weight)


def validation_windows(shape_dates: np.ndarray, trading_dates: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    date_to_idx = {d: i for i, d in enumerate(dates)}
    unique_dates = [pd.Timestamp(d) for d in np.sort(shape_dates)]
    out = []
    for offset in range(0, len(unique_dates), FORWARD_HORIZON):
        as_of = unique_dates[offset]
        idx = date_to_idx.get(as_of)
        if idx is None or idx + FORWARD_HORIZON >= len(dates):
            continue
        out.append((as_of, dates[idx + 1], dates[idx + FORWARD_HORIZON]))
    return out


def shape_grid() -> list[Shape]:
    return [
        Shape(top_k, power, MAX_WEIGHT, blend_extra)
        for top_k in (30, 40)
        for power in (4.0, 8.0, 16.0)
        for blend_extra in (0.50, 0.70)
    ]


def select_shape(
    models: dict[str, object],
    panel: pd.DataFrame,
    shape_df: pd.DataFrame,
    features: list[str],
    trading_dates: np.ndarray,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
) -> tuple[Shape, pd.DataFrame]:
    rows = []
    windows = validation_windows(shape_df["date"].unique(), trading_dates)
    score_cache: dict[pd.Timestamp, pd.Series] = {}
    for as_of, _, _ in windows:
        pred = prediction_frame(panel, as_of=as_of).dropna(subset=features).copy()
        score_cache[as_of] = pred.set_index(pred["stock_code"].astype(str).str.zfill(6)).index.to_series()
        # Store a placeholder index first, then overwrite for each blend in the
        # grid because blend changes the rank combination.
    pred_cache = {
        as_of: prediction_frame(panel, as_of=as_of).dropna(subset=features).copy()
        for as_of, _, _ in windows
    }
    for shape in shape_grid():
        values = []
        for as_of, start, end in windows:
            pred = pred_cache[as_of]
            scores = predict_scores(models, pred, features, shape.blend_extra)
            weights = build_portfolio(scores, shape)
            values.append(score_window(weights, prices, index_df, start, end)["excess_return"])
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        neg = int((arr < 0).sum())
        rows.append(
            {
                **shape.__dict__,
                "mean_excess": float(arr.mean()),
                "min_excess": float(arr.min()),
                "std_excess": float(arr.std()),
                "negative_windows": neg,
                "count": len(arr),
                "utility": float(arr.mean() + 0.50 * arr.min() - 0.20 * arr.std() - 0.004 * neg),
            }
        )
    table = pd.DataFrame(rows).sort_values(["utility", "mean_excess", "min_excess"], ascending=False)
    if table.empty:
        return Shape(30, 4.0, MAX_WEIGHT, 0.65), table
    best = table.iloc[0]
    return Shape(int(best["top_k"]), float(best["rank_power"]), float(best["max_weight"]), float(best["blend_extra"])), table


def validation_ic(models: dict[str, object], val_df: pd.DataFrame, features: list[str], blend_extra: float) -> float:
    ics = []
    for _, g in val_df.groupby("date"):
        if len(g) < 50:
            continue
        scores = predict_scores(models, g, features, blend_extra)
        target = g.set_index(g["stock_code"].astype(str).str.zfill(6))["target_rank_5d"]
        rho, _ = spearmanr(target.reindex(scores.index), scores)
        if not np.isnan(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


def fit_forest_rank(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 420,
    shape_days: int = 30,
    max_features: int = 24,
    corr_threshold: float = 0.86,
) -> dict:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    if lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices_fit = prices[prices["date"] >= min_date].copy()
        index_fit = index_df[index_df["date"] >= min_date].copy()
    else:
        prices_fit = prices
        index_fit = index_df

    raw_panel = add_rank_target(build_features(prices_fit, index_fit))
    features_all = candidate_features(raw_panel)
    panel = cross_sectional_normalize(raw_panel, features_all)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    pool = training_frame(panel, max_date=train_cutoff).dropna(subset=["target_rank_5d"]).copy()
    train_df, shape_df, train_end, shape_start = split_train_shape(pool, shape_days)
    selected, feature_table = select_features(
        train_df,
        features_all,
        max_features=max_features,
        corr_threshold=corr_threshold,
    )
    holdout_models = fit_models(train_df, selected)
    shape, shape_table = select_shape(
        holdout_models,
        panel,
        shape_df,
        selected,
        trading_dates,
        prices_fit,
        index_fit,
    )
    ic = validation_ic(holdout_models, shape_df, selected, shape.blend_extra)
    final_models = fit_models(pool, selected)
    return {
        "panel": panel,
        "features": selected,
        "feature_table": feature_table,
        "shape": shape,
        "shape_table": shape_table,
        "models": final_models,
        "train_end": train_end,
        "shape_start": shape_start,
        "train_rows": len(train_df),
        "shape_rows": len(shape_df),
        "validation_ic": ic,
    }


def generate_submission(fit: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    pred = prediction_frame(fit["panel"], as_of=as_of).dropna(subset=fit["features"]).copy()
    scores = predict_scores(fit["models"], pred, fit["features"], fit["shape"].blend_extra)
    weights = build_portfolio(scores, fit["shape"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--lookback-days", type=int, default=420)
    parser.add_argument("--shape-days", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--corr-threshold", type=float, default=0.86)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_forest_rank_portfolio.csv")
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--feature-report-out", default=None)
    parser.add_argument("--shape-report-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = fit_forest_rank(
        prices,
        index_df,
        as_of,
        lookback_days=args.lookback_days,
        shape_days=args.shape_days,
        max_features=args.max_features,
        corr_threshold=args.corr_threshold,
    )
    sub = generate_submission(fit, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta = pd.DataFrame(
            [
                {
                    "as_of": as_of.date().isoformat(),
                    "final_route": "forest_rank_portfolio",
                    **fit["shape"].__dict__,
                    "feature_count": len(fit["features"]),
                    "validation_ic": fit["validation_ic"],
                    "train_end": fit["train_end"].date().isoformat(),
                    "shape_start": fit["shape_start"].date().isoformat(),
                    "train_rows": fit["train_rows"],
                    "shape_rows": fit["shape_rows"],
                    "n_names": len(sub),
                    "max_actual_weight": float(sub["weight"].max()),
                }
            ]
        )
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    if args.feature_report_out:
        p = Path(args.feature_report_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        fit["feature_table"].assign(selected=lambda d: d["feature"].isin(fit["features"])).to_csv(p, index=False)
    if args.shape_report_out:
        p = Path(args.shape_report_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        fit["shape_table"].to_csv(p, index=False)
    print(f">> model=stage2_forest_rank_portfolio as_of={as_of.date()}")
    print(f">> selected_features={len(fit['features'])}: {', '.join(fit['features'])}")
    print(f">> validation_ic={fit['validation_ic']:.4f} shape={fit['shape']}")
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
