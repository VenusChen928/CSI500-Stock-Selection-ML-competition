"""Regularized stage2 stock-selection ensemble.

The stage1 post-mortem pointed at a specific failure mode: recent-window
overfitting plus an overly concentrated portfolio.  This script deliberately
leans the other way:

* train several simple/regularized base learners on cross-sectional rank labels,
* separate model early-stopping validation from portfolio-shape validation,
* select features using only past training data with an IC + correlation filter,
* build a top-K consensus portfolio with model-agreement and risk penalties,
* cap single-name weights below the competition maximum by default.

The goal is not to make the cleverest model.  It is to make a model whose score
is harder to explain away as one lucky LSTM window.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from baseline_xgboost import EMBARGO_DAYS, MAX_WEIGHT, MIN_STOCKS
from features import (
    CORE_FEATURE_COLUMNS,
    EXPERIMENTAL_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
from score_submission import score_window

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DEFAULT_LOOKBACK_DAYS = 900
DEFAULT_MODEL_VAL_DAYS = 20
DEFAULT_SHAPE_DAYS = 55
DEFAULT_MAX_FEATURES = 26
DEFAULT_CORR_THRESHOLD = 0.90
DEFAULT_MAX_WEIGHT = 0.06


@dataclass(frozen=True)
class PortfolioShape:
    top_k: int
    rank_power: float
    agreement_power: float
    risk_penalty: float
    equal_mix: float
    max_weight: float


@dataclass(frozen=True)
class SplitDates:
    train_end: pd.Timestamp
    model_val_start: pd.Timestamp
    model_val_end: pd.Timestamp
    shape_start: pd.Timestamp
    train_cutoff: pd.Timestamp


def _safe_rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) < 20 or y_true.nunique(dropna=True) < 3 or y_pred.nunique(dropna=True) < 3:
        return np.nan
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else np.nan


def add_rank_target(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["target_rank_5d"] = panel.groupby("date")[TARGET_COLUMN].rank(method="average", pct=True) - 0.5
    return panel


def cross_sectional_normalize(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """As-of-safe daily rank normalization for features.

    Rank normalization is intentionally simple and fast.  It also matches the
    task: we care more about the cross-sectional order of stocks on each as-of
    date than about raw feature scale.
    """
    out = panel.copy()
    out[features] = out.groupby("date")[features].rank(method="average", pct=True) - 0.5
    out[features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def split_train_model_shape(train_pool: pd.DataFrame, model_val_days: int, shape_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitDates]:
    all_dates = np.sort(train_pool["date"].unique())
    needed = shape_days + model_val_days + 2 * EMBARGO_DAYS + 40
    if len(all_dates) < needed:
        raise RuntimeError(f"Not enough dates for regularized split: have {len(all_dates)}, need {needed}")

    shape_start_idx = len(all_dates) - shape_days
    model_val_end_idx = shape_start_idx - EMBARGO_DAYS - 1
    model_val_start_idx = model_val_end_idx - model_val_days + 1
    train_end_idx = model_val_start_idx - EMBARGO_DAYS - 1
    if train_end_idx < 30:
        raise RuntimeError("Training split is too short")

    train_end = pd.Timestamp(all_dates[train_end_idx])
    model_val_start = pd.Timestamp(all_dates[model_val_start_idx])
    model_val_end = pd.Timestamp(all_dates[model_val_end_idx])
    shape_start = pd.Timestamp(all_dates[shape_start_idx])
    train_cutoff = pd.Timestamp(all_dates[-1])

    train_df = train_pool[train_pool["date"] <= train_end].copy()
    model_val_df = train_pool[(train_pool["date"] >= model_val_start) & (train_pool["date"] <= model_val_end)].copy()
    shape_df = train_pool[train_pool["date"] >= shape_start].copy()
    return train_df, model_val_df, shape_df, SplitDates(train_end, model_val_start, model_val_end, shape_start, train_cutoff)


def candidate_features(panel: pd.DataFrame) -> list[str]:
    return [c for c in EXPERIMENTAL_FEATURE_COLUMNS if c in panel.columns]


def feature_ic_table(train_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    grouped = list(train_df.groupby("date"))
    for feature in features:
        ics = []
        for _, g in grouped:
            ic = _safe_rank_ic(g["target_rank_5d"], g[feature])
            if not np.isnan(ic):
                ics.append(ic)
        if not ics:
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        rows.append(
            {
                "feature": feature,
                "mean_ic": mean_ic,
                "abs_mean_ic": abs(mean_ic),
                "std_ic": std_ic,
                "stability_score": abs(mean_ic) / (std_ic + 1e-6),
                "n_days": len(ics),
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_mean_ic", "stability_score"], ascending=False)


def select_features(
    train_df: pd.DataFrame,
    features: list[str],
    max_features: int = DEFAULT_MAX_FEATURES,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    ic_table = feature_ic_table(train_df, features)
    if ic_table.empty:
        return CORE_FEATURE_COLUMNS.copy(), ic_table

    ordered = ic_table["feature"].tolist()
    keep: list[str] = []
    corr = train_df[ordered].corr().abs()
    for feature in ordered:
        if feature in keep:
            continue
        if any(corr.loc[feature, kept] > corr_threshold for kept in keep):
            continue
        keep.append(feature)
        if len(keep) >= max_features:
            break

    for feature in CORE_FEATURE_COLUMNS:
        if feature in features and feature not in keep:
            if not any(corr.loc[feature, kept] > corr_threshold for kept in keep if feature in corr.index and kept in corr.columns):
                keep.append(feature)
        if len(keep) >= max_features:
            break
    return keep, ic_table


def _lgb_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber",
        learning_rate=0.025,
        n_estimators=360,
        num_leaves=15,
        max_depth=4,
        min_child_samples=180,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )


def _xgb_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=320,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=80,
        reg_alpha=0.35,
        reg_lambda=12.0,
        tree_method="hist",
        random_state=43,
        n_jobs=1,
        early_stopping_rounds=35,
    )


def _cat_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=260,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=15.0,
        random_seed=44,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )


def fit_base_models(
    train_df: pd.DataFrame,
    model_val_df: pd.DataFrame,
    features: list[str],
    base_models: Iterable[str],
    *,
    final_refit_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    y_train = train_df["target_rank_5d"].astype(float)
    y_val = model_val_df["target_rank_5d"].astype(float)
    models: dict[str, object] = {}
    for name in base_models:
        if name == "ridge":
            model = Ridge(alpha=35.0, random_state=45)
            fit_df = final_refit_df if final_refit_df is not None else train_df
            model.fit(fit_df[features], fit_df["target_rank_5d"].astype(float))
        elif name == "lightgbm":
            model = _lgb_model()
            if final_refit_df is None:
                model.fit(
                    train_df[features],
                    y_train,
                    eval_set=[(model_val_df[features], y_val)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(35, verbose=False)],
                )
            else:
                model.fit(final_refit_df[features], final_refit_df["target_rank_5d"].astype(float))
        elif name == "xgboost":
            model = _xgb_model()
            if final_refit_df is None:
                model.fit(
                    train_df[features],
                    y_train,
                    eval_set=[(model_val_df[features], y_val)],
                    verbose=False,
                )
            else:
                model.set_params(early_stopping_rounds=None)
                model.fit(final_refit_df[features], final_refit_df["target_rank_5d"].astype(float), verbose=False)
        elif name == "catboost":
            model = _cat_model()
            fit_df = final_refit_df if final_refit_df is not None else train_df
            if final_refit_df is None:
                model.fit(train_df[features], y_train, eval_set=(model_val_df[features], y_val), use_best_model=True)
            else:
                model.fit(fit_df[features], fit_df["target_rank_5d"].astype(float))
        else:
            raise ValueError(f"Unknown base model {name}")
        models[name] = model
    return models


def predict_model_scores(models: dict[str, object], df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[["date", "stock_code"]].copy()
    for name, model in models.items():
        out[name] = np.asarray(model.predict(df[features]), dtype=float)
    return out


def daily_model_ranks(pred: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    out = pred[["date", "stock_code"]].copy()
    for name in model_names:
        out[f"{name}_rank"] = pred.groupby("date")[name].rank(method="average", pct=True)
    return out


def validation_model_weights(pred: pd.DataFrame, target_df: pd.DataFrame, model_names: list[str]) -> pd.Series:
    merged = pred.merge(target_df[["date", "stock_code", "target_rank_5d"]], on=["date", "stock_code"], how="inner")
    weights = {}
    for name in model_names:
        ics = []
        for _, g in merged.groupby("date"):
            ic = _safe_rank_ic(g["target_rank_5d"], g[name])
            if not np.isnan(ic):
                ics.append(ic)
        mean_ic = float(np.mean(ics)) if ics else 0.0
        weights[name] = max(0.0, mean_ic)
    s = pd.Series(weights, dtype=float)
    if s.sum() <= 1e-9:
        s[:] = 1.0 / len(s)
    else:
        s = s / s.sum()
    return s


def combine_scores(pred: pd.DataFrame, model_weights: pd.Series) -> pd.DataFrame:
    model_names = list(model_weights.index)
    ranks = daily_model_ranks(pred, model_names)
    rank_cols = [f"{m}_rank" for m in model_names]
    score = np.zeros(len(ranks), dtype=float)
    for name in model_names:
        score += model_weights[name] * ranks[f"{name}_rank"].to_numpy()
    ranks["score"] = score
    ranks["agreement"] = (1.0 - ranks[rank_cols].std(axis=1).fillna(0.0) * np.sqrt(12.0)).clip(0.20, 1.0)
    return ranks[["date", "stock_code", "score", "agreement"]]


def _cap_and_mix(raw: pd.Series, max_weight: float, equal_mix: float) -> pd.Series:
    raw = raw.clip(lower=0)
    if raw.sum() <= 0:
        raw[:] = 1.0
    w = raw / raw.sum()
    if equal_mix > 0:
        w = (1.0 - equal_mix) * w + equal_mix * pd.Series(1.0 / len(w), index=w.index)
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


def build_consensus_portfolio(score_today: pd.DataFrame, panel_today: pd.DataFrame, shape: PortfolioShape) -> pd.Series:
    merged = score_today.merge(
        panel_today[["stock_code", "vol_20d_rank", "turnover_ma_20d", "ret_20d_rank", "drawdown_20d"]],
        on="stock_code",
        how="left",
    )
    merged["turnover_rank"] = merged["turnover_ma_20d"].rank(method="average", pct=True).fillna(0.5)
    # Penalize the exact stage1 failure mode: very high volatility + very high turnover.
    merged["risk_rank"] = 0.55 * merged["vol_20d_rank"].fillna(0.5) + 0.35 * merged["turnover_rank"] + 0.10 * (-merged["drawdown_20d"].fillna(0.0)).rank(pct=True)
    chosen = merged.sort_values("score", ascending=False).head(shape.top_k).copy()
    if len(chosen) < MIN_STOCKS:
        raise ValueError("too few selected names")
    raw = (
        chosen["score"].clip(0.01, 1.0).to_numpy() ** shape.rank_power
        * chosen["agreement"].clip(0.2, 1.0).to_numpy() ** shape.agreement_power
        * np.exp(-shape.risk_penalty * chosen["risk_rank"].clip(0, 1).to_numpy())
    )
    weights = _cap_and_mix(pd.Series(raw, index=chosen["stock_code"].astype(str).str.zfill(6)), shape.max_weight, shape.equal_mix)
    return weights


def validation_windows(shape_dates: np.ndarray, trading_dates: np.ndarray, horizon: int = FORWARD_HORIZON) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    unique_dates = [pd.Timestamp(d) for d in np.sort(shape_dates)]
    out = []
    for offset in range(0, len(unique_dates), horizon):
        as_of = unique_dates[offset]
        idx = date_to_idx.get(as_of)
        if idx is None or idx + horizon >= len(trading_dates):
            continue
        out.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + horizon])))
    return out


def precompute_window_returns(
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
) -> dict[pd.Timestamp, tuple[pd.Series, float]]:
    prices = prices.copy()
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    close = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    idx = index_df.sort_values("date").set_index("date")
    cache: dict[pd.Timestamp, tuple[pd.Series, float]] = {}
    for as_of, start, end in windows:
        prior_dates = close.index[close.index < start]
        if len(prior_dates) == 0:
            continue
        entry_date = pd.Timestamp(prior_dates[-1])
        window_close = close[(close.index >= start) & (close.index <= end)]
        if window_close.empty:
            continue
        entry = close.loc[entry_date]
        exit_ = window_close.ffill().iloc[-1]
        stock_ret = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        idx_window = idx[(idx.index >= start) & (idx.index <= end)]
        idx_prior = idx[idx.index < start]
        if idx_window.empty or idx_prior.empty:
            continue
        bench = float(idx_window["close"].iloc[-1] / idx_prior["close"].iloc[-1] - 1.0)
        cache[as_of] = (stock_ret, bench)
    return cache


def shape_grid(max_weight: float) -> list[PortfolioShape]:
    return [
        PortfolioShape(top_k, rank_power, agreement_power, risk_penalty, equal_mix, max_weight)
        for top_k in (30, 35, 40, 50, 60, 80, 100)
        for rank_power in (0.6, 0.9, 1.2, 1.6)
        for agreement_power in (0.0, 1.0)
        for risk_penalty in (0.0, 0.35, 0.70)
        for equal_mix in (0.0, 0.10, 0.25)
    ]


def select_shape(
    combined_scores: pd.DataFrame,
    panel: pd.DataFrame,
    shape_df: pd.DataFrame,
    trading_dates: np.ndarray,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    max_weight: float,
) -> tuple[PortfolioShape, pd.DataFrame]:
    windows = validation_windows(shape_df["date"].unique(), trading_dates)
    return_cache = precompute_window_returns(windows, prices, index_df)
    rows = []
    for shape in shape_grid(max_weight):
        scores = []
        for as_of, start, end in windows:
            if as_of not in return_cache:
                continue
            score_today = combined_scores[combined_scores["date"] == as_of]
            panel_today = panel[panel["date"] == as_of]
            if score_today.empty or panel_today.empty:
                continue
            weights = build_consensus_portfolio(score_today, panel_today, shape)
            stock_ret, bench_ret = return_cache[as_of]
            portfolio_ret = float((weights * stock_ret.reindex(weights.index).fillna(0.0)).sum())
            scores.append(portfolio_ret - bench_ret)
        if not scores:
            continue
        arr = np.asarray(scores)
        rows.append(
            {
                **shape.__dict__,
                "mean_excess": float(arr.mean()),
                "min_excess": float(arr.min()),
                "std_excess": float(arr.std()),
                "negative_windows": int((arr < 0).sum()),
                "count": len(arr),
                "utility": float(arr.mean() + 0.60 * arr.min() - 0.25 * arr.std() - 0.003 * (arr < 0).sum()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["utility", "mean_excess", "min_excess"], ascending=False)
    if table.empty:
        return PortfolioShape(60, 0.9, 1.0, 0.35, 0.20, max_weight), table
    best = table.iloc[0]
    return PortfolioShape(
        int(best["top_k"]),
        float(best["rank_power"]),
        float(best["agreement_power"]),
        float(best["risk_penalty"]),
        float(best["equal_mix"]),
        float(best["max_weight"]),
    ), table


def fit_regularized_consensus(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    model_val_days: int = DEFAULT_MODEL_VAL_DAYS,
    shape_days: int = DEFAULT_SHAPE_DAYS,
    max_features: int = DEFAULT_MAX_FEATURES,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    base_models: Iterable[str] = ("ridge", "lightgbm", "xgboost", "catboost"),
) -> dict:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    if lookback_days and lookback_days > 0:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices_fit = prices[prices["date"] >= min_date].copy()
        index_fit = index_df[index_df["date"] >= min_date].copy()
    else:
        prices_fit = prices.copy()
        index_fit = index_df.copy()

    raw_panel = build_features(prices_fit, index_fit)
    raw_panel = add_rank_target(raw_panel)
    features = candidate_features(raw_panel)
    print(f">> building rank-normalized panel with {len(features)} candidate features", flush=True)
    panel = cross_sectional_normalize(raw_panel, features)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    pool = training_frame(panel, max_date=train_cutoff).dropna(subset=["target_rank_5d"]).copy()
    train_df, model_val_df, shape_df, split_dates = split_train_model_shape(pool, model_val_days, shape_days)

    selected_features, feature_table = select_features(train_df, features, max_features=max_features, corr_threshold=corr_threshold)
    print(f">> selected {len(selected_features)} features; fitting holdout base models", flush=True)
    base_models = tuple(base_models)
    holdout_models = fit_base_models(train_df, model_val_df, selected_features, base_models)

    val_pred = predict_model_scores(holdout_models, model_val_df, selected_features)
    model_weights = validation_model_weights(val_pred, model_val_df, list(base_models))
    shape_pred = predict_model_scores(holdout_models, shape_df, selected_features)
    combined_shape = combine_scores(shape_pred, model_weights)
    print(f">> selecting portfolio shape over {shape_df['date'].nunique()} shape dates", flush=True)
    shape, shape_table = select_shape(combined_shape, panel, shape_df, trading_dates, prices_fit, index_fit, max_weight=max_weight)

    print(">> refitting final base models on all as-of-safe labels", flush=True)
    final_models = fit_base_models(
        train_df,
        model_val_df,
        selected_features,
        base_models,
        final_refit_df=pool,
    )
    return {
        "panel": panel,
        "prices": prices_fit,
        "index_df": index_fit,
        "trading_dates": trading_dates,
        "features": selected_features,
        "feature_table": feature_table,
        "model_weights": model_weights,
        "shape": shape,
        "shape_table": shape_table,
        "models": final_models,
        "split_dates": split_dates,
        "train_rows": len(train_df),
        "model_val_rows": len(model_val_df),
        "shape_rows": len(shape_df),
    }


def generate_submission(fit: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    panel = fit["panel"]
    pred_df = prediction_frame(panel, as_of=as_of).copy()
    raw_pred = predict_model_scores(fit["models"], pred_df, fit["features"])
    combined = combine_scores(raw_pred, fit["model_weights"])
    weights = build_consensus_portfolio(combined[combined["date"] == pd.Timestamp(as_of)], pred_df, fit["shape"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/route_outputs/stage2_regularized_consensus.csv")
    parser.add_argument("--feature-report-out", default=None)
    parser.add_argument("--shape-report-out", default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--model-val-days", type=int, default=DEFAULT_MODEL_VAL_DAYS)
    parser.add_argument("--shape-days", type=int, default=DEFAULT_SHAPE_DAYS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--base-models", nargs="+", default=["ridge", "lightgbm", "xgboost", "catboost"])
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = fit_regularized_consensus(
        prices,
        index_df,
        as_of=as_of,
        lookback_days=args.lookback_days,
        model_val_days=args.model_val_days,
        shape_days=args.shape_days,
        max_features=args.max_features,
        corr_threshold=args.corr_threshold,
        max_weight=args.max_weight,
        base_models=args.base_models,
    )
    sub = generate_submission(fit, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    if args.feature_report_out:
        feature_path = Path(args.feature_report_out)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        fit["feature_table"].assign(selected=lambda d: d["feature"].isin(fit["features"])).to_csv(feature_path, index=False)
    if args.shape_report_out:
        shape_path = Path(args.shape_report_out)
        shape_path.parent.mkdir(parents=True, exist_ok=True)
        fit["shape_table"].to_csv(shape_path, index=False)

    split = fit["split_dates"]
    print(f">> model=stage2_regularized_consensus as_of={as_of.date()}")
    print(f">> split train_end={split.train_end.date()} model_val={split.model_val_start.date()}..{split.model_val_end.date()} shape_start={split.shape_start.date()} cutoff={split.train_cutoff.date()}")
    print(f">> rows train={fit['train_rows']:,} model_val={fit['model_val_rows']:,} shape={fit['shape_rows']:,}")
    print(f">> selected_features={len(fit['features'])}: {', '.join(fit['features'])}")
    print(f">> model_weights={fit['model_weights'].round(3).to_dict()}")
    print(f">> selected_shape={fit['shape']}")
    print(">> shape table top")
    print(fit["shape_table"].head(10).to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
