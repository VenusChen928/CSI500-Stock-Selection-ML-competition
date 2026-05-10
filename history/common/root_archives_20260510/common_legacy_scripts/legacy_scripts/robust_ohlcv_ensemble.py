"""
Robust OHLCV-only ensemble for short-window CSI500 portfolio selection.

This experiment intentionally ignores optional open-data files.  It builds a
larger feature panel from the original competition OHLCV + index data, screens
features by recent rank-IC and pairwise correlation, then blends three robust
tabular learners:

  - LightGBM Huber loss.
  - XGBoost pseudo-Huber loss.
  - CatBoost quantile loss with ordered boosting.

The selected blend and portfolio shape are chosen by canonical
score_submission.py logic on recent validation windows.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from baseline_xgboost import MAX_WEIGHT, MIN_STOCKS
from score_submission import score_window

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_HORIZON = 3
DEFAULT_LOOKBACK_DAYS = 520
DEFAULT_VAL_DAYS = 18
DEFAULT_EMBARGO_DAYS = 3
DEFAULT_MAX_FEATURES = 48


@dataclass(frozen=True)
class Shape:
    top_k: int
    rank_power: float


@dataclass(frozen=True)
class Blend:
    lgbm: float
    xgb: float
    cat: float


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _per_stock_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    code = getattr(df, "name", None)
    df = df.sort_values("date").copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)
    turnover = df["turnover"].astype(float)

    for n in (1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 60, 120):
        df[f"ret_{n}d"] = close.pct_change(n)
        df[f"logret_{n}d"] = np.log(_safe_div(close, close.shift(n)))
    for n in (5, 10, 20, 60, 120):
        ma = close.rolling(n).mean()
        df[f"close_ma_gap_{n}d"] = close / ma - 1.0
        df[f"ma_slope_{n}d"] = ma / ma.shift(min(n, 10)) - 1.0
    for n in (3, 5, 10, 20, 60):
        ret = close.pct_change(1)
        df[f"vol_{n}d"] = ret.rolling(n).std()
        df[f"down_vol_{n}d"] = ret.clip(upper=0).rolling(n).std()
    df["vol_ratio_5_20"] = _safe_div(df["vol_5d"], df["vol_20d"])
    df["vol_ratio_10_60"] = _safe_div(df["vol_10d"], df["vol_60d"])

    for n in (5, 10, 20):
        vol_mean = volume.rolling(n).mean()
        vol_std = volume.rolling(n).std()
        amount_mean = amount.rolling(n).mean()
        amount_std = amount.rolling(n).std()
        turnover_mean = turnover.rolling(n).mean()
        turnover_std = turnover.rolling(n).std()
        df[f"volume_z_{n}d"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        df[f"amount_z_{n}d"] = (amount - amount_mean) / amount_std.replace(0, np.nan)
        df[f"turnover_z_{n}d"] = (turnover - turnover_mean) / turnover_std.replace(0, np.nan)
        df[f"turnover_ma_{n}d"] = turnover_mean

    df["intraday_ret"] = close / open_.replace(0, np.nan) - 1.0
    df["overnight_ret"] = open_ / close.shift(1).replace(0, np.nan) - 1.0
    df["high_low_range"] = high / low.replace(0, np.nan) - 1.0
    df["upper_shadow"] = high / np.maximum(open_, close).replace(0, np.nan) - 1.0
    df["lower_shadow"] = np.minimum(open_, close) / low.replace(0, np.nan) - 1.0
    df["close_to_high"] = close / high.replace(0, np.nan) - 1.0
    df["close_to_low"] = close / low.replace(0, np.nan) - 1.0

    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    df["close_pos_20d"] = (close - rolling_low_20) / (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    df["drawdown_20d"] = close / rolling_high_20.replace(0, np.nan) - 1.0
    df["amihud_20d"] = (close.pct_change(1).abs() / amount.replace(0, np.nan)).rolling(20).mean()

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df["mom_accel_3_10"] = df["ret_3d"] - df["ret_10d"]
    df["mom_accel_5_20"] = df["ret_5d"] - df["ret_20d"]
    df["reversal_3d"] = -df["ret_3d"]
    df["reversal_5d"] = -df["ret_5d"]

    df[f"target_{horizon}d"] = close.shift(-horizon) / close - 1.0
    if code is not None and "stock_code" not in df.columns:
        df["stock_code"] = code
    return df


def _index_features(index_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    idx = index_df.sort_values("date").copy()
    close = idx["close"].astype(float)
    for n in (1, 3, 5, 10, 20, 60):
        idx[f"idx_ret_{n}d"] = close.pct_change(n)
    idx["idx_vol_20d"] = close.pct_change(1).rolling(20).std()
    idx[f"idx_target_{horizon}d"] = close.shift(-horizon) / close - 1.0
    return idx[[
        "date",
        "idx_ret_1d",
        "idx_ret_3d",
        "idx_ret_5d",
        "idx_ret_10d",
        "idx_ret_20d",
        "idx_ret_60d",
        "idx_vol_20d",
        f"idx_target_{horizon}d",
    ]]


def build_panel(prices: pd.DataFrame, index_df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    try:
        panel = prices.groupby("stock_code", group_keys=False).apply(
            _per_stock_features,
            horizon=horizon,
            include_groups=False,
        ).reset_index(drop=True)
    except TypeError:
        panel = prices.groupby("stock_code", group_keys=False).apply(
            _per_stock_features,
            horizon=horizon,
        ).reset_index(drop=True)

    idx = _index_features(index_df.assign(date=pd.to_datetime(index_df["date"])), horizon)
    panel = panel.merge(idx, on="date", how="left")
    target = f"target_{horizon}d"
    idx_target = f"idx_target_{horizon}d"
    panel[f"target_excess_{horizon}d"] = panel[target] - panel[idx_target]
    for n in (1, 3, 5, 10, 20, 60):
        panel[f"excess_ret_{n}d"] = panel[f"ret_{n}d"] - panel[f"idx_ret_{n}d"]

    rank_bases = [
        "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "excess_ret_3d", "excess_ret_5d", "excess_ret_20d",
        "vol_5d", "vol_20d", "vol_ratio_5_20",
        "amount_z_20d", "turnover_z_20d", "close_pos_20d", "drawdown_20d",
        "rsi_14", "amihud_20d",
    ]
    for col in rank_bases:
        if col in panel.columns:
            panel[f"{col}_rank"] = panel.groupby("date")[col].rank(method="average", pct=True)
    for col in ("ret_3d", "ret_5d", "ret_20d", "vol_20d", "amount_z_20d"):
        panel[f"{col}_zcs"] = panel.groupby("date")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
    panel["breadth_ret_3d_pos"] = panel.groupby("date")["ret_3d"].transform(lambda s: (s > 0).mean())
    panel["breadth_ret_5d_pos"] = panel.groupby("date")["ret_5d"].transform(lambda s: (s > 0).mean())
    panel["dispersion_ret_3d"] = panel.groupby("date")["ret_3d"].transform("std")
    panel["dispersion_ret_5d"] = panel.groupby("date")["ret_5d"].transform("std")
    return panel.replace([np.inf, -np.inf], np.nan)


def candidate_features(panel: pd.DataFrame, horizon: int) -> list[str]:
    excluded = {"date", "stock_code", f"target_{horizon}d", f"idx_target_{horizon}d", f"target_excess_{horizon}d"}
    return [
        c for c in panel.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])
    ]


def rank_ic_table(df: pd.DataFrame, features: list[str], target_col: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        ics = []
        for _, g in df[["date", feature, target_col]].dropna().groupby("date"):
            if len(g) < 30 or g[feature].nunique() < 5:
                continue
            ic = g[feature].rank().corr(g[target_col].rank())
            if pd.notna(ic):
                ics.append(float(ic))
        if ics:
            rows.append({
                "feature": feature,
                "mean_rank_ic": float(np.mean(ics)),
                "abs_mean_rank_ic": float(abs(np.mean(ics))),
                "std_rank_ic": float(np.std(ics)),
                "n_days": len(ics),
            })
    return pd.DataFrame(rows).sort_values("abs_mean_rank_ic", ascending=False)


def select_features(
    train_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    max_features: int = DEFAULT_MAX_FEATURES,
    corr_threshold: float = 0.96,
    recent_days: int = 120,
) -> tuple[list[str], pd.DataFrame]:
    dates = np.sort(train_df["date"].unique())
    if recent_days and len(dates) > recent_days:
        screen_df = train_df[train_df["date"] >= pd.Timestamp(dates[-recent_days])].copy()
    else:
        screen_df = train_df
    ic = rank_ic_table(screen_df, features, target_col)
    selected: list[str] = []
    corr_cache = screen_df[features].corr(method="spearman").abs()
    for feature in ic["feature"]:
        if len(selected) >= max_features:
            break
        if not selected or corr_cache.loc[feature, selected].max() < corr_threshold:
            selected.append(feature)
    return selected, ic


def split_train_val(df: pd.DataFrame, val_days: int, embargo_days: int):
    dates = np.sort(df["date"].unique())
    val_start = pd.Timestamp(dates[-val_days])
    train_end = pd.Timestamp(dates[-(val_days + embargo_days + 1)])
    return df[df["date"] <= train_end].copy(), df[df["date"] >= val_start].copy(), train_end, val_start


def sample_weights(df: pd.DataFrame, target_col: str) -> np.ndarray:
    dates = np.sort(df["date"].unique())
    date_rank = pd.Series(np.arange(len(dates)), index=dates)
    recency = df["date"].map(date_rank).to_numpy(dtype=float)
    recency = recency / max(recency.max(), 1.0)
    tail = df[target_col].abs().rank(pct=True).to_numpy()
    return (0.75 + 0.50 * recency + 0.50 * tail).astype(float)


def _model_frame(df: pd.DataFrame, features: list[str], fill_values: pd.Series) -> pd.DataFrame:
    out = df[features].copy()
    return out.fillna(fill_values)


def train_models(
    train_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    fill_values: pd.Series,
    include_catboost: bool = True,
):
    X = _model_frame(train_df, features, fill_values)
    y = train_df[target_col].astype(float)
    w = sample_weights(train_df, target_col)
    models = {}
    models["lgbm"] = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        learning_rate=0.035,
        n_estimators=360,
        num_leaves=95,
        min_child_samples=35,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.03,
        reg_lambda=1.4,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ).fit(X, y, sample_weight=w)
    models["xgb"] = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=320,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.86,
        colsample_bytree=0.86,
        min_child_weight=8,
        reg_alpha=0.02,
        reg_lambda=1.8,
        tree_method="hist",
        random_state=43,
        n_jobs=-1,
    ).fit(X, y, sample_weight=w, verbose=False)
    if include_catboost:
        models["cat"] = cb.CatBoostRegressor(
            loss_function="Quantile:alpha=0.55",
            boosting_type="Plain",
            iterations=180,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=6.0,
            random_seed=44,
            verbose=False,
            thread_count=4,
            allow_writing_files=False,
        ).fit(X, y, sample_weight=w)
    return models


def predict_all(models: dict, df: pd.DataFrame, features: list[str], fill_values: pd.Series) -> pd.DataFrame:
    X = _model_frame(df, features, fill_values)
    out = pd.DataFrame({"date": df["date"].to_numpy(), "stock_code": df["stock_code"].astype(str).to_numpy()})
    for name, model in models.items():
        pred = pd.Series(model.predict(X), index=df.index)
        out[name] = pred.groupby(df["date"]).transform(lambda s: (s - s.mean()) / max(s.std(ddof=0), 1e-8)).to_numpy()
    if "cat" not in out.columns:
        out["cat"] = 0.0
    return out


def apply_cap(weights: pd.Series) -> pd.Series:
    w = weights[weights > 0].copy()
    if len(w) < MIN_STOCKS:
        raise ValueError("too few names")
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


def portfolio_from_scores(scores: pd.Series, shape: Shape) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(shape.top_k)
    ranks = np.arange(len(chosen), 0, -1, dtype=float)
    raw = ranks ** shape.rank_power
    return apply_cap(pd.Series(raw, index=chosen.index))


def validation_windows(val_dates: np.ndarray, trading_dates: np.ndarray, horizon: int):
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for d in val_dates[::horizon]:
        d = pd.Timestamp(d)
        idx = date_to_idx.get(d)
        if idx is None or idx + horizon >= len(trading_dates):
            continue
        windows.append((d, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + horizon])))
    return windows


def blend_grid(step: float = 0.25, include_catboost: bool = True) -> list[Blend]:
    if not include_catboost:
        return [Blend(float(w), float(1.0 - w), 0.0) for w in np.arange(0, 1 + 1e-9, step)]
    values = np.arange(0, 1 + 1e-9, step)
    blends = []
    for lgbm, xg in itertools.product(values, repeat=2):
        cat = 1.0 - lgbm - xg
        if cat >= -1e-9:
            blends.append(Blend(float(lgbm), float(xg), float(max(0.0, cat))))
    return blends


def select_blend_shape(
    val_pred: pd.DataFrame,
    val_dates: np.ndarray,
    trading_dates: np.ndarray,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    horizon: int,
    include_catboost: bool = True,
):
    windows = validation_windows(val_dates, trading_dates, horizon)
    rows = []
    pred_cache = {pd.Timestamp(d): g.set_index("stock_code") for d, g in val_pred.groupby("date")}
    for blend in blend_grid(0.25, include_catboost=include_catboost):
        for top_k in (35, 40, 45, 50, 55, 60, 70, 80, 100):
            for power in (0.45, 0.60, 0.75, 1.00, 1.25, 1.50):
                shape = Shape(top_k, power)
                scores = []
                for d, start, end in windows:
                    frame = pred_cache.get(d)
                    if frame is None:
                        continue
                    blended = blend.lgbm * frame["lgbm"] + blend.xgb * frame["xgb"] + blend.cat * frame["cat"]
                    weights = portfolio_from_scores(blended, shape)
                    scores.append(score_window(weights, prices, index_df, start, end)["excess_return"])
                if scores:
                    mean = float(np.mean(scores))
                    min_ = float(np.min(scores))
                    std = float(np.std(scores))
                    rows.append({
                        "lgbm_weight": blend.lgbm,
                        "xgb_weight": blend.xgb,
                        "cat_weight": blend.cat,
                        "top_k": top_k,
                        "rank_power": power,
                        "mean_excess_return": mean,
                        "sum_excess_return": float(np.sum(scores)),
                        "min_excess_return": min_,
                        "std_excess_return": std,
                        "utility_score": mean + 0.7 * min_ - 0.35 * std,
                        "n_windows": len(scores),
                    })
    table = pd.DataFrame(rows).sort_values(
        ["utility_score", "mean_excess_return", "min_excess_return"],
        ascending=False,
    )
    best = table.iloc[0]
    return (
        Blend(float(best["lgbm_weight"]), float(best["xgb_weight"]), float(best["cat_weight"])),
        Shape(int(best["top_k"]), float(best["rank_power"])),
        table,
    )


def fit_ensemble(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int = DEFAULT_HORIZON,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    max_features: int = DEFAULT_MAX_FEATURES,
    refit_full: bool = False,
    include_catboost: bool = True,
    feature_screen_days: int = 120,
):
    as_of = pd.Timestamp(as_of)
    if lookback_days:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()
    panel = build_panel(prices, index_df, horizon=horizon)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - horizon)])
    target_col = f"target_excess_{horizon}d"
    features = candidate_features(panel, horizon)
    train_pool = panel[panel["date"] <= train_cutoff].dropna(subset=[target_col]).copy()
    train_df, val_df, train_end, val_start = split_train_val(train_pool, val_days, embargo_days)
    selected_features, ic = select_features(
        train_df,
        features,
        target_col,
        max_features=max_features,
        recent_days=feature_screen_days,
    )
    fill_values = train_df[selected_features].median()
    val_models = train_models(
        train_df,
        selected_features,
        target_col,
        fill_values,
        include_catboost=include_catboost,
    )
    val_pred = predict_all(val_models, val_df, selected_features, fill_values)
    blend, shape, shape_table = select_blend_shape(
        val_pred,
        np.sort(val_df["date"].unique()),
        trading_dates,
        prices,
        index_df,
        horizon,
        include_catboost=include_catboost,
    )
    final_train = train_pool if refit_full else train_df
    final_fill = final_train[selected_features].median()
    models = train_models(
        final_train,
        selected_features,
        target_col,
        final_fill,
        include_catboost=include_catboost,
    )
    return {
        "panel": panel,
        "features": selected_features,
        "feature_ic": ic,
        "models": models,
        "fill_values": final_fill,
        "blend": blend,
        "shape": shape,
        "shape_table": shape_table,
        "train_df": train_df,
        "val_df": val_df,
        "train_pool": train_pool,
        "train_end": train_end,
        "val_start": val_start,
        "trading_dates": trading_dates,
    }


def generate_submission(fit: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    pred_df = fit["panel"][fit["panel"]["date"] == pd.Timestamp(as_of)].copy()
    pred = predict_all(fit["models"], pred_df, fit["features"], fit["fill_values"])
    frame = pred.set_index("stock_code")
    blend = fit["blend"]
    scores = blend.lgbm * frame["lgbm"] + blend.xgb * frame["xgb"] + blend.cat * frame["cat"]
    weights = portfolio_from_scores(scores, fit["shape"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date")
    parser.add_argument("--out", default="submissions/robust_ohlcv_ensemble.csv")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--feature-screen-days", type=int, default=120)
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="Refit base learners on train+validation pool after selecting blend/shape.",
    )
    parser.add_argument(
        "--no-catboost",
        action="store_true",
        help="Skip CatBoost for faster feature/shape iteration.",
    )
    parser.add_argument("--feature-report-out", default=None)
    parser.add_argument("--shape-report-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(prices["date"].max())

    fit = fit_ensemble(
        prices,
        index_df,
        as_of=as_of,
        horizon=args.horizon,
        lookback_days=args.lookback_days if args.lookback_days > 0 else None,
        val_days=args.val_days,
        max_features=args.max_features,
        refit_full=args.refit_full,
        include_catboost=not args.no_catboost,
        feature_screen_days=args.feature_screen_days,
    )
    submission = generate_submission(fit, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if args.feature_report_out:
        feature_path = Path(args.feature_report_out)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        fit["feature_ic"].to_csv(feature_path, index=False)
    if args.shape_report_out:
        shape_path = Path(args.shape_report_out)
        shape_path.parent.mkdir(parents=True, exist_ok=True)
        fit["shape_table"].to_csv(shape_path, index=False)

    print(f">> model=robust_ohlcv_ensemble as_of={as_of.date()} horizon={args.horizon}")
    print(f">> train rows={len(fit['train_df']):,} train_end={fit['train_end'].date()}")
    print(f">> val rows={len(fit['val_df']):,} val_start={fit['val_start'].date()}")
    print(f">> selected_features={len(fit['features'])}")
    print(">> top feature rank-IC")
    print(fit["feature_ic"].head(12).to_string(index=False))
    print(f">> selected blend={fit['blend']}")
    print(f">> selected shape={fit['shape']}")
    print(">> validation blend/shape table")
    print(fit["shape_table"].head(12).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
