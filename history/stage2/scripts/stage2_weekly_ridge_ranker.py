"""As-of-safe weekly ridge ranker for Stage2.

This route is deliberately simpler than the tree/LSTM stack.  It trains a
regularized linear rank model on data available at ``as_of`` only, using all
known 5-trading-day targets but up-weighting complete Monday-Friday evaluation
windows.  The goal is a low-leakage, low-variance challenger that learns the
weekly-cycle feature direction without memorizing a small number of windows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from features import (
    ALPHA_FEATURE_COLUMNS,
    CALENDAR_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    QUALITY_FEATURE_COLUMNS,
    TARGET_EXCESS_COLUMN,
    WEEKLY_CYCLE_FEATURE_COLUMNS,
    build_features,
)
from stage2_weekly_cycle_tree import calendar_alpha

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAX_WEIGHT = 0.095


def feature_pool(panel: pd.DataFrame, feature_set: str) -> list[str]:
    groups = {
        "weekly": WEEKLY_CYCLE_FEATURE_COLUMNS,
        "momentum_quality": list(dict.fromkeys(MOMENTUM_FEATURE_COLUMNS + QUALITY_FEATURE_COLUMNS)),
        "alpha_quality": list(dict.fromkeys(ALPHA_FEATURE_COLUMNS + QUALITY_FEATURE_COLUMNS + CALENDAR_FEATURE_COLUMNS)),
        "compact": [
            "ret_1d_rank",
            "ret_3d_rank",
            "ret_5d_rank",
            "ret_10d_rank",
            "ret_20d_rank",
            "ret_60d_rank",
            "excess_ret_5d_rank",
            "excess_ret_20d_rank",
            "mom_accel_5_20_rank",
            "trend_quality_20d_rank",
            "trend_efficiency_20d_rank",
            "vol_ratio_5_20_rank",
            "downside_vol_20d_rank",
            "amount_z_20d_rank",
            "obv_20d_rank",
            "gap_mean_20d_rank",
            "intraday_mean_5d_rank",
            "close_location_ma20_rank",
            "weekly_risk_appetite_rank",
            "weekly_friday_derisk_rank",
            "month_start_flow_rank",
            "month_end_defensive_rank",
            "post_gap_reopen_flow_rank",
            "weekly_carry_quality_rank",
            "eval_is_full_workweek",
            "eval_month_start",
            "eval_month_end",
            "weekend_gap_days",
        ],
    }
    if feature_set not in groups:
        raise ValueError(f"unknown feature_set={feature_set}")
    return [c for c in dict.fromkeys(groups[feature_set]) if c in panel.columns]


def trading_windows(trading_dates: np.ndarray) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in np.sort(trading_dates)]
    out: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for idx in range(len(dates) - FORWARD_HORIZON):
        eval_dates = dates[idx + 1 : idx + FORWARD_HORIZON + 1]
        out[dates[idx]] = (eval_dates[0], eval_dates[-1])
    return out


def complete_workweek_flags(trading_dates: np.ndarray) -> dict[pd.Timestamp, float]:
    flags: dict[pd.Timestamp, float] = {}
    for as_of, (start, end) in trading_windows(trading_dates).items():
        flags[as_of] = float(start.weekday() == 0 and end.weekday() == 4 and (end - start).days == 4)
    return flags


def rank_target_by_date(frame: pd.DataFrame) -> np.ndarray:
    ranks = frame.groupby("date")[TARGET_EXCESS_COLUMN].rank(method="average", pct=True)
    return ranks.to_numpy(dtype=float) - 0.5


def normalize_features(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    data = frame[features].replace([np.inf, -np.inf], np.nan)
    good = [col for col in features if data[col].notna().mean() >= 0.80 and data[col].nunique(dropna=True) > 2]
    if not good:
        raise RuntimeError("no usable ridge features")
    x = data[good].fillna(data[good].median(numeric_only=True)).to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    return x, good


def standardize_train_pred(train_x: np.ndarray, pred_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    return (train_x - mu) / sigma, (pred_x - mu) / sigma, mu, sigma


def correlation_filter_np(x: np.ndarray, features: list[str], threshold: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    if threshold <= 0 or threshold >= 1 or x.shape[1] <= 1:
        keep = np.arange(x.shape[1])
        return x, features, keep
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    keep_idx: list[int] = []
    dropped = np.zeros(x.shape[1], dtype=bool)
    for idx in range(x.shape[1]):
        if dropped[idx]:
            continue
        keep_idx.append(idx)
        dropped |= corr[idx] > threshold
        dropped[keep_idx] = False
    keep = np.asarray(keep_idx, dtype=int)
    return x[:, keep], [features[i] for i in keep], keep


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    pred_x: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_x, pred_x, _, _ = standardize_train_pred(train_x, pred_x)
    train_x = np.column_stack([np.ones(train_x.shape[0]), train_x])
    pred_x = np.column_stack([np.ones(pred_x.shape[0]), pred_x])
    sw = np.sqrt(np.asarray(sample_weight, dtype=float).clip(1e-6))
    xw = train_x * sw[:, None]
    yw = train_y * sw
    penalty = np.eye(train_x.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(xw.T @ xw + penalty) @ (xw.T @ yw)
    return pred_x @ coef, coef


def cap_weights(raw: pd.Series, max_weight: float) -> pd.Series:
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


def portfolio_from_score(
    score: pd.Series,
    *,
    top_k: int,
    rank_power: float,
    max_weight: float,
    score_mix: float,
) -> pd.Series:
    selected = score.sort_values(ascending=False).head(max(MIN_STOCKS, int(top_k)))
    ranks = pd.Series(np.arange(len(selected), 0, -1, dtype=float), index=selected.index)
    rank_raw = ranks.pow(rank_power)
    shifted = selected - float(selected.min()) + 1e-6
    score_raw = shifted.clip(lower=1e-6)
    raw = (1.0 - score_mix) * rank_raw / rank_raw.sum() + score_mix * score_raw / score_raw.sum()
    return cap_weights(raw, max_weight=max_weight)


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    feature_set: str = "compact",
    alpha: float = 25.0,
    corr_threshold: float = 0.92,
    half_life_days: float = 180.0,
    fullweek_weight: float = 2.5,
    nonfull_weight: float = 0.6,
    calendar_blend: float = 0.25,
    risk_penalty: float = 0.10,
    top_k: int = 35,
    rank_power: float = 4.0,
    max_weight: float = MAX_WEIGHT,
    score_mix: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(prices, index_df)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    trading_dates = np.sort(panel["date"].dropna().unique())
    fullweek_flags = complete_workweek_flags(trading_dates)

    features = feature_pool(panel, feature_set)
    train = panel[(panel["date"] < as_of) & panel[TARGET_EXCESS_COLUMN].notna()].copy()
    pred = panel[panel["date"] == as_of].copy()
    if train.empty or pred.empty:
        raise RuntimeError(f"empty train/pred frame for {as_of.date()}")

    train_x, features = normalize_features(train, features)
    pred_x = pred[features].replace([np.inf, -np.inf], np.nan)
    pred_x = pred_x.fillna(train[features].median(numeric_only=True)).to_numpy(dtype=float)
    pred_x = np.where(np.isfinite(pred_x), pred_x, np.nanmedian(train_x, axis=0))
    train_x, features, keep = correlation_filter_np(train_x, features, corr_threshold)
    pred_x = pred_x[:, keep]

    train_y = rank_target_by_date(train)
    age_days = (as_of - pd.to_datetime(train["date"])).dt.days.to_numpy(dtype=float)
    weights = np.power(0.5, age_days / max(float(half_life_days), 1.0))
    is_full = train["date"].map(fullweek_flags).fillna(0.0).to_numpy(dtype=float)
    weights *= np.where(is_full > 0.5, fullweek_weight, nonfull_weight)
    if "eval_month_start" in train:
        weights *= 1.0 + 0.08 * train["eval_month_start"].to_numpy(dtype=float).clip(0, 1)
    if "eval_month_end" in train:
        weights *= 1.0 + 0.08 * train["eval_month_end"].to_numpy(dtype=float).clip(0, 1)

    pred_score, coef = ridge_fit_predict(train_x, train_y, pred_x, weights, alpha=alpha)
    codes = pred["stock_code"].astype(str).str.zfill(6)
    score = pd.Series(pred_score, index=codes)
    model_rank = score.rank(method="average", pct=True)

    today = pred.copy()
    cal = calendar_alpha(today).reindex(codes).fillna(0.5)
    final = (1.0 - calendar_blend) * model_rank + calendar_blend * cal
    if risk_penalty:
        risk_cols = [c for c in ["vol_20d_rank", "downside_vol_20d_rank", "overnight_vol_20d_rank"] if c in today.columns]
        if risk_cols:
            risk = today.set_index(codes)[risk_cols].astype(float).mean(axis=1).rank(method="average", pct=True)
            final = final - risk_penalty * risk.reindex(final.index).fillna(0.5)
    weights_out = portfolio_from_score(
        final,
        top_k=top_k,
        rank_power=rank_power,
        max_weight=max_weight,
        score_mix=score_mix,
    )
    sub = pd.DataFrame({"stock_code": weights_out.index, "weight": weights_out.values})
    diagnostics = pd.DataFrame(
        {
            "stock_code": final.sort_values(ascending=False).head(80).index,
            "final_confidence": final.sort_values(ascending=False).head(80).values,
            "ridge_rank": model_rank.reindex(final.sort_values(ascending=False).head(80).index).values,
            "calendar_alpha": cal.reindex(final.sort_values(ascending=False).head(80).index).values,
        }
    )
    top_coef_idx = np.argsort(np.abs(coef[1:]))[::-1][:20]
    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "model": "stage2_weekly_ridge_ranker",
                "feature_set": feature_set,
                "features": len(features),
                "train_rows": len(train),
                "train_dates": train["date"].nunique(),
                "alpha": alpha,
                "corr_threshold": corr_threshold,
                "half_life_days": half_life_days,
                "fullweek_weight": fullweek_weight,
                "nonfull_weight": nonfull_weight,
                "calendar_blend": calendar_blend,
                "risk_penalty": risk_penalty,
                "top_k": top_k,
                "rank_power": rank_power,
                "max_weight": max_weight,
                "score_mix": score_mix,
                "n_names": len(sub),
                "max_observed_weight": float(sub["weight"].max()),
                "effective_n": float(1.0 / np.square(sub["weight"].to_numpy()).sum()),
                "top_coefficients": " | ".join(f"{features[i]}:{coef[i + 1]:.4g}" for i in top_coef_idx),
            }
        ]
    )
    return sub, meta, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--feature-set", choices=["compact", "weekly", "momentum_quality", "alpha_quality"], default="compact")
    parser.add_argument("--alpha", type=float, default=25.0)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--fullweek-weight", type=float, default=2.5)
    parser.add_argument("--nonfull-weight", type=float, default=0.6)
    parser.add_argument("--calendar-blend", type=float, default=0.25)
    parser.add_argument("--risk-penalty", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--rank-power", type=float, default=4.0)
    parser.add_argument("--max-weight", type=float, default=MAX_WEIGHT)
    parser.add_argument("--score-mix", type=float, default=0.10)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_weekly_ridge_ranker.csv")
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

    sub, meta, diagnostics = generate_submission(
        prices,
        index_df,
        as_of,
        feature_set=args.feature_set,
        alpha=args.alpha,
        corr_threshold=args.corr_threshold,
        half_life_days=args.half_life_days,
        fullweek_weight=args.fullweek_weight,
        nonfull_weight=args.nonfull_weight,
        calendar_blend=args.calendar_blend,
        risk_penalty=args.risk_penalty,
        top_k=args.top_k,
        rank_power=args.rank_power,
        max_weight=args.max_weight,
        score_mix=args.score_mix,
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
    print(f">> model=stage2_weekly_ridge_ranker as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
