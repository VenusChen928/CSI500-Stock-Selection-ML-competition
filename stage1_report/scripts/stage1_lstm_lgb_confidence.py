"""
Stage1 LSTM + LightGBM confidence portfolio.

This 3-trading-day experiment uses only the original competition OHLCV/index
data.  LSTM captures short sequence signal, LightGBM confirms tabular signal,
and a hand-auditable confidence layer controls final weights:

  score      = alpha * LSTM + (1-alpha) * LightGBM + long-trend tilt
  confidence = model agreement + trend strength - correctly oriented risk

The default risk-balanced policy is tuned for the 3-trading-day stage1 window.
Validation mode remains available for research sweeps.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = REPORT_DIR.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "stage2_report" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lstm_rank_weight as lstm
from baseline_xgboost import MAX_WEIGHT, MIN_STOCKS
from features import EXPERIMENTAL_FEATURE_COLUMNS, build_features

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_HORIZON = 3
DEFAULT_LOOKBACK_DAYS = 520
DEFAULT_VAL_DAYS = 18
DEFAULT_EMBARGO_DAYS = 3

LONG_FEATURES = [
    "ret_20d",
    "ret_60d",
    "close_over_ma20",
    "close_over_ma60",
    "excess_ret_20d",
    "ret_20d_rank",
    "excess_ret_20d_rank",
]
RISK_FEATURES = [
    "vol_5d",
    "vol_20d",
    "vol_ratio_5_20",
    "high_low_range",
    "drawdown_20d",
    "amount_z_20d",
    "turnover_z_20d",
]
RISK_ABS_FEATURES = {"amount_z_20d", "turnover_z_20d"}
RISK_NEGATIVE_FEATURES = {"drawdown_20d"}


def _rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True).fillna(0.5)


@dataclass(frozen=True)
class HybridPolicy:
    alpha_lstm: float
    long_weight: float
    risk_penalty: float
    top_k: int
    rank_power: float
    confidence_power: float


AGGRESSIVE_LSTM_POLICY = HybridPolicy(
    alpha_lstm=0.95,
    long_weight=0.0,
    risk_penalty=0.30,
    top_k=40,
    rank_power=1.35,
    confidence_power=3.00,
)


def _target_columns(horizon: int) -> tuple[str, str, str]:
    return f"target_{horizon}d", f"idx_target_{horizon}d", f"target_excess_{horizon}d"


def add_horizon_target(panel: pd.DataFrame, index_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_col, idx_col, excess_col = _target_columns(horizon)
    panel = panel.sort_values(["stock_code", "date"]).copy()
    if target_col not in panel.columns:
        panel[target_col] = panel.groupby("stock_code")["close"].shift(-horizon) / panel["close"] - 1.0
    idx = index_df[["date", "close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx[idx_col] = idx["close"].shift(-horizon) / idx["close"] - 1.0
    panel = panel.merge(idx[["date", idx_col]], on="date", how="left")
    panel[excess_col] = panel[target_col] - panel[idx_col]
    return panel.replace([np.inf, -np.inf], np.nan)


def split_train_val(pool: pd.DataFrame, val_days: int, embargo_days: int):
    dates = np.sort(pool["date"].unique())
    val_start = pd.Timestamp(dates[-val_days])
    train_end = pd.Timestamp(dates[-(val_days + embargo_days + 1)])
    return pool[pool["date"] <= train_end].copy(), pool[pool["date"] >= val_start].copy(), train_end, val_start


def daily_z(frame: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    g = frame.groupby("date")[value_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    frame[out_col] = ((frame[value_col] - mean) / std).clip(-4, 4).fillna(0.0)
    return frame


def rank_ic(df: pd.DataFrame, feature: str, target: str) -> float:
    ics = []
    for _, g in df[["date", feature, target]].dropna().groupby("date"):
        if len(g) < 30 or g[feature].nunique() < 5:
            continue
        ic = g[feature].rank().corr(g[target].rank())
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def select_lgb_features(train_df: pd.DataFrame, features: list[str], target_col: str, max_features: int = 34) -> tuple[list[str], pd.DataFrame]:
    dates = np.sort(train_df["date"].unique())
    screen = train_df[train_df["date"] >= pd.Timestamp(dates[-80])] if len(dates) > 80 else train_df
    rows = []
    for feature in features:
        if feature not in screen.columns:
            continue
        rows.append({"feature": feature, "mean_rank_ic": rank_ic(screen, feature, target_col)})
    table = pd.DataFrame(rows)
    table["abs_mean_rank_ic"] = table["mean_rank_ic"].abs()
    table = table.sort_values("abs_mean_rank_ic", ascending=False)
    corr = screen[features].corr(method="spearman").abs()
    selected = []
    for feature in table["feature"]:
        if len(selected) >= max_features:
            break
        if selected and corr.loc[feature, selected].max() >= 0.96:
            continue
        selected.append(feature)
    return selected, table


def fit_lgbm_h3(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp, horizon: int):
    panel = add_horizon_target(build_features(prices, index_df), index_df, horizon)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - horizon)])
    _, _, target_col = _target_columns(horizon)
    features = [c for c in EXPERIMENTAL_FEATURE_COLUMNS if c in panel.columns]
    pool = panel[panel["date"] <= train_cutoff].dropna(subset=[target_col]).copy()
    train_df, val_df, train_end, val_start = split_train_val(pool, DEFAULT_VAL_DAYS, DEFAULT_EMBARGO_DAYS)
    selected_features, feature_table = select_lgb_features(train_df, features, target_col)
    fill_values = train_df[selected_features].median()
    X = train_df[selected_features].fillna(fill_values)
    y = train_df[target_col].astype(float)
    recency = train_df["date"].rank(pct=True).to_numpy()
    tail = y.abs().rank(pct=True).to_numpy()
    weights = 0.85 + 0.35 * recency + 0.35 * tail
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        learning_rate=0.035,
        n_estimators=420,
        num_leaves=95,
        min_child_samples=35,
        subsample=0.86,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.03,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y, sample_weight=weights)
    return {
        "panel": panel,
        "trading_dates": trading_dates,
        "train_df": train_df,
        "val_df": val_df,
        "train_end": train_end,
        "val_start": val_start,
        "features": selected_features,
        "feature_table": feature_table,
        "fill_values": fill_values,
        "model": model,
        "target_col": target_col,
    }


def predict_lgb(fit: dict, dates: list[pd.Timestamp]) -> pd.DataFrame:
    df = fit["panel"][fit["panel"]["date"].isin([pd.Timestamp(d) for d in dates])].copy()
    X = df[fit["features"]].fillna(fit["fill_values"])
    out = df[["date", "stock_code"]].copy()
    out["lgb_raw"] = fit["model"].predict(X)
    return daily_z(out, "lgb_raw", "lgb_score")[["date", "stock_code", "lgb_score"]]


def predict_lstm(fit: dict, dates: list[pd.Timestamp]) -> pd.DataFrame:
    records = lstm.build_records(
        fit["panel"],
        [pd.Timestamp(d) for d in dates],
        target_horizon=fit.get("target_horizon", DEFAULT_HORIZON),
        require_targets=False,
    )
    records = fit["normalizer"].apply(records)
    pred = lstm.predict_records(fit["model"], records)
    if pred.empty:
        return pd.DataFrame(columns=["date", "stock_code", "lstm_score"])
    pred = daily_z(pred, "rank_pred", "lstm_score")
    return pred[["date", "stock_code", "lstm_score"]]


def build_signal_features(panel: pd.DataFrame, train_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in sorted(set(LONG_FEATURES + RISK_FEATURES)) if c in panel.columns]
    signal = panel[["date", "stock_code"] + cols].copy()
    for col in cols:
        signal = daily_z(signal, col, f"{col}_z")
        if col in RISK_ABS_FEATURES:
            signal[f"{col}_risk_raw"] = signal[col].abs()
        elif col in RISK_NEGATIVE_FEATURES:
            signal[f"{col}_risk_raw"] = -signal[col]
        else:
            signal[f"{col}_risk_raw"] = signal[col]
        signal = daily_z(signal, f"{col}_risk_raw", f"{col}_risk_z")
    rows = []
    for col in [c for c in LONG_FEATURES if c in panel.columns]:
        rows.append({"feature": col, "mean_rank_ic": rank_ic(train_df, col, target_col)})
    feature_ic = pd.DataFrame(rows)
    long_cols = []
    for col in [c for c in LONG_FEATURES if c in panel.columns]:
        sign = 1.0
        if not feature_ic.empty:
            match = feature_ic[feature_ic["feature"] == col]
            if not match.empty and float(match["mean_rank_ic"].iloc[0]) < 0:
                sign = -1.0
        signal[f"{col}_signed"] = sign * signal[f"{col}_z"]
        long_cols.append(f"{col}_signed")
    risk_z = [f"{c}_risk_z" for c in RISK_FEATURES if c in panel.columns]
    signal["long_signal"] = signal[long_cols].mean(axis=1).fillna(0.0) if long_cols else 0.0
    signal["risk_signal"] = signal[risk_z].mean(axis=1).fillna(0.0) if risk_z else 0.0
    signal["stability_signal"] = -signal["risk_signal"]
    return signal[["date", "stock_code", "long_signal", "risk_signal", "stability_signal"]], feature_ic


def assemble_frame(lstm_pred: pd.DataFrame, lgb_pred: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    out = lgb_pred.merge(lstm_pred, on=["date", "stock_code"], how="inner")
    out = out.merge(signals, on=["date", "stock_code"], how="left")
    out["long_signal"] = out["long_signal"].fillna(0.0)
    out["risk_signal"] = out["risk_signal"].fillna(0.0)
    out["stability_signal"] = out["stability_signal"].fillna(0.0)
    out["model_agreement"] = 1.0 - (out["lstm_score"] - out["lgb_score"]).abs().rank(pct=True)
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


def weights_for_date(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    policy: HybridPolicy,
    confidence_mode: str = "risk-balanced",
) -> pd.Series:
    d = frame[frame["date"] == pd.Timestamp(as_of)].copy()
    d["score"] = (
        policy.alpha_lstm * d["lstm_score"]
        + (1.0 - policy.alpha_lstm) * d["lgb_score"]
        + policy.long_weight * d["long_signal"]
    )
    if confidence_mode == "enhanced":
        lstm_rank = _rank_pct(d["lstm_score"])
        lgb_rank = _rank_pct(d["lgb_score"])
        directional_agreement = np.sqrt(lstm_rank * lgb_rank)
        confidence_raw = (
            0.30 * directional_agreement
            + 0.25 * _rank_pct(d["score"])
            + 0.18 * d["model_agreement"].fillna(0.5)
            + 0.17 * _rank_pct(d["stability_signal"])
            + 0.10 * _rank_pct(d["long_signal"].abs())
            - policy.risk_penalty * _rank_pct(d["risk_signal"])
        )
    else:
        confidence_raw = (
            0.45 * d["model_agreement"]
            + 0.30 * d["score"].abs().rank(pct=True)
            + 0.25 * d["long_signal"].abs().rank(pct=True)
            - policy.risk_penalty * d["risk_signal"].rank(pct=True)
        )
    d["confidence"] = confidence_raw.rank(method="average", pct=True).clip(0.05, 1.0)
    d = d.sort_values("score", ascending=False).head(policy.top_k)
    ranks = np.arange(len(d), 0, -1, dtype=float)
    raw = (ranks ** policy.rank_power) * (d["confidence"].to_numpy() ** policy.confidence_power)
    return apply_cap(pd.Series(raw, index=d["stock_code"].astype(str)))


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


def precompute_window_returns(prices: pd.DataFrame, index_df: pd.DataFrame, windows) -> dict[pd.Timestamp, tuple[pd.Series, float]]:
    close = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    close.index = pd.to_datetime(close.index)
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").set_index("date")
    cache = {}
    for as_of, start, end in windows:
        before = close.index[close.index < start]
        if len(before) == 0:
            continue
        entry = close.loc[before[-1]]
        exit_ = close[(close.index >= start) & (close.index <= end)].ffill().iloc[-1]
        stock_rets = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        idx_before = idx[idx.index < start]
        idx_win = idx[(idx.index >= start) & (idx.index <= end)]
        if idx_before.empty or idx_win.empty:
            continue
        bench = float(idx_win["close"].iloc[-1] / idx_before["close"].iloc[-1] - 1.0)
        cache[pd.Timestamp(as_of)] = (stock_rets, bench)
    return cache


def policy_grid():
    for alpha in (0.40, 0.50, 0.60, 0.70, 0.80):
        for long_weight in (0.0, 0.10, 0.20, 0.35):
            for risk_penalty in (0.0, 0.15, 0.30, 0.45):
                for top_k in (30, 35, 40, 45, 50):
                    for rank_power in (0.85, 1.10, 1.35, 1.60):
                        for confidence_power in (0.0, 0.50, 1.00, 1.50):
                            yield HybridPolicy(alpha, long_weight, risk_penalty, top_k, rank_power, confidence_power)


def policy_from_args(args: argparse.Namespace) -> HybridPolicy | None:
    if args.policy_mode == "validation":
        return None
    if args.policy_mode == "aggressive-lstm":
        return AGGRESSIVE_LSTM_POLICY
    return HybridPolicy(
        alpha_lstm=args.fixed_alpha_lstm,
        long_weight=args.fixed_long_weight,
        risk_penalty=args.fixed_risk_penalty,
        top_k=args.fixed_top_k,
        rank_power=args.fixed_rank_power,
        confidence_power=args.fixed_confidence_power,
    )


def select_policy(
    frame: pd.DataFrame,
    val_dates: np.ndarray,
    trading_dates: np.ndarray,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    horizon: int,
    confidence_mode: str = "risk-balanced",
):
    windows = validation_windows(val_dates, trading_dates, horizon)
    returns = precompute_window_returns(prices, index_df, windows)
    rows = []
    for policy in policy_grid():
        scores = []
        for d, _, _ in windows:
            if d not in returns:
                continue
            try:
                weights = weights_for_date(frame, d, policy, confidence_mode=confidence_mode)
            except ValueError:
                continue
            stock_rets, bench = returns[d]
            scores.append(float((weights * stock_rets.reindex(weights.index).fillna(0.0)).sum() - bench))
        if scores:
            mean = float(np.mean(scores))
            min_ = float(np.min(scores))
            std = float(np.std(scores))
            rows.append({
                **policy.__dict__,
                "mean_excess_return": mean,
                "sum_excess_return": float(np.sum(scores)),
                "min_excess_return": min_,
                "std_excess_return": std,
                "utility_score": mean + 0.35 * min_ - 0.20 * std,
                "n_windows": len(scores),
            })
    table = pd.DataFrame(rows).sort_values(["utility_score", "mean_excess_return", "min_excess_return"], ascending=False)
    best = table.iloc[0]
    policy = HybridPolicy(
        float(best["alpha_lstm"]),
        float(best["long_weight"]),
        float(best["risk_penalty"]),
        int(best["top_k"]),
        float(best["rank_power"]),
        float(best["confidence_power"]),
    )
    return policy, table


def fit_hybrid(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
    lookback_days: int,
    fixed_policy: HybridPolicy | None = None,
    confidence_mode: str = "risk-balanced",
):
    as_of = pd.Timestamp(as_of)
    if lookback_days:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices_fit = prices[prices["date"] >= min_date].copy()
        index_fit = index_df[index_df["date"] >= min_date].copy()
    else:
        prices_fit = prices.copy()
        index_fit = index_df.copy()
    lgb_fit = fit_lgbm_h3(prices_fit, index_fit, as_of, horizon)
    lstm_fit = lstm.fit_lstm(prices_fit, index_fit, as_of, policy_horizon=horizon, target_horizon=horizon)
    val_dates = [pd.Timestamp(d) for d in np.sort(lgb_fit["val_df"]["date"].unique())]
    pred_dates = sorted(set(val_dates + [as_of]))
    signal_frame, signal_ic = build_signal_features(lgb_fit["panel"], lgb_fit["train_df"], lgb_fit["target_col"])
    frame = assemble_frame(
        predict_lstm(lstm_fit, pred_dates),
        predict_lgb(lgb_fit, pred_dates),
        signal_frame,
    )
    if fixed_policy is None:
        policy, policy_table = select_policy(
            frame,
            np.asarray(val_dates),
            lgb_fit["trading_dates"],
            prices_fit,
            index_fit,
            horizon,
            confidence_mode=confidence_mode,
        )
    else:
        policy = fixed_policy
        policy_table = pd.DataFrame([{**policy.__dict__, "policy_mode": "fixed"}])
    return {
        "frame": frame,
        "policy": policy,
        "policy_table": policy_table,
        "lgb_feature_table": lgb_fit["feature_table"],
        "signal_ic": signal_ic,
        "lgb_fit": lgb_fit,
    }


def generate_submission(fit: dict, as_of: pd.Timestamp, confidence_mode: str = "risk-balanced") -> pd.DataFrame:
    weights = weights_for_date(
        fit["frame"],
        pd.Timestamp(as_of),
        fit["policy"],
        confidence_mode=confidence_mode,
    )
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--out", default=str(REPORT_DIR / "generated" / "stage1_lstm_lgb_confidence.csv"))
    parser.add_argument("--policy-report-out", default=None)
    parser.add_argument(
        "--confidence-mode",
        choices=["risk-balanced", "legacy", "enhanced"],
        default="risk-balanced",
        help=(
            "risk-balanced uses the tuned confidence layer with correctly oriented risk; "
            "enhanced adds stricter directional model confirmation."
        ),
    )
    parser.add_argument(
        "--policy-mode",
        choices=["validation", "aggressive-lstm", "fixed"],
        default="validation",
        help=(
            "validation tunes the portfolio layer on recent validation windows; "
            "aggressive-lstm uses the short-window LSTM confidence profile; "
            "fixed uses the --fixed-* arguments."
        ),
    )
    parser.add_argument("--fixed-alpha-lstm", type=float, default=AGGRESSIVE_LSTM_POLICY.alpha_lstm)
    parser.add_argument("--fixed-long-weight", type=float, default=AGGRESSIVE_LSTM_POLICY.long_weight)
    parser.add_argument("--fixed-risk-penalty", type=float, default=AGGRESSIVE_LSTM_POLICY.risk_penalty)
    parser.add_argument("--fixed-top-k", type=int, default=AGGRESSIVE_LSTM_POLICY.top_k)
    parser.add_argument("--fixed-rank-power", type=float, default=AGGRESSIVE_LSTM_POLICY.rank_power)
    parser.add_argument("--fixed-confidence-power", type=float, default=AGGRESSIVE_LSTM_POLICY.confidence_power)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(prices["date"].max())

    fixed_policy = policy_from_args(args)
    fit = fit_hybrid(
        prices,
        index_df,
        as_of,
        args.horizon,
        args.lookback_days,
        fixed_policy=fixed_policy,
        confidence_mode=args.confidence_mode,
    )
    submission = generate_submission(fit, as_of, confidence_mode=args.confidence_mode)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if args.policy_report_out:
        report_path = Path(args.policy_report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fit["policy_table"].to_csv(report_path, index=False)
    print(f">> model=stage1_lstm_lgb_confidence as_of={as_of.date()} horizon={args.horizon}")
    print(f">> policy_mode={args.policy_mode}")
    print(f">> confidence_mode={args.confidence_mode}")
    print(f">> selected policy={fit['policy']}")
    print(">> policy table")
    print(fit["policy_table"].head(12).to_string(index=False))
    print(">> lgb feature rank-IC")
    print(fit["lgb_feature_table"].head(12).to_string(index=False))
    print(">> signal rank-IC")
    print(fit["signal_ic"].to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
