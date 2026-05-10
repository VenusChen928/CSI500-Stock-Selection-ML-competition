"""
Stage1 OHLCV signal/confidence portfolio.

This is an independent 3-trading-day experiment.  It ignores optional open-data
files and first converts original OHLCV/index data into interpretable daily
signals:

  - long_signal: smoothed medium/long trend, with feature direction learned from
    recent rank-IC.
  - short_signal: short-horizon reversal/continuation and price-action signal.
  - risk_signal: short-horizon jitter/volatility/liquidity instability.
  - confidence: trend strength and long/short agreement, penalized by jitter.

Validation then chooses how to combine score, confidence, top-k, and weight
curve by the canonical score_submission.py excess-return metric.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import MAX_WEIGHT, MIN_STOCKS
from score_submission import score_window

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_HORIZON = 3
DEFAULT_LOOKBACK_DAYS = 520
DEFAULT_VAL_DAYS = 18
DEFAULT_EMBARGO_DAYS = 3


LONG_FEATURES = [
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "excess_ret_20d",
    "excess_ret_60d",
    "ma_gap_20d",
    "ma_gap_60d",
    "ma_slope_20d",
    "ma_slope_60d",
    "ema_gap_20d",
    "ema_gap_60d",
]

SHORT_FEATURES = [
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_5d",
    "excess_ret_1d",
    "excess_ret_3d",
    "intraday_ret",
    "overnight_ret",
    "close_pos_10d",
    "mom_accel_3_10",
    "reversal_3d",
    "volume_z_5d",
    "amount_z_5d",
    "turnover_z_5d",
]

RISK_FEATURES = [
    "vol_3d",
    "vol_5d",
    "vol_10d",
    "vol_ratio_3_20",
    "range_ma_3d",
    "range_ma_5d",
    "overnight_abs_5d",
    "intraday_abs_5d",
    "jump_abs_3d",
    "drawdown_20d_abs",
    "turnover_z_5d_abs",
    "amount_z_5d_abs",
]


@dataclass(frozen=True)
class SignalSpec:
    long_features: tuple[str, ...]
    short_features: tuple[str, ...]
    risk_features: tuple[str, ...]
    long_signs: tuple[float, ...]
    short_signs: tuple[float, ...]


@dataclass(frozen=True)
class Policy:
    long_weight: float
    short_weight: float
    risk_penalty: float
    top_k: int
    rank_power: float
    confidence_power: float


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _per_stock_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    stock_code = getattr(df, "name", None)
    df = df.sort_values("date").copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)
    turnover = df["turnover"].astype(float)

    for n in (1, 2, 3, 5, 10, 20, 60, 120):
        df[f"ret_{n}d"] = close.pct_change(n)
    for n in (10, 20, 60):
        ma = close.rolling(n).mean()
        ema = close.ewm(span=n, adjust=False, min_periods=max(5, n // 3)).mean()
        df[f"ma_gap_{n}d"] = close / ma - 1.0
        df[f"ema_gap_{n}d"] = close / ema - 1.0
        df[f"ma_slope_{n}d"] = ma / ma.shift(min(10, n)) - 1.0

    ret1 = close.pct_change(1)
    for n in (3, 5, 10, 20):
        df[f"vol_{n}d"] = ret1.rolling(n).std()
    df["vol_ratio_3_20"] = _safe_div(df["vol_3d"], df["vol_20d"])
    df["vol_ratio_5_20"] = _safe_div(df["vol_5d"], df["vol_20d"])

    range_ = high / low.replace(0, np.nan) - 1.0
    df["intraday_ret"] = close / open_.replace(0, np.nan) - 1.0
    df["overnight_ret"] = open_ / close.shift(1).replace(0, np.nan) - 1.0
    df["high_low_range"] = range_
    df["range_ma_3d"] = range_.rolling(3).mean()
    df["range_ma_5d"] = range_.rolling(5).mean()
    df["overnight_abs_5d"] = df["overnight_ret"].abs().rolling(5).mean()
    df["intraday_abs_5d"] = df["intraday_ret"].abs().rolling(5).mean()
    df["jump_abs_3d"] = ret1.abs().rolling(3).max()

    rolling_high_10 = high.rolling(10).max()
    rolling_low_10 = low.rolling(10).min()
    rolling_high_20 = high.rolling(20).max()
    df["close_pos_10d"] = (close - rolling_low_10) / (rolling_high_10 - rolling_low_10).replace(0, np.nan)
    df["drawdown_20d"] = close / rolling_high_20.replace(0, np.nan) - 1.0
    df["drawdown_20d_abs"] = df["drawdown_20d"].abs()

    for n in (5, 20):
        vol_mean = volume.rolling(n).mean()
        vol_std = volume.rolling(n).std()
        amount_mean = amount.rolling(n).mean()
        amount_std = amount.rolling(n).std()
        turnover_mean = turnover.rolling(n).mean()
        turnover_std = turnover.rolling(n).std()
        df[f"volume_z_{n}d"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        df[f"amount_z_{n}d"] = (amount - amount_mean) / amount_std.replace(0, np.nan)
        df[f"turnover_z_{n}d"] = (turnover - turnover_mean) / turnover_std.replace(0, np.nan)
    df["turnover_z_5d_abs"] = df["turnover_z_5d"].abs()
    df["amount_z_5d_abs"] = df["amount_z_5d"].abs()
    df["mom_accel_3_10"] = df["ret_3d"] - df["ret_10d"]
    df["reversal_3d"] = -df["ret_3d"]

    df[f"target_{horizon}d"] = close.shift(-horizon) / close - 1.0
    if stock_code is not None and "stock_code" not in df.columns:
        df["stock_code"] = stock_code
    return df


def _index_features(index_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    close = idx["close"].astype(float)
    for n in (1, 3, 5, 20, 60):
        idx[f"idx_ret_{n}d"] = close.pct_change(n)
    idx[f"idx_target_{horizon}d"] = close.shift(-horizon) / close - 1.0
    return idx[["date", "idx_ret_1d", "idx_ret_3d", "idx_ret_5d", "idx_ret_20d", "idx_ret_60d", f"idx_target_{horizon}d"]]


def robust_cs_z(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    zcols = {}
    grouped = panel.groupby("date", sort=False)
    for col in columns:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0, np.nan)
        zcols[f"{col}_z"] = ((panel[col] - mean) / std).clip(-4.0, 4.0).fillna(0.0)
    return pd.concat([panel, pd.DataFrame(zcols, index=panel.index)], axis=1)


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
    panel = panel.merge(_index_features(index_df, horizon), on="date", how="left")
    for n in (1, 3, 5, 20, 60):
        panel[f"excess_ret_{n}d"] = panel[f"ret_{n}d"] - panel[f"idx_ret_{n}d"]
    panel[f"target_excess_{horizon}d"] = panel[f"target_{horizon}d"] - panel[f"idx_target_{horizon}d"]
    needed = sorted(set(LONG_FEATURES + SHORT_FEATURES + RISK_FEATURES))
    panel = robust_cs_z(panel.replace([np.inf, -np.inf], np.nan), needed)
    return panel


def split_train_val(panel: pd.DataFrame, as_of: pd.Timestamp, horizon: int, val_days: int, embargo_days: int):
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - horizon)])
    target_col = f"target_excess_{horizon}d"
    pool = panel[panel["date"] <= train_cutoff].dropna(subset=[target_col]).copy()
    dates = np.sort(pool["date"].unique())
    val_start = pd.Timestamp(dates[-val_days])
    train_end = pd.Timestamp(dates[-(val_days + embargo_days + 1)])
    train_df = pool[pool["date"] <= train_end].copy()
    val_df = pool[pool["date"] >= val_start].copy()
    return train_df, val_df, trading_dates, train_end, val_start


def rank_ic(df: pd.DataFrame, z_col: str, target_col: str) -> float:
    ics = []
    for _, g in df[["date", z_col, target_col]].dropna().groupby("date"):
        if len(g) < 30 or g[z_col].nunique() < 5:
            continue
        ic = g[z_col].rank().corr(g[target_col].rank())
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def select_group_features(
    train_df: pd.DataFrame,
    raw_features: list[str],
    target_col: str,
    max_features: int,
    corr_threshold: float,
) -> tuple[list[str], list[float], pd.DataFrame]:
    z_features = [f"{f}_z" for f in raw_features if f"{f}_z" in train_df.columns]
    rows = []
    for col in z_features:
        ic = rank_ic(train_df, col, target_col)
        rows.append({"feature": col, "mean_rank_ic": ic, "abs_mean_rank_ic": abs(ic), "direction": 1.0 if ic >= 0 else -1.0})
    table = pd.DataFrame(rows).sort_values("abs_mean_rank_ic", ascending=False)
    corr = train_df[z_features].corr(method="spearman").abs()
    selected = []
    signs = []
    for _, row in table.iterrows():
        col = row["feature"]
        if len(selected) >= max_features:
            break
        if selected and corr.loc[col, selected].max() >= corr_threshold:
            continue
        selected.append(col)
        signs.append(float(row["direction"]))
    return selected, signs, table


def build_signal_frame(panel: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    out = panel[["date", "stock_code"]].copy()
    if spec.long_features:
        long_arr = panel[list(spec.long_features)].fillna(0.0).to_numpy() * np.asarray(spec.long_signs)
        out["long_signal"] = long_arr.mean(axis=1)
    else:
        out["long_signal"] = 0.0
    if spec.short_features:
        short_arr = panel[list(spec.short_features)].fillna(0.0).to_numpy() * np.asarray(spec.short_signs)
        out["short_signal"] = short_arr.mean(axis=1)
    else:
        out["short_signal"] = 0.0
    if spec.risk_features:
        out["risk_signal"] = panel[list(spec.risk_features)].fillna(0.0).mean(axis=1)
    else:
        out["risk_signal"] = 0.0
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


def weights_for_date(signals: pd.DataFrame, as_of: pd.Timestamp, policy: Policy) -> pd.Series:
    d = signals[signals["date"] == pd.Timestamp(as_of)].copy()
    score = (
        policy.long_weight * d["long_signal"]
        + policy.short_weight * d["short_signal"]
        - policy.risk_penalty * d["risk_signal"]
    )
    confidence_raw = (
        0.65 * d["long_signal"].abs()
        + 0.35 * np.maximum(d["long_signal"] * d["short_signal"], 0.0)
        - 0.70 * d["risk_signal"]
    )
    d["score"] = score
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


def precompute_window_returns(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> dict[pd.Timestamp, tuple[pd.Series, float]]:
    close = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    close.index = pd.to_datetime(close.index)
    idx = index_df.sort_values("date").copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")
    cache = {}
    for as_of, start, end in windows:
        before_dates = close.index[close.index < start]
        if len(before_dates) == 0:
            continue
        entry = close.loc[before_dates[-1]]
        in_window = close[(close.index >= start) & (close.index <= end)]
        if in_window.empty:
            continue
        exit_ = in_window.ffill().iloc[-1]
        rets = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        idx_before = idx[idx.index < start]
        idx_window = idx[(idx.index >= start) & (idx.index <= end)]
        if idx_before.empty or idx_window.empty:
            continue
        bench = float(idx_window["close"].iloc[-1] / idx_before["close"].iloc[-1] - 1.0)
        cache[pd.Timestamp(as_of)] = (rets, bench)
    return cache


def policy_grid():
    for long_weight in (0.45, 0.60, 0.75):
        for short_weight in (0.35, 0.50, 0.65):
            for risk_penalty in (0.0, 0.20, 0.40):
                for top_k in (30, 35, 40, 50, 60):
                    for rank_power in (0.85, 1.15, 1.45):
                        for confidence_power in (0.0, 0.75, 1.25):
                            yield Policy(long_weight, short_weight, risk_penalty, top_k, rank_power, confidence_power)


def select_policy(signals: pd.DataFrame, val_df: pd.DataFrame, trading_dates: np.ndarray, prices: pd.DataFrame, index_df: pd.DataFrame, horizon: int):
    windows = validation_windows(np.sort(val_df["date"].unique()), trading_dates, horizon)
    returns_cache = precompute_window_returns(prices, index_df, windows)
    rows = []
    for policy in policy_grid():
        scores = []
        for d, _, _ in windows:
            if d not in returns_cache:
                continue
            try:
                weights = weights_for_date(signals, d, policy)
            except ValueError:
                continue
            stock_rets, bench_ret = returns_cache[d]
            aligned = stock_rets.reindex(weights.index).fillna(0.0)
            scores.append(float((weights * aligned).sum() - bench_ret))
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
                "utility_score": mean + 0.5 * min_ - 0.25 * std,
                "n_windows": len(scores),
            })
    table = pd.DataFrame(rows).sort_values(["utility_score", "mean_excess_return", "min_excess_return"], ascending=False)
    best = table.iloc[0]
    policy = Policy(
        float(best["long_weight"]),
        float(best["short_weight"]),
        float(best["risk_penalty"]),
        int(best["top_k"]),
        float(best["rank_power"]),
        float(best["confidence_power"]),
    )
    return policy, table


def fit_signal_model(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp, horizon: int, lookback_days: int, val_days: int):
    as_of = pd.Timestamp(as_of)
    if lookback_days:
        min_date = as_of - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()
    panel = build_panel(prices, index_df, horizon)
    train_df, val_df, trading_dates, train_end, val_start = split_train_val(panel, as_of, horizon, val_days, DEFAULT_EMBARGO_DAYS)
    target_col = f"target_excess_{horizon}d"
    long_features, long_signs, long_ic = select_group_features(train_df, LONG_FEATURES, target_col, 5, 0.90)
    short_features, short_signs, short_ic = select_group_features(train_df, SHORT_FEATURES, target_col, 7, 0.90)
    risk_features, _, risk_ic = select_group_features(train_df, RISK_FEATURES, target_col, 5, 0.90)
    spec = SignalSpec(tuple(long_features), tuple(short_features), tuple(risk_features), tuple(long_signs), tuple(short_signs))
    signals = build_signal_frame(panel, spec)
    policy, policy_table = select_policy(signals, val_df, trading_dates, prices, index_df, horizon)
    return {
        "panel": panel,
        "signals": signals,
        "spec": spec,
        "policy": policy,
        "policy_table": policy_table,
        "feature_ic": pd.concat([
            long_ic.assign(group="long"),
            short_ic.assign(group="short"),
            risk_ic.assign(group="risk"),
        ], ignore_index=True),
        "train_end": train_end,
        "val_start": val_start,
        "train_df": train_df,
        "val_df": val_df,
    }


def generate_submission(fit: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    weights = weights_for_date(fit["signals"], pd.Timestamp(as_of), fit["policy"])
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--out", default="submissions/stage1_signal_confidence.csv")
    parser.add_argument("--feature-report-out", default=None)
    parser.add_argument("--policy-report-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(prices["date"].max())

    fit = fit_signal_model(prices, index_df, as_of, args.horizon, args.lookback_days, args.val_days)
    submission = generate_submission(fit, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if args.feature_report_out:
        Path(args.feature_report_out).parent.mkdir(parents=True, exist_ok=True)
        fit["feature_ic"].to_csv(args.feature_report_out, index=False)
    if args.policy_report_out:
        Path(args.policy_report_out).parent.mkdir(parents=True, exist_ok=True)
        fit["policy_table"].to_csv(args.policy_report_out, index=False)

    print(f">> model=stage1_signal_confidence as_of={as_of.date()} horizon={args.horizon}")
    print(f">> train rows={len(fit['train_df']):,} train_end={fit['train_end'].date()}")
    print(f">> val rows={len(fit['val_df']):,} val_start={fit['val_start'].date()}")
    print(f">> selected spec={fit['spec']}")
    print(f">> selected policy={fit['policy']}")
    print(">> policy table")
    print(fit["policy_table"].head(12).to_string(index=False))
    print(">> feature ic")
    print(fit["feature_ic"].sort_values(["group", "abs_mean_rank_ic"], ascending=[True, False]).head(20).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
