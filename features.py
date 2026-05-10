"""
Feature engineering for the CSI500 stock-selection baseline.

A small set of classic technical features + cross-sectional ranks.  Students are
encouraged to extend this (add fundamentals, industry dummies, alternative data,
better cross-sectional normalization, etc.).

The target is the 5-trading-day forward return on the forward-adjusted close,
i.e. what the portfolio earns if you hold a $1 position from close(t) to close(t+5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# Stable feature set selected by score-window ablation.  The richer candidate
# features are still computed below for research, but the production model uses
# this compact set because broad feature stuffing hurt recent out-of-sample
# excess return.
CORE_FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
]

# Reference feature set inspired by the external stage1 project.  Kept separate
# from CORE so we can test it honestly before promoting it to production.
REFERENCE_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "amplitude_ma_20d",
    "amplitude_ma_20d_rank",
]

MOMENTUM_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "ret_3d",
    "ret_3d_rank",
    "ret_10d_rank",
    "ret_60d_rank",
    "mom_accel_5_20",
    "mom_accel_5_20_rank",
    "vol_5d",
    "vol_ratio_5_20",
    "vol_ratio_5_20_rank",
    "amount_z_20d",
    "amount_z_20d_rank",
    "turnover_z_20d",
    "turnover_z_20d_rank",
    "close_pos_20d",
    "close_pos_20d_rank",
    "drawdown_20d",
    "drawdown_20d_rank",
    "excess_ret_5d",
    "excess_ret_20d",
    "excess_ret_5d_rank",
    "excess_ret_20d_rank",
    "breadth_ret_5d_pos",
    "dispersion_ret_5d",
]

QUALITY_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "trend_quality_20d",
    "trend_quality_20d_rank",
    "trend_quality_60d",
    "trend_quality_60d_rank",
    "trend_efficiency_20d",
    "trend_efficiency_20d_rank",
    "vol_ratio_20_60",
    "vol_ratio_20_60_rank",
    "downside_vol_20d",
    "downside_vol_20d_rank",
    "beta_60d",
    "beta_60d_rank",
    "residual_ret_20d",
    "residual_ret_20d_rank",
    "amount_z_20d",
    "amount_z_20d_rank",
    "volume_price_confirm_20d",
    "close_pos_20d",
    "close_pos_20d_rank",
    "drawdown_20d",
    "drawdown_20d_rank",
]

ALPHA_FEATURE_COLUMNS = [
    "ret_2d",
    "ret_15d",
    "ret_30d",
    "ret_120d",
    "ret_2d_rank",
    "ret_15d_rank",
    "ret_30d_rank",
    "ret_120d_rank",
    "ma5_over_ma20",
    "ma20_over_ma60",
    "ma5_over_ma20_rank",
    "ma20_over_ma60_rank",
    "range_vol_20d",
    "range_vol_20d_rank",
    "amplitude_z_20d",
    "amplitude_z_20d_rank",
    "amount_trend_5_20",
    "amount_trend_5_20_rank",
    "turnover_trend_5_20",
    "turnover_trend_5_20_rank",
    "price_volume_corr_20d",
    "price_volume_corr_20d_rank",
    "obv_20d",
    "obv_20d_rank",
    "close_location_1d",
    "close_location_ma5",
    "close_location_ma20",
    "close_location_ma5_rank",
    "close_location_ma20_rank",
    "gap_mean_5d",
    "gap_mean_20d",
    "intraday_mean_5d",
    "intraday_mean_20d",
    "gap_mean_5d_rank",
    "gap_mean_20d_rank",
    "intraday_mean_5d_rank",
    "intraday_mean_20d_rank",
    "overnight_vol_20d",
    "intraday_vol_20d",
    "overnight_vol_20d_rank",
    "intraday_vol_20d_rank",
    "ret_max_20d",
    "ret_min_20d",
    "ret_skew_20d",
    "ret_kurt_20d",
    "ret_max_20d_rank",
    "ret_min_20d_rank",
    "ret_skew_20d_rank",
    "ret_kurt_20d_rank",
    "market_corr_60d",
    "market_corr_60d_rank",
]

CALENDAR_FEATURE_COLUMNS = [
    "asof_weekday_norm",
    "asof_is_monday",
    "asof_is_friday",
    "asof_month_pos",
    "asof_month_start",
    "asof_month_end",
    "eval_starts_monday",
    "eval_ends_friday",
    "eval_is_full_workweek",
    "eval_crosses_month",
    "eval_month_start",
    "eval_month_end",
    "weekend_gap_days",
    "weekly_risk_appetite",
    "weekly_friday_derisk",
    "month_start_flow",
    "month_end_defensive",
    "post_gap_reopen_flow",
    "weekly_carry_quality",
    "weekly_risk_appetite_rank",
    "weekly_friday_derisk_rank",
    "month_start_flow_rank",
    "month_end_defensive_rank",
    "post_gap_reopen_flow_rank",
    "weekly_carry_quality_rank",
]

WEEKLY_CYCLE_FEATURE_COLUMNS = list(
    dict.fromkeys(
        MOMENTUM_FEATURE_COLUMNS
        + QUALITY_FEATURE_COLUMNS
        + ALPHA_FEATURE_COLUMNS
        + CALENDAR_FEATURE_COLUMNS
    )
)

CANDIDATE_FEATURE_GROUPS = {
    "short_reversal": ["ret_3d", "mom_accel_5_20", "reversal_3d"],
    "risk_liquidity": ["vol_5d", "vol_ratio_5_20", "amount_z_20d", "turnover_z_20d", "amount_z_20d_rank"],
    "price_action": [
        "intraday_ret", "overnight_ret", "high_low_range",
        "amplitude_ma_20d", "close_pos_20d", "drawdown_20d",
        "amplitude_ma_20d_rank",
    ],
    "market_relative": [
        "idx_ret_5d", "idx_ret_20d", "idx_vol_20d",
        "excess_ret_5d", "excess_ret_20d",
        "excess_ret_5d_rank", "excess_ret_20d_rank",
    ],
    "market_state": ["breadth_ret_5d_pos", "dispersion_ret_5d"],
    "alpha_microstructure": ALPHA_FEATURE_COLUMNS,
    "calendar_weekly_cycle": CALENDAR_FEATURE_COLUMNS,
}

EXPERIMENTAL_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    feature
    for group in CANDIDATE_FEATURE_GROUPS.values()
    for feature in group
]

# columns used downstream by the production baseline
FEATURE_COLUMNS = CORE_FEATURE_COLUMNS
TARGET_COLUMN = "target_5d"
TARGET_3D_COLUMN = "target_3d"
TARGET_EXCESS_COLUMN = "target_excess_5d"
TARGET_EXCESS_3D_COLUMN = "target_excess_3d"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    stock_code = getattr(df, "name", None)
    df = df.sort_values("date").copy()
    close = df["close"]
    open_ = df["open"] if "open" in df.columns else close
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    df["ret_1d"] = close.pct_change(1)
    df["ret_2d"] = close.pct_change(2)
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_15d"] = close.pct_change(15)
    df["ret_20d"] = close.pct_change(20)
    df["ret_30d"] = close.pct_change(30)
    df["ret_60d"] = close.pct_change(60)
    df["ret_120d"] = close.pct_change(120)
    df["mom_accel_5_20"] = df["ret_5d"] - df["ret_20d"]
    df["reversal_3d"] = -df["ret_3d"]

    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_60d"] = df["ret_1d"].rolling(60).std()
    df["vol_ratio_5_20"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)
    df["vol_ratio_20_60"] = df["vol_20d"] / df["vol_60d"].replace(0, np.nan)
    downside = df["ret_1d"].clip(upper=0.0)
    df["downside_vol_20d"] = downside.rolling(20).std()
    path_length_20d = df["ret_1d"].abs().rolling(20).sum()
    df["trend_efficiency_20d"] = df["ret_20d"].abs() / path_length_20d.replace(0, np.nan)
    df["trend_quality_20d"] = df["ret_20d"] / (df["vol_20d"].replace(0, np.nan) * np.sqrt(20.0))
    df["trend_quality_60d"] = df["ret_60d"] / (df["vol_60d"].replace(0, np.nan) * np.sqrt(60.0))

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "amount" in df.columns:
        amount = df["amount"].astype(float)
        amount_mean = amount.rolling(20).mean()
        amount_std = amount.rolling(20).std().replace(0, np.nan)
        df["amount_z_20d"] = (amount - amount_mean) / amount_std
    else:
        df["amount_z_20d"] = np.nan

    if "turnover" in df.columns:
        turnover = df["turnover"].astype(float)
        turnover_mean = turnover.rolling(20).mean()
        turnover_std = turnover.rolling(20).std().replace(0, np.nan)
        df["turnover_ma_20d"] = turnover_mean
        df["turnover_z_20d"] = (turnover - turnover_mean) / turnover_std
    else:
        df["turnover_ma_20d"] = np.nan
        df["turnover_z_20d"] = np.nan

    df["intraday_ret"] = close / open_.replace(0, np.nan) - 1.0
    df["overnight_ret"] = open_ / close.shift(1).replace(0, np.nan) - 1.0
    df["gap_mean_5d"] = df["overnight_ret"].rolling(5).mean()
    df["gap_mean_20d"] = df["overnight_ret"].rolling(20).mean()
    df["intraday_mean_5d"] = df["intraday_ret"].rolling(5).mean()
    df["intraday_mean_20d"] = df["intraday_ret"].rolling(20).mean()
    df["overnight_vol_20d"] = df["overnight_ret"].rolling(20).std()
    df["intraday_vol_20d"] = df["intraday_ret"].rolling(20).std()
    df["high_low_range"] = high / low.replace(0, np.nan) - 1.0
    prev_close = close.shift(1).replace(0, np.nan)
    df["amplitude_1d"] = (high - low) / prev_close
    df["amplitude_ma_20d"] = df["amplitude_1d"].rolling(20).mean()
    amplitude_std_20d = df["amplitude_1d"].rolling(20).std().replace(0, np.nan)
    df["amplitude_z_20d"] = (df["amplitude_1d"] - df["amplitude_ma_20d"]) / amplitude_std_20d
    df["range_vol_20d"] = df["high_low_range"].rolling(20).std()
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    df["ma5_over_ma20"] = ma5 / ma20.replace(0, np.nan) - 1.0
    df["ma20_over_ma60"] = ma20 / ma60.replace(0, np.nan) - 1.0
    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    df["close_pos_20d"] = (close - rolling_low_20) / (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    df["drawdown_20d"] = close / close.rolling(20).max().replace(0, np.nan) - 1.0
    daily_range = (high - low).replace(0, np.nan)
    df["close_location_1d"] = (close - low) / daily_range
    df["close_location_ma5"] = df["close_location_1d"].rolling(5).mean()
    df["close_location_ma20"] = df["close_location_1d"].rolling(20).mean()
    df["ret_max_20d"] = df["ret_1d"].rolling(20).max()
    df["ret_min_20d"] = df["ret_1d"].rolling(20).min()
    df["ret_skew_20d"] = df["ret_1d"].rolling(20).skew()
    df["ret_kurt_20d"] = df["ret_1d"].rolling(20).kurt()

    volume_change = vol.pct_change().replace([np.inf, -np.inf], np.nan)
    df["price_volume_corr_20d"] = df["ret_1d"].rolling(20).corr(volume_change)
    signed_volume = np.sign(df["ret_1d"].fillna(0.0)) * vol
    df["obv_20d"] = signed_volume.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    if "amount" in df.columns:
        df["amount_trend_5_20"] = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan) - 1.0
    else:
        df["amount_trend_5_20"] = np.nan
    if "turnover" in df.columns:
        df["turnover_trend_5_20"] = turnover.rolling(5).mean() / turnover_mean.replace(0, np.nan) - 1.0
    else:
        df["turnover_trend_5_20"] = np.nan

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df[TARGET_3D_COLUMN] = close.shift(-3) / close - 1.0
    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    if stock_code is not None and "stock_code" not in df.columns:
        df["stock_code"] = stock_code
    return df


def _index_features(index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    close = idx["close"]
    idx["idx_ret_1d"] = close.pct_change(1)
    idx["idx_ret_5d"] = close.pct_change(5)
    idx["idx_ret_20d"] = close.pct_change(20)
    idx["idx_vol_20d"] = idx["idx_ret_1d"].rolling(20).std()
    idx["idx_ret_1d_mean_60"] = idx["idx_ret_1d"].rolling(60).mean()
    idx["idx_ret_1d_var_60"] = idx["idx_ret_1d"].rolling(60).var()
    idx["idx_target_3d"] = close.shift(-3) / close - 1.0
    idx["idx_target_5d"] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return idx[
        [
            "date",
            "idx_ret_1d",
            "idx_ret_5d",
            "idx_ret_20d",
            "idx_vol_20d",
            "idx_ret_1d_mean_60",
            "idx_ret_1d_var_60",
            "idx_target_3d",
            "idx_target_5d",
        ]
    ]


def _calendar_cycle_features(dates: pd.Series | pd.Index | np.ndarray) -> pd.DataFrame:
    """Known-in-advance calendar features for a five-business-day hold.

    These use only the as-of date and standard business-day calendar, not
    future prices.  They are intentionally approximate for exchange-specific
    holidays, but capture the broad weekly/monthly cycle we can know before
    submission: Monday risk-on/reopen effects, Friday de-risking, month
    start/end flow, and weekend gap length.
    """
    unique_dates = pd.Series(pd.to_datetime(pd.Index(dates).unique()), name="date").sort_values()
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date in unique_dates:
        date = pd.Timestamp(date)
        next_bdays = [date + BDay(i) for i in range(1, FORWARD_HORIZON + 1)]
        first = pd.Timestamp(next_bdays[0])
        last = pd.Timestamp(next_bdays[-1])
        days_in_month = float(date.days_in_month)
        eval_days = [pd.Timestamp(day) for day in next_bdays]
        rows.append(
            {
                "date": date,
                "asof_weekday_norm": date.weekday() / 4.0,
                "asof_is_monday": float(date.weekday() == 0),
                "asof_is_friday": float(date.weekday() == 4),
                "asof_month_pos": (date.day - 1.0) / max(days_in_month - 1.0, 1.0),
                "asof_month_start": float(date.day <= 5),
                "asof_month_end": float(date.day >= max(25, date.days_in_month - 4)),
                "eval_starts_monday": float(first.weekday() == 0),
                "eval_ends_friday": float(last.weekday() == 4),
                "eval_is_full_workweek": float(
                    first.weekday() == 0
                    and last.weekday() == 4
                    and (last - first).days == 4
                ),
                "eval_crosses_month": float(first.month != last.month),
                "eval_month_start": float(any(day.day <= 5 for day in eval_days)),
                "eval_month_end": float(any(day.day >= max(25, day.days_in_month - 4) for day in eval_days)),
                "weekend_gap_days": float(max((first - date).days - 1, 0)),
            }
        )
    return pd.DataFrame(rows)


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank of selected features (values in [0, 1])."""
    for base in [
        "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "ret_2d", "ret_15d", "ret_30d", "ret_120d",
        "mom_accel_5_20", "vol_20d", "vol_ratio_5_20",
        "trend_quality_20d", "trend_quality_60d",
        "trend_efficiency_20d", "vol_ratio_20_60",
        "downside_vol_20d", "beta_60d", "residual_ret_20d",
        "excess_ret_5d", "excess_ret_20d", "amount_z_20d",
        "turnover_z_20d", "close_pos_20d", "drawdown_20d",
        "amplitude_ma_20d",
        "ma5_over_ma20", "ma20_over_ma60", "range_vol_20d",
        "amplitude_z_20d", "amount_trend_5_20", "turnover_trend_5_20",
        "price_volume_corr_20d", "obv_20d", "close_location_ma5",
        "close_location_ma20", "gap_mean_5d", "gap_mean_20d",
        "intraday_mean_5d", "intraday_mean_20d", "overnight_vol_20d",
        "intraday_vol_20d", "ret_max_20d", "ret_min_20d",
        "ret_skew_20d", "ret_kurt_20d", "market_corr_60d",
        "weekly_risk_appetite", "weekly_friday_derisk",
        "month_start_flow", "month_end_defensive",
        "post_gap_reopen_flow", "weekly_carry_quality",
    ]:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )
    panel["breadth_ret_5d_pos"] = panel.groupby("date")["ret_5d"].transform(lambda s: (s > 0).mean())
    panel["dispersion_ret_5d"] = panel.groupby("date")["ret_5d"].transform("std")
    return panel


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target.

    Parameters
    ----------
    prices : DataFrame with columns [date, stock_code, open, close, high, low,
             volume, amount, turnover?]

    Returns
    -------
    DataFrame with FEATURE_COLUMNS and TARGET_COLUMN populated.  Rows where any
    feature is NaN (typically the first ~60 days per stock) are kept so callers
    can decide how to handle them.
    """
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    grouped = prices.groupby("stock_code", group_keys=False)
    try:
        panel = grouped.apply(_per_stock_features, include_groups=False).reset_index(drop=True)
    except TypeError:
        panel = grouped.apply(_per_stock_features).reset_index(drop=True)
    if index_df is not None:
        panel = panel.merge(_index_features(index_df), on="date", how="left")
    else:
        panel["idx_ret_1d"] = 0.0
        panel["idx_ret_5d"] = 0.0
        panel["idx_ret_20d"] = 0.0
        panel["idx_vol_20d"] = 0.0
        panel["idx_ret_1d_mean_60"] = 0.0
        panel["idx_ret_1d_var_60"] = 0.0
        panel["idx_target_3d"] = 0.0
        panel["idx_target_5d"] = 0.0
    panel = panel.sort_values(["stock_code", "date"]).copy()
    panel["_ret_idx_prod"] = panel["ret_1d"] * panel["idx_ret_1d"]
    stock_group = panel.groupby("stock_code", group_keys=False)
    ret_mean_60 = stock_group["ret_1d"].transform(lambda s: s.rolling(60).mean())
    prod_mean_60 = stock_group["_ret_idx_prod"].transform(lambda s: s.rolling(60).mean())
    cov_60 = prod_mean_60 - ret_mean_60 * panel["idx_ret_1d_mean_60"]
    panel["beta_60d"] = cov_60 / panel["idx_ret_1d_var_60"].replace(0, np.nan)
    ret_std_60 = stock_group["ret_1d"].transform(lambda s: s.rolling(60).std())
    panel["market_corr_60d"] = cov_60 / (ret_std_60 * np.sqrt(panel["idx_ret_1d_var_60"]).replace(0, np.nan))
    panel["residual_ret_20d"] = panel["ret_20d"] - panel["beta_60d"] * panel["idx_ret_20d"]
    panel["volume_price_confirm_20d"] = panel["ret_20d"] * panel["amount_z_20d"]
    panel = panel.drop(columns=["_ret_idx_prod"])
    panel["excess_ret_5d"] = panel["ret_5d"] - panel["idx_ret_5d"]
    panel["excess_ret_20d"] = panel["ret_20d"] - panel["idx_ret_20d"]
    panel[TARGET_EXCESS_3D_COLUMN] = panel[TARGET_3D_COLUMN] - panel["idx_target_3d"]
    panel[TARGET_EXCESS_COLUMN] = panel[TARGET_COLUMN] - panel["idx_target_5d"]
    panel = panel.merge(_calendar_cycle_features(panel["date"]), on="date", how="left")
    panel["weekly_risk_appetite"] = panel["eval_starts_monday"] * (
        panel["ret_3d"].fillna(0.0) + 0.5 * panel["obv_20d"].fillna(0.0) - panel["vol_ratio_5_20"].fillna(0.0)
    )
    panel["weekly_friday_derisk"] = panel["eval_ends_friday"] * (
        panel["trend_quality_20d"].fillna(0.0) - panel["vol_ratio_5_20"].fillna(0.0) - panel["overnight_vol_20d"].fillna(0.0)
    )
    panel["month_start_flow"] = panel["eval_month_start"] * (
        panel["amount_z_20d"].fillna(0.0) + panel["obv_20d"].fillna(0.0)
    )
    panel["month_end_defensive"] = panel["eval_month_end"] * (
        -panel["vol_20d"].fillna(0.0) - panel["downside_vol_20d"].fillna(0.0) + panel["trend_efficiency_20d"].fillna(0.0)
    )
    panel["post_gap_reopen_flow"] = panel["weekend_gap_days"] * (
        panel["gap_mean_20d"].fillna(0.0) + 0.5 * panel["amount_trend_5_20"].fillna(0.0)
    )
    panel["weekly_carry_quality"] = panel["eval_is_full_workweek"] * (
        panel["obv_20d"].fillna(0.0)
        + panel["trend_quality_20d"].fillna(0.0)
        + panel["close_location_ma20"].fillna(0.0)
        - panel["vol_ratio_5_20"].fillna(0.0)
    )
    panel = _cross_sectional_ranks(panel)
    return panel


def training_frame(
    panel: pd.DataFrame,
    min_date=None,
    max_date=None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Rows usable for supervised training: all features present AND target present.

    The target for date t uses close(t+5), so rows within the last 5 trading
    days of the panel are dropped automatically (target is NaN there).
    """
    df = panel.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date (defaults to the latest date)."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df
