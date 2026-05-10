"""Stage2 validation IC diagnostics.

The competition is scored by realized portfolio excess return, but validation
IC is useful as an overfit check: it asks whether a score/factor has stable
cross-sectional ranking power before we trust it for stock selection or
confidence weighting.

This script reports two separate views:
* realized_asof IC: diagnostic only, computed against the known future window;
* trailing validation IC: production-safe, using only dates whose 5-day targets
  would have been known by the as-of date.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON
from features import TARGET_COLUMN, TARGET_EXCESS_COLUMN, build_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_FACTORS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "mom_accel_5_20",
    "vol_20d",
    "vol_ratio_5_20",
    "downside_vol_20d",
    "turnover_ma_20d",
    "amount_z_20d",
    "intraday_ret",
    "overnight_ret",
    "close_over_ma20",
    "close_over_ma60",
    "close_pos_20d",
    "drawdown_20d",
    "trend_quality_20d",
    "trend_efficiency_20d",
    "residual_ret_20d",
]
ALPHA_FIELDS = [
    ("factor_route", "factor_route_factor", "factor_route_direction"),
    ("regime_alpha", "regime_alpha_factor", "regime_alpha_direction"),
    ("liquidity_alpha", "liquidity_alpha_factor", "liquidity_alpha_direction"),
    ("tree_secondary_alpha", "tree_secondary_alpha_factor", "tree_secondary_alpha_direction"),
    ("final_confidence_alpha", "final_confidence_alpha_factor", "final_confidence_alpha_direction"),
]


def rolling_windows(
    trading_dates: np.ndarray,
    windows: int,
    step: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    out = []
    max_asof_idx = len(trading_dates) - FORWARD_HORIZON - 1
    idx = max_asof_idx - step * (windows - 1)
    while idx <= max_asof_idx:
        if idx >= 120:
            as_of = pd.Timestamp(trading_dates[idx])
            out.append((as_of, pd.Timestamp(trading_dates[idx + 1]), pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])))
        idx += step
    return out


def explicit_windows(
    trading_dates: np.ndarray,
    as_of_values: list[str],
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    out = []
    for value in as_of_values:
        as_of = pd.to_datetime(value, format="%Y%m%d")
        if as_of not in dates:
            raise ValueError(f"as_of {value} is not in trading dates")
        idx = dates.index(as_of)
        if idx + FORWARD_HORIZON >= len(dates):
            raise ValueError(f"as_of {value} does not have {FORWARD_HORIZON} future trading days")
        out.append((as_of, dates[idx + 1], dates[idx + FORWARD_HORIZON]))
    return out


def direction_score(values: pd.Series, direction: str) -> pd.Series:
    if direction == "high":
        return values.astype(float)
    if direction == "low":
        return -values.astype(float)
    raise ValueError(f"unknown direction: {direction}")


def rank_ic(frame: pd.DataFrame, factor: str, direction: str, target: str, min_names: int) -> float:
    tmp = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < min_names:
        return np.nan
    score = direction_score(tmp[factor], direction)
    if score.nunique(dropna=True) < 5 or tmp[target].nunique(dropna=True) < 5:
        return np.nan
    return float(score.rank(method="average").corr(tmp[target].rank(method="average")))


def top_tail_stats(
    frame: pd.DataFrame,
    factor: str,
    direction: str,
    target: str,
    top_k: int,
    min_names: int,
) -> dict[str, float]:
    tmp = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(tmp) < max(min_names, top_k):
        return {
            f"top{top_k}_{target}_mean": np.nan,
            f"universe_{target}_mean": np.nan,
            f"top{top_k}_{target}_spread": np.nan,
        }
    tmp["_score"] = direction_score(tmp[factor], direction)
    top = tmp.nlargest(top_k, "_score")
    universe_mean = float(tmp[target].mean())
    top_mean = float(top[target].mean())
    return {
        f"top{top_k}_{target}_mean": top_mean,
        f"universe_{target}_mean": universe_mean,
        f"top{top_k}_{target}_spread": top_mean - universe_mean,
    }


def trailing_dates(
    trading_dates: list[pd.Timestamp],
    as_of: pd.Timestamp,
    lookback_days: int,
) -> list[pd.Timestamp]:
    asof_idx = trading_dates.index(pd.Timestamp(as_of))
    known_target_idx = asof_idx - FORWARD_HORIZON
    if known_target_idx < 0:
        return []
    eligible = trading_dates[: known_target_idx + 1]
    return eligible[-lookback_days:]


def trailing_ic_stats(
    panel_by_date: dict[pd.Timestamp, pd.DataFrame],
    dates: list[pd.Timestamp],
    factor: str,
    direction: str,
    target: str,
    top_k: int,
    min_names: int,
) -> dict[str, float]:
    ics = []
    spreads = []
    for date in dates:
        day = panel_by_date.get(pd.Timestamp(date))
        if day is None:
            continue
        ic = rank_ic(day, factor, direction, target, min_names)
        if not np.isnan(ic):
            ics.append(ic)
        spread = top_tail_stats(day, factor, direction, target, top_k, min_names)[f"top{top_k}_{target}_spread"]
        if not np.isnan(spread):
            spreads.append(spread)

    if not ics:
        return {
            f"trailing_{target}_ic_mean": np.nan,
            f"trailing_{target}_ic_std": np.nan,
            f"trailing_{target}_icir": np.nan,
            f"trailing_{target}_ic_pos_rate": np.nan,
            f"trailing_{target}_days": 0,
            f"trailing_top{top_k}_{target}_spread_mean": np.nan,
        }
    ic_arr = np.asarray(ics, dtype=float)
    spread_arr = np.asarray(spreads, dtype=float)
    std = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else np.nan
    return {
        f"trailing_{target}_ic_mean": float(ic_arr.mean()),
        f"trailing_{target}_ic_std": std,
        f"trailing_{target}_icir": float(ic_arr.mean() / std) if std and not np.isnan(std) else np.nan,
        f"trailing_{target}_ic_pos_rate": float((ic_arr > 0).mean()),
        f"trailing_{target}_days": int(len(ic_arr)),
        f"trailing_top{top_k}_{target}_spread_mean": float(spread_arr.mean()) if len(spread_arr) else np.nan,
    }


def clean_meta_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def load_active_layers(meta_dir: Path | None, prefix: str, as_of: pd.Timestamp) -> tuple[str, list[tuple[str, str, str]]]:
    if meta_dir is None:
        return "", []
    meta_path = meta_dir / f"{prefix}_meta_{as_of.strftime('%Y%m%d')}.csv"
    if not meta_path.exists():
        return "", []
    meta = pd.read_csv(meta_path)
    if meta.empty:
        return "", []
    row = meta.iloc[0]
    final_route = clean_meta_value(row.get("final_route", ""))
    active = []
    for layer, factor_col, direction_col in ALPHA_FIELDS:
        factor = clean_meta_value(row.get(factor_col, ""))
        direction = clean_meta_value(row.get(direction_col, ""))
        if factor and direction:
            active.append((layer, factor, direction))
    return final_route, active


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--as-of", nargs="*", default=None)
    parser.add_argument("--factors", nargs="*", default=DEFAULT_FACTORS)
    parser.add_argument("--directions", nargs="*", choices=["high", "low"], default=["high", "low"])
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-names", type=int, default=80)
    parser.add_argument("--meta-dir", default="")
    parser.add_argument("--meta-prefix", default="hybrid_gate")
    parser.add_argument("--out", default="submissions/stage2/reports/stage2_ic_diagnostics_20260510.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    panel = build_features(prices, index_df)
    panel["date"] = pd.to_datetime(panel["date"])
    panel_by_date = {pd.Timestamp(date): day.copy() for date, day in panel.groupby("date", sort=False)}
    trading_dates = [pd.Timestamp(d) for d in np.sort(prices["date"].unique())]
    windows = explicit_windows(np.asarray(trading_dates), args.as_of) if args.as_of else rolling_windows(np.asarray(trading_dates), args.windows, args.step)
    meta_dir = Path(args.meta_dir) if args.meta_dir else None

    rows = []
    for as_of, start, end in windows:
        final_route, active_layers = load_active_layers(meta_dir, args.meta_prefix, as_of)
        active_lookup = {(factor, direction): layer for layer, factor, direction in active_layers}
        today = panel_by_date.get(as_of, pd.DataFrame())
        known_dates = trailing_dates(trading_dates, as_of, args.lookback_days)
        for factor in args.factors:
            if factor not in panel.columns:
                continue
            for direction in args.directions:
                row = {
                    "as_of": as_of.date().isoformat(),
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "final_route": final_route,
                    "factor": factor,
                    "direction": direction,
                    "active_layer": active_lookup.get((factor, direction), ""),
                }
                for target in [TARGET_EXCESS_COLUMN, TARGET_COLUMN]:
                    row[f"realized_{target}_rank_ic"] = rank_ic(today, factor, direction, target, args.min_names)
                    row.update(top_tail_stats(today, factor, direction, target, args.top_k, args.min_names))
                    row.update(trailing_ic_stats(panel_by_date, known_dates, factor, direction, target, args.top_k, args.min_names))
                rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    active = out[out["active_layer"] != ""].copy()
    active_out = out_path.with_name(out_path.stem + "_active.csv")
    active.to_csv(active_out, index=False)

    print(f">> wrote {out_path}")
    print(f">> wrote {active_out}")
    if not active.empty:
        cols = [
            "as_of",
            "active_layer",
            "factor",
            "direction",
            "realized_target_excess_5d_rank_ic",
            "trailing_target_excess_5d_ic_mean",
            "trailing_target_excess_5d_icir",
            "trailing_target_excess_5d_ic_pos_rate",
            f"top{args.top_k}_target_excess_5d_spread",
            f"trailing_top{args.top_k}_target_excess_5d_spread_mean",
        ]
        print(active[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
