"""As-of-safe dynamic factor portfolio for stage2.

This route is deliberately model-light.  For each prediction date it looks only
at factor portfolios whose five-day outcomes are already known by `as_of`,
selects the most stable recent factor/shape, and applies that factor to the
current cross-section.  The goal is to add a different source of alpha to the
tree-heavy meta ensemble without introducing test-window leakage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import FORWARD_HORIZON, MIN_STOCKS
from features import (
    ALPHA_FEATURE_COLUMNS,
    CORE_FEATURE_COLUMNS,
    EXPERIMENTAL_FEATURE_COLUMNS,
    QUALITY_FEATURE_COLUMNS,
    build_features,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAX_WEIGHT = 0.10

HIGH_VALUE_FACTOR_POOL = [
    "overnight_ret",
    "intraday_mean_5d",
    "gap_mean_5d",
    "obv_20d",
    "turnover_ma_20d",
    "turnover_trend_5_20",
    "amount_trend_5_20",
    "amount_z_20d",
    "downside_vol_20d",
    "drawdown_20d",
    "close_pos_20d",
    "close_location_ma5",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "residual_ret_20d",
    "trend_quality_20d",
    "trend_efficiency_20d",
    "market_corr_60d",
    "price_volume_corr_20d",
]


@dataclass(frozen=True)
class FactorShape:
    factor: str
    direction: str
    top_k: int
    rank_power: float
    max_weight: float


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


def candidate_features(panel: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in HIGH_VALUE_FACTOR_POOL + CORE_FEATURE_COLUMNS + QUALITY_FEATURE_COLUMNS + ALPHA_FEATURE_COLUMNS + EXPERIMENTAL_FEATURE_COLUMNS:
        if name in seen or name not in panel.columns:
            continue
        if name.startswith("idx_") or name.startswith("target"):
            continue
        if name in {"breadth_ret_5d_pos", "dispersion_ret_5d"}:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def factor_portfolio(panel: pd.DataFrame, as_of: pd.Timestamp, shape: FactorShape) -> pd.Series:
    today = panel[panel["date"] == as_of][["stock_code", shape.factor]].dropna().copy()
    if today.empty:
        raise ValueError(f"no factor rows for {shape.factor} at {as_of.date()}")
    today["stock_code"] = today["stock_code"].astype(str).str.zfill(6)
    ascending = shape.direction == "low"
    chosen = today.sort_values(shape.factor, ascending=ascending).head(shape.top_k)
    if len(chosen) < MIN_STOCKS:
        raise ValueError(f"too few selected names for {shape.factor}")
    ranks = np.arange(len(chosen), 0, -1, dtype=float) ** shape.rank_power
    raw = pd.Series(ranks, index=chosen["stock_code"].to_numpy())
    return cap_weights(raw, shape.max_weight)


def realized_return_cache(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    trading_dates: np.ndarray,
    as_of_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, tuple[pd.Series, float]]:
    px = prices.copy()
    px["stock_code"] = px["stock_code"].astype(str).str.zfill(6)
    close = px.pivot(index="date", columns="stock_code", values="close").sort_index()
    idx = index_df.sort_values("date").set_index("date")
    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    out: dict[pd.Timestamp, tuple[pd.Series, float]] = {}
    for hist_as_of in as_of_dates:
        pos = date_to_pos.get(hist_as_of)
        if pos is None or pos + FORWARD_HORIZON >= len(trading_dates):
            continue
        end = pd.Timestamp(trading_dates[pos + FORWARD_HORIZON])
        if hist_as_of not in close.index or end not in close.index:
            continue
        entry = close.loc[hist_as_of].replace(0, np.nan)
        exit_ = close.loc[end]
        stock_ret = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if hist_as_of not in idx.index or end not in idx.index:
            continue
        bench = float(idx.loc[end, "close"] / idx.loc[hist_as_of, "close"] - 1.0)
        out[hist_as_of] = (stock_ret, bench)
    return out


def recent_realized_asofs(trading_dates: np.ndarray, as_of: pd.Timestamp, trailing_windows: int, step: int) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    as_of_pos = dates.index(pd.Timestamp(as_of))
    last_hist_pos = as_of_pos - FORWARD_HORIZON
    positions = list(range(max(120, last_hist_pos - trailing_windows * step + 1), last_hist_pos + 1, step))
    return [dates[p] for p in positions if p + FORWARD_HORIZON <= as_of_pos]


def evaluate_shape(
    panel: pd.DataFrame,
    shape: FactorShape,
    return_cache: dict[pd.Timestamp, tuple[pd.Series, float]],
) -> dict:
    values: list[float] = []
    for hist_as_of, (stock_ret, bench) in return_cache.items():
        try:
            weights = factor_portfolio(panel, hist_as_of, shape)
        except Exception:
            continue
        values.append(float((weights * stock_ret.reindex(weights.index).fillna(0.0)).sum() - bench))
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    neg = int((arr < 0).sum())
    return {
        "factor": shape.factor,
        "direction": shape.direction,
        "top_k": shape.top_k,
        "rank_power": shape.rank_power,
        "max_weight": shape.max_weight,
        "mean_excess": float(arr.mean()),
        "median_excess": float(np.median(arr)),
        "min_excess": float(arr.min()),
        "std_excess": float(arr.std()),
        "negative_windows": neg,
        "count": int(len(arr)),
        "utility": float(arr.mean() + 0.45 * arr.min() - 0.20 * arr.std() - 0.004 * neg),
    }


def select_dynamic_factor(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    trailing_windows: int = 24,
    step: int = 5,
    max_candidates: int = 12,
) -> tuple[FactorShape, pd.DataFrame]:
    trading_dates = np.sort(panel["date"].unique())
    hist_asofs = recent_realized_asofs(trading_dates, as_of, trailing_windows=trailing_windows, step=step)
    cache = realized_return_cache(prices, index_df, trading_dates, hist_asofs)
    if len(cache) < 8:
        raise RuntimeError(f"not enough realized historical windows before {as_of.date()}: {len(cache)}")

    # Pre-screen with daily rank IC on already-realized history.  This reduces
    # correlated feature hunting before the portfolio-shape sweep.
    rows = []
    feature_list = candidate_features(panel)
    for factor in feature_list:
        ics = []
        for hist_as_of, (stock_ret, _) in cache.items():
            today = panel[panel["date"] == hist_as_of][["stock_code", factor]].dropna().copy()
            if len(today) < 50 or today[factor].nunique() < 5:
                continue
            today["stock_code"] = today["stock_code"].astype(str).str.zfill(6)
            aligned_ret = stock_ret.reindex(today["stock_code"]).to_numpy()
            rho = pd.Series(today[factor].to_numpy()).corr(pd.Series(aligned_ret), method="spearman")
            if pd.notna(rho):
                ics.append(float(rho))
        if not ics:
            continue
        mean_ic = float(np.mean(ics))
        rows.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "abs_mean_ic": abs(mean_ic),
                "icir": abs(mean_ic) / (float(np.std(ics)) + 1e-6),
                "ic_pos_rate": float(np.mean(np.asarray(ics) > 0)),
                "ic_count": len(ics),
            }
        )
    ic_table = pd.DataFrame(rows).sort_values(["abs_mean_ic", "icir"], ascending=False)
    screened = ic_table.head(max_candidates)["factor"].tolist()
    if not screened:
        screened = CORE_FEATURE_COLUMNS.copy()

    results = []
    for factor in screened:
        for direction in ("high", "low"):
            for top_k in (30, 40, 50):
                for rank_power in (1.0, 2.0, 4.0, 8.0):
                    shape = FactorShape(factor, direction, top_k, rank_power, MAX_WEIGHT)
                    row = evaluate_shape(panel, shape, cache)
                    if row:
                        results.append(row)
    table = pd.DataFrame(results)
    if table.empty:
        raise RuntimeError("dynamic factor sweep produced no candidates")
    table = table.merge(ic_table, on="factor", how="left")
    table = table.sort_values(["utility", "mean_excess", "min_excess"], ascending=False)
    best = table.iloc[0]
    return (
        FactorShape(
            factor=str(best["factor"]),
            direction=str(best["direction"]),
            top_k=int(best["top_k"]),
            rank_power=float(best["rank_power"]),
            max_weight=float(best["max_weight"]),
        ),
        table,
    )


def generate_dynamic_factor_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    trailing_windows: int = 36,
    step: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    prices = prices[prices["date"] <= as_of].copy()
    index_df = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(prices, index_df)
    shape, table = select_dynamic_factor(
        panel,
        prices,
        index_df,
        as_of,
        trailing_windows=trailing_windows,
        step=step,
    )
    weights = factor_portfolio(panel, as_of, shape)
    sub = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    best = table.iloc[0].to_dict()
    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "final_route": "dynamic_factor_portfolio",
                "trailing_windows": trailing_windows,
                "step": step,
                **shape.__dict__,
                "selection_mean_excess": best.get("mean_excess", np.nan),
                "selection_min_excess": best.get("min_excess", np.nan),
                "selection_negative_windows": best.get("negative_windows", np.nan),
                "selection_count": best.get("count", np.nan),
                "mean_ic": best.get("mean_ic", np.nan),
                "icir": best.get("icir", np.nan),
                "n_names": len(sub),
                "max_actual_weight": float(sub["weight"].max()),
            }
        ]
    )
    return sub, meta, table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--trailing-windows", type=int, default=24)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_dynamic_factor_portfolio.csv")
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--selection-report-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    sub, meta, table = generate_dynamic_factor_submission(
        prices,
        index_df,
        as_of,
        trailing_windows=args.trailing_windows,
        step=args.step,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(meta_path, index=False)
    if args.selection_report_out:
        report_path = Path(args.selection_report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(report_path, index=False)
    print(f">> model=stage2_dynamic_factor_portfolio as_of={as_of.date()}")
    print(meta.to_string(index=False))
    print(f">> selection top")
    print(table.head(8).to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
