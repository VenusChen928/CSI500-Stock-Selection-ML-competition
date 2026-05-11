"""
Validation-tuned XGBoost portfolio baseline.

The model is intentionally close to the original baseline, but the portfolio
construction is selected on historical score windows instead of being fixed to
linear rank weights.  By default the selected portfolio must use a non-flat
rank-weight curve so stronger model ranks receive larger positions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import (
    EMBARGO_DAYS,
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    MAX_WEIGHT,
    MIN_STOCKS,
    build_features,
    prediction_frame,
    train_model,
    training_frame,
)
from score_submission import score_window

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VAL_DAYS = 15
DEFAULT_LOOKBACK_DAYS = 475
DEFAULT_TOPK_GRID = (40, 45, 50, 55, 60, 65, 70, 80, 100)
DEFAULT_POWER_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
EQUAL_WEIGHT_POWER = 0.0


def time_decay_weights(
    train_df: pd.DataFrame,
    half_life: int | None,
    floor: float = 0.0,
) -> np.ndarray | None:
    """Exponential date-level sample weights with an optional old-data floor."""
    if half_life is None or half_life <= 0:
        return None
    sorted_dates = np.sort(train_df["date"].unique())
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(sorted_dates)}
    n_dates = len(sorted_dates)
    date_idx = train_df["date"].map(date_to_idx).to_numpy()
    delta = (n_dates - 1) - date_idx
    weights = np.exp(-np.log(2.0) * delta / half_life)
    if floor > 0:
        weights = np.maximum(weights, floor)
    return weights


@dataclass(frozen=True)
class PortfolioShape:
    top_k: int
    rank_power: float


def split_train_val(
    df: pd.DataFrame,
    val_days: int = DEFAULT_VAL_DAYS,
    embargo_days: int = EMBARGO_DAYS,
):
    all_dates = np.sort(df["date"].unique())
    if len(all_dates) < val_days + embargo_days + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-val_days])
    train_end = pd.Timestamp(all_dates[-(val_days + embargo_days + 1)])
    train_df = df[df["date"] <= train_end].copy()
    val_df = df[df["date"] >= val_start].copy()
    return train_df, val_df, train_end, val_start


def _apply_weight_cap(weights: pd.Series) -> pd.Series:
    w = weights[weights > 0].copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} names")
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


def build_shaped_portfolio(scores: pd.Series, shape: PortfolioShape) -> pd.Series:
    if shape.top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    chosen = scores.sort_values(ascending=False).head(shape.top_k)
    ranks = np.arange(len(chosen), 0, -1, dtype=float)
    if shape.rank_power == 0:
        raw = np.ones(len(chosen), dtype=float)
    else:
        raw = ranks ** shape.rank_power
    weights = pd.Series(raw / raw.sum(), index=chosen.index)
    return _apply_weight_cap(weights)


def validation_windows(
    val_dates: np.ndarray,
    trading_dates: np.ndarray,
    horizon: int = FORWARD_HORIZON,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(trading_dates)}
    windows = []
    for offset in range(0, len(val_dates), horizon):
        as_of = pd.Timestamp(val_dates[offset])
        idx = date_to_idx.get(as_of)
        if idx is None or idx + horizon >= len(trading_dates):
            continue
        windows.append(
            (
                as_of,
                pd.Timestamp(trading_dates[idx + 1]),
                pd.Timestamp(trading_dates[idx + horizon]),
            )
        )
    return windows


def select_shape(
    model,
    panel: pd.DataFrame,
    val_dates: np.ndarray,
    trading_dates: np.ndarray,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    topk_grid: tuple[int, ...] = DEFAULT_TOPK_GRID,
    power_grid: tuple[float, ...] = DEFAULT_POWER_GRID,
    allow_equal_weight: bool = False,
    shape_horizon: int = FORWARD_HORIZON,
    features: list[str] | None = None,
) -> tuple[PortfolioShape, pd.DataFrame]:
    features = features or FEATURE_COLUMNS
    windows = validation_windows(val_dates, trading_dates, horizon=shape_horizon)
    if not windows:
        return PortfolioShape(top_k=50, rank_power=0.5), pd.DataFrame()

    rows = []
    score_cache = {}
    for as_of, _, _ in windows:
        pred = prediction_frame(panel, as_of=as_of).copy()
        pred["score"] = model.predict(pred[features])
        score_cache[as_of] = pred.set_index("stock_code")["score"]

    candidate_powers = list(power_grid)
    if allow_equal_weight and EQUAL_WEIGHT_POWER not in candidate_powers:
        candidate_powers = [EQUAL_WEIGHT_POWER] + candidate_powers
    for top_k in topk_grid:
        for rank_power in candidate_powers:
            scores = []
            shape = PortfolioShape(top_k=top_k, rank_power=rank_power)
            for as_of, start, end in windows:
                weights = build_shaped_portfolio(score_cache[as_of], shape)
                result = score_window(weights, prices, index_df, start, end)
                scores.append(result["excess_return"])
            rows.append(
                {
                    "top_k": top_k,
                    "rank_power": rank_power,
                    "mean_excess_return": float(np.mean(scores)),
                    "sum_excess_return": float(np.sum(scores)),
                    "min_excess_return": float(np.min(scores)),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["mean_excess_return", "min_excess_return", "sum_excess_return"],
        ascending=False,
    )
    best = table.iloc[0]
    return PortfolioShape(int(best["top_k"]), float(best["rank_power"])), table


def fit_tuned_model(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    val_days: int = DEFAULT_VAL_DAYS,
    embargo_days: int = EMBARGO_DAYS,
    lookback_days: int | None = DEFAULT_LOOKBACK_DAYS,
    allow_equal_weight: bool = False,
    shape_horizon: int = FORWARD_HORIZON,
    features: list[str] | None = None,
    half_life: int | None = None,
    weight_floor: float = 0.0,
    target_column: str = "target_5d",
    target_horizon: int = FORWARD_HORIZON,
):
    features = features or FEATURE_COLUMNS
    raw_trading_dates = np.sort(prices["date"].unique())
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(raw_trading_dates[-1])
    if lookback_days is not None and lookback_days > 0:
        min_date = as_of_ts - pd.Timedelta(days=lookback_days)
        prices = prices[prices["date"] >= min_date].copy()
        index_df = index_df[index_df["date"] >= min_date].copy()
    panel = build_features(prices, index_df)
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - target_horizon)])
    train_pool = training_frame(panel, max_date=train_cutoff, target_column=target_column).dropna(subset=features)
    train_df, val_df, train_end, val_start = split_train_val(
        train_pool,
        val_days=val_days,
        embargo_days=embargo_days,
    )
    sample_weight = time_decay_weights(train_df, half_life=half_life, floor=weight_floor)
    model = train_model(
        train_df,
        val_df,
        features=features,
        sample_weight=sample_weight,
        target_column=target_column,
    )
    shape, shape_table = select_shape(
        model=model,
        panel=panel,
        val_dates=np.sort(val_df["date"].unique()),
        trading_dates=trading_dates,
        prices=prices,
        index_df=index_df,
        allow_equal_weight=allow_equal_weight,
        shape_horizon=shape_horizon,
        features=features,
    )
    return {
        "panel": panel,
        "features": features,
        "target_column": target_column,
        "trading_dates": trading_dates,
        "train_df": train_df,
        "val_df": val_df,
        "train_end": train_end,
        "val_start": val_start,
        "model": model,
        "shape": shape,
        "shape_table": shape_table,
    }


def generate_submission(fit_result: dict, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    trading_dates = fit_result["trading_dates"]
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(trading_dates[-1])
    pred = prediction_frame(fit_result["panel"], as_of=as_of_ts).copy()
    features = fit_result.get("features", FEATURE_COLUMNS)
    pred = pred.dropna(subset=features)
    pred["score"] = fit_result["model"].predict(pred[features])
    weights = build_shaped_portfolio(
        pred.set_index("stock_code")["score"],
        fit_result["shape"],
    )
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    parser.add_argument("--out", default="submission_tuned_xgboost.csv")
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--embargo-days", type=int, default=EMBARGO_DAYS)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="calendar days of recent data to train on; use 0 for full history",
    )
    parser.add_argument(
        "--allow-equal-weight",
        action="store_true",
        help="include rank_power=0 in the validation search",
    )
    parser.add_argument(
        "--shape-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Trading-day horizon used to select top_k/rank_power on validation windows.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else None

    fit_result = fit_tuned_model(
        prices,
        index_df,
        as_of=as_of,
        val_days=args.val_days,
        embargo_days=args.embargo_days,
        lookback_days=args.lookback_days if args.lookback_days > 0 else None,
        allow_equal_weight=args.allow_equal_weight,
        shape_horizon=args.shape_horizon,
    )
    submission = generate_submission(fit_result, as_of=as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    shape = fit_result["shape"]
    print(f">> train rows: {len(fit_result['train_df']):,} up to {fit_result['train_end'].date()}")
    print(f">> val rows:   {len(fit_result['val_df']):,} from {fit_result['val_start'].date()}")
    print(f">> selected shape: top_k={shape.top_k}, rank_power={shape.rank_power:.2f}")
    if not fit_result["shape_table"].empty:
        print(">> validation shape table")
        print(fit_result["shape_table"].head(15).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
