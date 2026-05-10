"""Weekly-alpha overlay for the stage2 complete-week task.

The old stage2 winner was tuned on rolling five-trading-day windows that often
crossed weekends/holidays.  This script rebases the portfolio layer to the
official stage2 shape: Friday as-of, then the next complete Monday-Friday
evaluation week.

Design:
1. Start from the existing meta portfolio, which is still the safest base.
2. Build as-of-only weekly alpha scores from low short-volatility, flow
   confirmation, and short momentum features.
3. Re-rank the blended base/alpha confidence into a non-equal-weight top-k
   portfolio capped at the competition 10% maximum.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MIN_STOCKS = 30
COMPETITION_MAX_WEIGHT = 0.10
DEFAULT_META_CACHE = (
    ROOT
    / "stage2_report"
    / "backtests"
    / "full_week"
    / "stage2_fullweek_meta_12w_20260510"
)


FORMULAS: dict[str, list[tuple[str, str, float]]] = {
    # Best complete-week validation mean in the focused overlay scan.
    "vol_obv_amt": [
        ("vol_ratio_5_20", "low", 0.45),
        ("obv_20d", "high", 0.35),
        ("amount_z_20d", "high", 0.20),
    ],
    # More conservative floor: low overnight risk plus flow confirmation.
    "stable_flow": [
        ("vol_ratio_5_20", "low", 0.35),
        ("overnight_vol_20d", "low", 0.25),
        ("obv_20d", "high", 0.25),
        ("amount_z_20d", "high", 0.15),
    ],
    # Current 2026-05-08 regime is a strong post-holiday tape.  This route
    # allows more short-momentum continuation while retaining a volatility gate.
    "weekly_momo_conf": [
        ("ret_1d", "high", 0.35),
        ("ret_3d", "high", 0.20),
        ("vol_ratio_5_20", "low", 0.25),
        ("obv_20d", "high", 0.20),
    ],
    "gap_flow": [
        ("gap_mean_20d", "high", 0.35),
        ("obv_20d", "high", 0.25),
        ("vol_ratio_5_20", "low", 0.25),
        ("ret_1d", "high", 0.15),
    ],
}


MODE_CONFIGS: dict[str, dict] = {
    "stable": {
        "formula": "vol_obv_amt",
        "alpha_top_k": 50,
        "alpha_power": 4.0,
        "confidence_penalty": 0.10,
        "base_weight": 0.60,
        "final_top_k": 35,
        "final_power": 8.0,
        "max_weight": COMPETITION_MAX_WEIGHT,
    },
    "floor": {
        "formula": "gap_flow",
        "alpha_top_k": 30,
        "alpha_power": 6.0,
        "confidence_penalty": 0.10,
        "base_weight": 0.70,
        "final_top_k": 50,
        "final_power": 4.0,
        "max_weight": COMPETITION_MAX_WEIGHT,
    },
    "current_regime": {
        "formula": "weekly_momo_conf",
        "alpha_top_k": 50,
        "alpha_power": 4.0,
        "confidence_penalty": 0.20,
        "base_weight": 0.50,
        "final_top_k": 35,
        "final_power": 4.0,
        "max_weight": COMPETITION_MAX_WEIGHT,
    },
}


def normalize_weights(weights: pd.Series) -> pd.Series:
    weights = weights[weights > 0].astype(float).copy()
    if len(weights) < MIN_STOCKS:
        raise ValueError(f"portfolio must contain at least {MIN_STOCKS} stocks")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weights.index = weights.index.astype(str).str.zfill(6)
    return weights / total


def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = normalize_weights(weights)
    for _ in range(100):
        over = weights > max_weight
        if not over.any():
            break
        excess = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        free = ~over
        if not free.any():
            break
        weights[free] += excess * weights[free] / weights[free].sum()
    return weights / weights.sum()


def ranked_weights(codes: list[str], power: float, max_weight: float) -> pd.Series:
    if len(codes) < MIN_STOCKS:
        raise ValueError(f"need at least {MIN_STOCKS} selected stocks")
    raw = pd.Series(np.arange(len(codes), 0, -1, dtype=float) ** power, index=pd.Index(codes, name="stock_code"))
    return cap_weights(raw, max_weight=max_weight)


def read_submission(path: Path) -> pd.Series:
    sub = pd.read_csv(path, dtype={"stock_code": str})
    if "stock_code" not in sub.columns or "weight" not in sub.columns:
        raise ValueError(f"{path} is not a stock_code/weight submission")
    sub["stock_code"] = sub["stock_code"].astype(str).str.zfill(6)
    return normalize_weights(sub.groupby("stock_code")["weight"].sum())


def load_or_generate_base(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    meta_cache_dir: Path | None,
    base_csv: Path | None,
) -> tuple[pd.Series, str]:
    if base_csv is not None:
        return read_submission(base_csv), f"csv:{base_csv}"

    stamp = as_of.strftime("%Y%m%d")
    if meta_cache_dir is not None:
        cached = meta_cache_dir / f"meta_portfolio_ensemble_{stamp}.csv"
        if cached.exists():
            return read_submission(cached), f"cache:{cached}"

    fit_prices = prices[prices["date"] <= as_of].copy()
    fit_index = index_df[index_df["date"] <= as_of].copy()
    from stage2_meta_portfolio_ensemble import generate_meta_ensemble

    sub, _ = generate_meta_ensemble(fit_prices, fit_index, as_of)
    return normalize_weights(sub.set_index(sub["stock_code"].astype(str).str.zfill(6))["weight"]), "live:meta_portfolio_ensemble"


def signed_rank(frame: pd.DataFrame, column: str, direction: str) -> pd.Series:
    ranks = frame[column].astype(float).rank(method="average", pct=True)
    if direction == "high":
        return ranks
    if direction == "low":
        return 1.0 - ranks
    raise ValueError(f"unknown direction {direction}")


def alpha_weights(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    formula: str,
    top_k: int,
    power: float,
    max_weight: float,
    confidence_penalty: float,
) -> tuple[pd.Series, pd.DataFrame]:
    if formula not in FORMULAS:
        raise ValueError(f"unknown formula {formula}; valid={sorted(FORMULAS)}")

    frame = panel[panel["date"] == as_of].copy()
    if frame.empty:
        raise ValueError(f"no feature rows for as_of={as_of.date()}")

    score = pd.Series(0.0, index=frame.index)
    used: list[str] = []
    total = 0.0
    for column, direction, weight in FORMULAS[formula]:
        if column not in frame.columns:
            continue
        part = signed_rank(frame, column, direction)
        score = score + weight * part
        total += abs(weight)
        used.append(f"{column}:{direction}:{weight:g}")
    if total <= 0:
        raise ValueError(f"formula {formula} has no available features")
    score = score / total

    if confidence_penalty:
        if "vol_20d" in frame.columns:
            score = score - confidence_penalty * frame["vol_20d"].rank(method="average", pct=True)
        if "overnight_vol_20d" in frame.columns:
            score = score - 0.5 * confidence_penalty * frame["overnight_vol_20d"].rank(method="average", pct=True)
        if "drawdown_20d" in frame.columns:
            score = score + 0.2 * confidence_penalty * frame["drawdown_20d"].rank(method="average", pct=True)
        if "amount_z_20d" in frame.columns:
            score = score + 0.1 * confidence_penalty * frame["amount_z_20d"].rank(method="average", pct=True)

    frame["_weekly_alpha_score"] = score.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["_weekly_alpha_score"]).sort_values("_weekly_alpha_score", ascending=False)
    selected = frame.head(max(MIN_STOCKS, int(top_k)))["stock_code"].astype(str).str.zfill(6).tolist()
    weights = ranked_weights(selected, power=power, max_weight=max_weight)
    diagnostics = frame.head(10)[["stock_code", "_weekly_alpha_score"]].copy()
    diagnostics["formula_terms"] = " | ".join(used)
    return weights, diagnostics


def market_regime(panel: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    idx = index_df[index_df["date"] <= as_of].sort_values("date").copy()
    close = idx["close"]
    cur = panel[panel["date"] == as_of].copy()
    return {
        "idx_ret_3d": float(close.iloc[-1] / close.iloc[-4] - 1.0) if len(close) >= 4 else np.nan,
        "idx_ret_5d": float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else np.nan,
        "idx_ret_20d": float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else np.nan,
        "breadth_ret_3d_pos": float((cur["ret_3d"] > 0).mean()) if "ret_3d" in cur.columns and not cur.empty else np.nan,
        "breadth_ret_5d_pos": float((cur["ret_5d"] > 0).mean()) if "ret_5d" in cur.columns and not cur.empty else np.nan,
    }


def resolve_mode(requested: str, regime: dict[str, float]) -> str:
    if requested != "auto":
        return requested
    if (
        regime.get("idx_ret_20d", 0.0) >= 0.10
        and regime.get("idx_ret_5d", 0.0) >= 0.04
        and regime.get("breadth_ret_3d_pos", 0.0) >= 0.65
    ):
        return "current_regime"
    return "stable"


def generate_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    mode: str = "auto",
    meta_cache_dir: Path | None = None,
    base_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    fit_prices = prices[prices["date"] <= as_of].copy()
    fit_index = index_df[index_df["date"] <= as_of].copy()
    panel = build_features(fit_prices, fit_index)
    regime = market_regime(panel, fit_index, as_of)
    effective_mode = resolve_mode(mode, regime)
    config = MODE_CONFIGS[effective_mode]

    base, base_source = load_or_generate_base(
        prices=prices,
        index_df=index_df,
        as_of=as_of,
        meta_cache_dir=meta_cache_dir,
        base_csv=base_csv,
    )
    alpha, diagnostics = alpha_weights(
        panel=panel,
        as_of=as_of,
        formula=config["formula"],
        top_k=config["alpha_top_k"],
        power=config["alpha_power"],
        max_weight=config["max_weight"],
        confidence_penalty=config["confidence_penalty"],
    )

    blended = base.mul(config["base_weight"]).add(
        alpha.mul(1.0 - config["base_weight"]),
        fill_value=0.0,
    )
    ranked_codes = blended.sort_values(ascending=False).head(config["final_top_k"]).index.astype(str).str.zfill(6).tolist()
    final_weights = ranked_weights(ranked_codes, power=config["final_power"], max_weight=config["max_weight"])
    sub = pd.DataFrame({"stock_code": final_weights.index, "weight": final_weights.values})
    meta = pd.DataFrame(
        [
            {
                "as_of": as_of.date().isoformat(),
                "requested_mode": mode,
                "effective_mode": effective_mode,
                "base_source": base_source,
                "formula": config["formula"],
                "alpha_top_k": config["alpha_top_k"],
                "alpha_power": config["alpha_power"],
                "confidence_penalty": config["confidence_penalty"],
                "base_weight": config["base_weight"],
                "final_top_k": config["final_top_k"],
                "final_power": config["final_power"],
                "max_weight": config["max_weight"],
                "n_names": len(sub),
                "max_actual_weight": float(sub["weight"].max()),
                **regime,
            }
        ]
    )
    return sub, meta, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--mode", choices=["auto", "stable", "floor", "current_regime"], default="auto")
    parser.add_argument(
        "--meta-cache-dir",
        default=None,
        help="Optional directory containing meta_portfolio_ensemble_YYYYMMDD.csv. Defaults to live no-cache generation.",
    )
    parser.add_argument("--base-csv", default=None)
    parser.add_argument("--out", default="stage2_report/route_outputs/stage2_weekly_alpha_overlay.csv")
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

    meta_cache_dir = Path(args.meta_cache_dir) if args.meta_cache_dir else None
    base_csv = Path(args.base_csv) if args.base_csv else None
    sub, meta, diagnostics = generate_submission(
        prices=prices,
        index_df=index_df,
        as_of=as_of,
        mode=args.mode,
        meta_cache_dir=meta_cache_dir,
        base_csv=base_csv,
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

    print(f">> model=stage2_weekly_alpha_overlay as_of={as_of.date()} mode={meta.loc[0, 'effective_mode']}")
    print(meta.to_string(index=False))
    print(f">> wrote {len(sub)} names to {out_path}")
    print(f"   weight summary: min={sub['weight'].min():.4f} max={sub['weight'].max():.4f} sum={sub['weight'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
