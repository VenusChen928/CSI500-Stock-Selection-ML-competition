"""
Stage1 consensus portfolio for the 3-trading-day evaluation window.

The model combines two low-variance tree rankers:

  1. Tuned XGBoost with 3-day portfolio-shape validation.
  2. LightGBM with the same 3-day shape validation.

Stock selection prioritizes names both models agree on. If the intersection has
fewer than the competition minimum, it tops up by the combined model weight. The
default alpha keeps LightGBM weights on the consensus set because recent
stage1-style backtests favored that variant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = REPORT_DIR.parent
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from features import FORWARD_HORIZON
from lightgbm_portfolio import fit_lightgbm_model, generate_submission as generate_lgb_submission
from tuned_xgboost_portfolio import (
    _apply_weight_cap,
    fit_tuned_model,
    generate_submission as generate_xgb_submission,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STAGE1_HORIZON = 3
DEFAULT_ALPHA_TUNED_XGB = 0.0


def consensus_weights(
    xgb_weights: pd.Series,
    lgb_weights: pd.Series,
    alpha_tuned_xgb: float = DEFAULT_ALPHA_TUNED_XGB,
) -> pd.Series:
    """Build a capped portfolio from tree-model consensus names.

    `alpha_tuned_xgb=0` means use LightGBM weights after consensus selection;
    `alpha_tuned_xgb=1` means use tuned-XGB weights after consensus selection.
    """
    xgb_weights = xgb_weights[xgb_weights > 0].copy()
    lgb_weights = lgb_weights[lgb_weights > 0].copy()
    all_codes = sorted(set(xgb_weights.index) | set(lgb_weights.index))
    blended = (
        alpha_tuned_xgb * xgb_weights.reindex(all_codes).fillna(0.0)
        + (1.0 - alpha_tuned_xgb) * lgb_weights.reindex(all_codes).fillna(0.0)
    )

    intersection = set(xgb_weights.index) & set(lgb_weights.index)
    xgb_rank = xgb_weights.rank(ascending=False, method="first")
    lgb_rank = lgb_weights.rank(ascending=False, method="first")
    rank_score = pd.Series(
        {
            code: xgb_rank.get(code, len(xgb_weights) + 1)
            + lgb_rank.get(code, len(lgb_weights) + 1)
            for code in all_codes
        }
    ).sort_values()

    chosen = [code for code in rank_score.index if code in intersection and blended.get(code, 0.0) > 0]
    for code in blended.sort_values(ascending=False).index:
        if len([c for c in chosen if blended.get(c, 0.0) > 0]) >= 30:
            break
        if code not in chosen and blended[code] > 0:
            chosen.append(code)

    return _apply_weight_cap(blended.reindex(chosen).fillna(0.0))


def generate_consensus_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    alpha_tuned_xgb: float = DEFAULT_ALPHA_TUNED_XGB,
    shape_horizon: int = DEFAULT_STAGE1_HORIZON,
) -> pd.DataFrame:
    xgb_fit = fit_tuned_model(
        prices,
        index_df,
        as_of=as_of,
        shape_horizon=shape_horizon,
    )
    lgb_fit = fit_lightgbm_model(
        prices,
        index_df,
        as_of=as_of,
        shape_horizon=shape_horizon,
    )
    xgb = generate_xgb_submission(xgb_fit, as_of=as_of).set_index("stock_code")["weight"]
    lgb = generate_lgb_submission(lgb_fit, as_of=as_of).set_index("stock_code")["weight"]
    weights = consensus_weights(xgb, lgb, alpha_tuned_xgb=alpha_tuned_xgb)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    parser.add_argument("--out", default=str(REPORT_DIR / "generated" / "stage1_consensus_portfolio.csv"))
    parser.add_argument("--alpha-tuned-xgb", type=float, default=DEFAULT_ALPHA_TUNED_XGB)
    parser.add_argument("--shape-horizon", type=int, default=DEFAULT_STAGE1_HORIZON)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])

    submission = generate_consensus_submission(
        prices,
        index_df,
        as_of=as_of,
        alpha_tuned_xgb=args.alpha_tuned_xgb,
        shape_horizon=args.shape_horizon,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(
        f">> model=stage1_consensus as_of={as_of.date()} "
        f"alpha_tuned_xgb={args.alpha_tuned_xgb:.2f} shape_horizon={args.shape_horizon}"
    )
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
