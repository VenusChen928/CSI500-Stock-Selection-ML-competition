"""Stage1 guarded ensemble.

This route keeps the high-upside risk-balanced LSTM portfolio in normal short
window regimes, but falls back to the previous LightGBM-led consensus portfolio
when the market looks like a short-term rally that has started to pull back.

That regime was the main failure mode in recent 3-trading-day backtests: the
LSTM's concentrated picks often still rose, but underperformed the broad
benchmark during broad-market continuation windows.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = REPORT_DIR.parent
TREE_HELPERS = PROJECT_ROOT / "stage2_report" / "scripts"
for path in (SCRIPT_DIR, TREE_HELPERS, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage1_lstm_lgb_confidence import (  # noqa: E402
    AGGRESSIVE_LSTM_POLICY,
    fit_hybrid,
    generate_submission as generate_lstm_submission,
)
from stage1_consensus_portfolio import generate_consensus_submission  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class RegimeDecision:
    alpha_lstm: float
    reason: str
    idx_ret_1d: float
    idx_ret_3d: float
    idx_ret_5d: float
    idx_ret_10d: float
    idx_vol_10d: float


def _index_regime(index_df: pd.DataFrame, as_of: pd.Timestamp) -> tuple[float, float, float, float, float]:
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").set_index("date")
    if as_of not in idx.index:
        raise ValueError(f"as_of {as_of.date()} is not in index data")
    pos = idx.index.get_loc(as_of)
    close = idx["close"].astype(float)

    def ret(days: int) -> float:
        if pos < days:
            return 0.0
        return float(close.iloc[pos] / close.iloc[pos - days] - 1.0)

    ret1 = ret(1)
    ret3 = ret(3)
    ret5 = ret(5)
    ret10 = ret(10)
    vol10 = float(close.pct_change().iloc[max(0, pos - 10) : pos + 1].std())
    return ret1, ret3, ret5, ret10, vol10


def decide_alpha(
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    defensive_alpha: float = 0.0,
    normal_alpha: float = 1.0,
    pullback_ret1_threshold: float = 0.0,
    rally_ret3_threshold: float = 0.0,
    rally_ret5_threshold: float = 0.0,
) -> RegimeDecision:
    ret1, ret3, ret5, ret10, vol10 = _index_regime(index_df, as_of)
    if ret1 < pullback_ret1_threshold and ret3 > rally_ret3_threshold and ret5 > rally_ret5_threshold:
        return RegimeDecision(
            alpha_lstm=defensive_alpha,
            reason=(
                "defensive_lgb_floor: index pulled back while 3d/5d trend "
                f"stayed positive (ret1={ret1:.4f}, ret3={ret3:.4f}, ret5={ret5:.4f})"
            ),
            idx_ret_1d=ret1,
            idx_ret_3d=ret3,
            idx_ret_5d=ret5,
            idx_ret_10d=ret10,
            idx_vol_10d=vol10,
        )
    return RegimeDecision(
        alpha_lstm=normal_alpha,
        reason=(
            "normal_lstm: no rally-pullback guard triggered "
            f"(ret1={ret1:.4f}, ret3={ret3:.4f}, ret5={ret5:.4f})"
        ),
        idx_ret_1d=ret1,
        idx_ret_3d=ret3,
        idx_ret_5d=ret5,
        idx_ret_10d=ret10,
        idx_vol_10d=vol10,
    )


def _weights(submission: pd.DataFrame) -> pd.Series:
    sub = submission.copy()
    sub["stock_code"] = sub["stock_code"].astype(str).str.zfill(6)
    weights = sub.set_index("stock_code")["weight"].astype(float)
    return weights[weights > 0]


def blend_submissions(lstm_sub: pd.DataFrame, lgb_sub: pd.DataFrame, alpha_lstm: float) -> pd.DataFrame:
    lstm = _weights(lstm_sub)
    lgb = _weights(lgb_sub)
    codes = sorted(set(lstm.index) | set(lgb.index))
    weights = alpha_lstm * lstm.reindex(codes).fillna(0.0) + (1.0 - alpha_lstm) * lgb.reindex(codes).fillna(0.0)
    weights = weights[weights > 0]
    weights = weights / weights.sum()
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def generate_guarded_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int = 3,
    lookback_days: int = 520,
    defensive_alpha: float = 0.0,
    normal_alpha: float = 1.0,
    pullback_ret1_threshold: float = 0.0,
    rally_ret3_threshold: float = 0.0,
    rally_ret5_threshold: float = 0.0,
) -> tuple[pd.DataFrame, RegimeDecision]:
    lstm_fit = fit_hybrid(
        prices,
        index_df,
        as_of,
        horizon=horizon,
        lookback_days=lookback_days,
        fixed_policy=AGGRESSIVE_LSTM_POLICY,
        confidence_mode="risk-balanced",
    )
    lstm_sub = generate_lstm_submission(lstm_fit, as_of, confidence_mode="risk-balanced")
    lgb_sub = generate_consensus_submission(
        prices,
        index_df,
        as_of=as_of,
        alpha_tuned_xgb=0.0,
        shape_horizon=horizon,
    )
    decision = decide_alpha(
        index_df,
        as_of,
        defensive_alpha=defensive_alpha,
        normal_alpha=normal_alpha,
        pullback_ret1_threshold=pullback_ret1_threshold,
        rally_ret3_threshold=rally_ret3_threshold,
        rally_ret5_threshold=rally_ret5_threshold,
    )
    return blend_submissions(lstm_sub, lgb_sub, alpha_lstm=decision.alpha_lstm), decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lookback-days", type=int, default=520)
    parser.add_argument("--defensive-alpha", type=float, default=0.0)
    parser.add_argument("--normal-alpha", type=float, default=1.0)
    parser.add_argument("--pullback-ret1-threshold", type=float, default=0.0)
    parser.add_argument("--rally-ret3-threshold", type=float, default=0.0)
    parser.add_argument("--rally-ret5-threshold", type=float, default=0.0)
    parser.add_argument("--out", default=str(REPORT_DIR / "generated" / "stage1_guarded_ensemble.csv"))
    parser.add_argument("--decision-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(prices["date"].max())

    submission, decision = generate_guarded_submission(
        prices,
        index_df,
        as_of=as_of,
        horizon=args.horizon,
        lookback_days=args.lookback_days,
        defensive_alpha=args.defensive_alpha,
        normal_alpha=args.normal_alpha,
        pullback_ret1_threshold=args.pullback_ret1_threshold,
        rally_ret3_threshold=args.rally_ret3_threshold,
        rally_ret5_threshold=args.rally_ret5_threshold,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if args.decision_out:
        decision_path = Path(args.decision_out)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([decision.__dict__]).to_csv(decision_path, index=False)
    print(f">> model=stage1_guarded_ensemble as_of={as_of.date()}")
    print(f">> alpha_lstm={decision.alpha_lstm:.2f}")
    print(f">> reason={decision.reason}")
    print(
        f">> idx_ret_1d={decision.idx_ret_1d:.4f} "
        f"idx_ret_3d={decision.idx_ret_3d:.4f} "
        f"idx_ret_5d={decision.idx_ret_5d:.4f} "
        f"idx_ret_10d={decision.idx_ret_10d:.4f} "
        f"idx_vol_10d={decision.idx_vol_10d:.4f}"
    )
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
