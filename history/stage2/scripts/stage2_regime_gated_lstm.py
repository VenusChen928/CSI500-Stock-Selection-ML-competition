"""
Regime-routed 5-day stage2 ensemble.

This keeps the historically strongest idea, gated layered LSTM, but makes the
gate regime-aware:
  - fixed-policy LSTM when a strong medium-term trend is pulling back,
  - tuned XGB during broad recovery from a weak 20-day tape,
  - valuation LightGBM when valuation/size confirmation is more stable,
  - LightGBM as defense in an extreme weak-market selloff.

The rules are deliberately simple and auditable; they should be judged only by
multi-window score_submission tests.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "archive" / "legacy_scripts"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import FORWARD_HORIZON, build_features

DATA_DIR = ROOT / "data"
DEFAULT_OPEN_DIR = ROOT / "archive" / "data_unused" / "open"
FIXED_LSTM_TOP_K = 30
FIXED_LSTM_TEMPERATURE = 0.65
FIXED_LSTM_RANK_BLEND = 1.0


@dataclass(frozen=True)
class RegimeDecision:
    route: str
    reason: str
    idx_ret_1d: float
    idx_ret_3d: float
    idx_ret_5d: float
    idx_ret_10d: float
    idx_ret_20d: float
    idx_vol_20d: float
    breadth_ret_5d_pos: float
    median_ret_5d: float
    median_ret_20d: float


def _index_ret(close: pd.Series, pos: int, days: int) -> float:
    if pos < days:
        return 0.0
    return float(close.iloc[pos] / close.iloc[pos - days] - 1.0)


def regime_features(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").set_index("date")
    if as_of not in idx.index:
        raise ValueError(f"as_of {as_of.date()} not found in index data")
    pos = idx.index.get_loc(as_of)
    close = idx["close"].astype(float)

    panel = build_features(prices[prices["date"] <= as_of].copy(), index_df[index_df["date"] <= as_of].copy())
    today = panel[panel["date"] == as_of]
    return {
        "idx_ret_1d": _index_ret(close, pos, 1),
        "idx_ret_3d": _index_ret(close, pos, 3),
        "idx_ret_5d": _index_ret(close, pos, 5),
        "idx_ret_10d": _index_ret(close, pos, 10),
        "idx_ret_20d": _index_ret(close, pos, 20),
        "idx_vol_20d": float(close.pct_change().iloc[max(0, pos - 20) : pos + 1].std()),
        "breadth_ret_5d_pos": float(today["breadth_ret_5d_pos"].median()),
        "median_ret_5d": float(today["ret_5d"].median()),
        "median_ret_20d": float(today["ret_20d"].median()),
    }


def decide_route(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp) -> RegimeDecision:
    f = regime_features(prices, index_df, as_of)
    ret1 = f["idx_ret_1d"]
    ret5 = f["idx_ret_5d"]
    ret10 = f["idx_ret_10d"]
    ret20 = f["idx_ret_20d"]
    breadth = f["breadth_ret_5d_pos"]
    median5 = f["median_ret_5d"]

    if ret20 < -0.08 and breadth < 0.30 and ret1 < 0:
        route = "lightgbm"
        reason = "extreme_weak_selloff_defense"
    elif ret20 < -0.08 and breadth < 0.30 and ret1 >= 0:
        route = "fixed_lstm"
        reason = "panic_rebound_sequence_signal"
    elif ret20 < -0.075 and ret5 >= 0 and breadth >= 0.45:
        route = "fixed_lstm"
        reason = "downtrend_stabilization_sequence_signal"
    elif ret20 < -0.07 and ret5 < 0 and breadth >= 0.35:
        route = "open_lgb_valuation"
        reason = "weak_choppy_tape_use_valuation_confirmation"
    elif ret20 < 0 and ret5 > 0.035 and breadth > 0.65:
        route = "tuned_xgb"
        reason = "broad_recovery_from_weak_20d_tape"
    elif ret20 > 0.06 and ret1 < 0:
        route = "fixed_lstm"
        reason = "strong_medium_trend_short_pullback_sequence_signal"
    elif ret20 > 0 and ret5 > 0.02 and ret1 >= 0:
        route = "open_lgb_valuation"
        reason = "positive_trend_use_valuation_confirmation"
    elif ret10 > 0.04 and ret1 < 0 and median5 <= 0.005:
        route = "tuned_xgb"
        reason = "rally_pullback_without_strong_20d_sequence_edge"
    else:
        route = "open_lgb_valuation"
        reason = "default_stable_valuation_tree"

    return RegimeDecision(route=route, reason=reason, **f)


def _generate_layered_lstm(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp, open_dir: str | Path) -> pd.DataFrame:
    import lstm_rank_weight as lstm
    from layered_lstm_portfolio import generate_layered_submission, select_layer
    from open_data_features import add_open_data_features

    fit = lstm.fit_lstm(prices, index_df, as_of)
    panel_open, _ = add_open_data_features(fit["panel"], open_dir=open_dir)
    policy, table = select_layer(fit, panel_open, as_of, fit["prices"], fit["index_df"])
    print(f">> layered_policy={policy}")
    print(">> layered_policy_table")
    print(table.head(8).to_string(index=False))
    return generate_layered_submission(fit, panel_open, as_of, policy)


def _generate_fixed_lstm(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    import lstm_rank_weight as lstm

    fit = lstm.fit_lstm(prices, index_df, as_of)
    records = lstm.build_records(
        fit["panel"],
        [as_of],
        target_horizon=fit.get("target_horizon", FORWARD_HORIZON),
        require_targets=False,
    )
    records = fit["normalizer"].apply(records)
    pred = lstm.predict_records(fit["model"], records)
    policy = lstm.Policy(FIXED_LSTM_TOP_K, FIXED_LSTM_TEMPERATURE, FIXED_LSTM_RANK_BLEND)
    weights = lstm.weights_from_predictions(pred, as_of, policy)
    print(f">> fixed_lstm_policy={policy}")
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values})


def generate_regime_submission(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    open_dir: str | Path = DEFAULT_OPEN_DIR,
) -> tuple[pd.DataFrame, RegimeDecision]:
    decision = decide_route(prices, index_df, as_of)
    print(f">> route={decision.route} reason={decision.reason}")

    if decision.route == "fixed_lstm":
        sub = _generate_fixed_lstm(prices, index_df, as_of)
    elif decision.route == "layered_lstm":
        sub = _generate_layered_lstm(prices, index_df, as_of, open_dir)
    elif decision.route == "tuned_xgb":
        from tuned_xgboost_portfolio import fit_tuned_model, generate_submission as generate_tuned_submission

        fit = fit_tuned_model(prices=prices, index_df=index_df, as_of=as_of)
        sub = generate_tuned_submission(fit, as_of=as_of)
    elif decision.route == "lightgbm":
        from lightgbm_portfolio import fit_lightgbm_model, generate_submission as generate_lightgbm_submission

        fit = fit_lightgbm_model(prices=prices, index_df=index_df, as_of=as_of)
        sub = generate_lightgbm_submission(fit, as_of=as_of)
    elif decision.route == "open_lgb_valuation":
        from stage2_open_tree_portfolio import fit_open_tree_model, generate_submission as generate_open_tree_submission

        fit = fit_open_tree_model(
            prices=prices,
            index_df=index_df,
            as_of=as_of,
            model_type="lightgbm",
            groups=["valuation"],
            open_dir=open_dir,
        )
        sub = generate_open_tree_submission(fit, as_of=as_of)
    else:
        raise ValueError(f"Unsupported route {decision.route}")
    return sub, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DEFAULT_OPEN_DIR))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default="submissions/stage2/current_best/stage2_regime_gated_lstm.csv")
    parser.add_argument("--decision-out", default=None)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    submission, decision = generate_regime_submission(prices, index_df, as_of, open_dir=args.open_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    if args.decision_out:
        decision_path = Path(args.decision_out)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([decision.__dict__]).to_csv(decision_path, index=False)
    print(f">> as_of={as_of.date()} route={decision.route}")
    print(f">> reason={decision.reason}")
    print(
        f">> idx_ret_1d={decision.idx_ret_1d:.4f} idx_ret_5d={decision.idx_ret_5d:.4f} "
        f"idx_ret_10d={decision.idx_ret_10d:.4f} idx_ret_20d={decision.idx_ret_20d:.4f} "
        f"breadth={decision.breadth_ret_5d_pos:.4f}"
    )
    print(f">> wrote {len(submission)} names to {out_path}")
    print(f"   weight summary: min={submission['weight'].min():.4f} max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
