"""
Stage2 5-trading-day multi-window backtest runner.

Each model is executed in a fresh subprocess so Torch / OpenMP state does not
leak across windows.  Scores are computed with the canonical score_submission.py
logic for a fair apples-to-apples comparison.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "common" / "legacy_scripts"
ARCHIVED_STAGE2 = ROOT / "history" / "stage2" / "scripts"
PYTHON = Path("/opt/anaconda3/envs/mlcomp-sp26/bin/python")
FORWARD_HORIZON = 5


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{LEGACY}:{env.get('PYTHONPATH', '')}"
    env["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:/opt/homebrew/opt/openssl@3/lib"
    env["MLCOMP_DEVICE"] = "cpu"
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MLCOMP_TORCH_THREADS",
        "MLCOMP_TORCH_INTEROP_THREADS",
    ]:
        env[key] = "1"
    return env


def run(cmd: list[str], *, timeout: int | None = None) -> str:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.check_output(
        cmd,
        cwd=ROOT,
        env=_env(),
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def parse_score(text: str) -> dict[str, float]:
    row: dict[str, float] = {}
    for line in text.splitlines():
        if "portfolio return" in line:
            row["portfolio_return"] = float(line.split()[3].replace("%", "")) / 100.0
        elif "benchmark return" in line:
            row["benchmark_return"] = float(line.split()[3].replace("%", "")) / 100.0
        elif "excess return" in line:
            row["excess_return"] = float(line.split()[3].replace("%", "")) / 100.0
    return row


def score(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    text = run([
        str(PYTHON),
        "score_submission.py",
        str(path),
        "--start",
        start.strftime("%Y%m%d"),
        "--end",
        end.strftime("%Y%m%d"),
    ])
    return parse_score(text)


def rolling_windows(trading_dates: np.ndarray, windows: int, step: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    out = []
    max_asof_idx = len(trading_dates) - FORWARD_HORIZON - 1
    idx = max_asof_idx - step * (windows - 1)
    while idx <= max_asof_idx:
        if idx >= 120:
            as_of = pd.Timestamp(trading_dates[idx])
            start = pd.Timestamp(trading_dates[idx + 1])
            end = pd.Timestamp(trading_dates[idx + FORWARD_HORIZON])
            out.append((as_of, start, end))
        idx += step
    return out


def full_week_windows(trading_dates: np.ndarray, windows: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return latest complete Mon-Fri evaluation windows.

    The original rolling window walks by trading days, so a five-trading-day
    score can span weekends or holidays.  For diagnostics that should represent
    one uninterrupted market workweek, require the five evaluation dates to be
    Monday through Friday with no calendar gaps.
    """
    dates = [pd.Timestamp(d) for d in trading_dates]
    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for idx in range(len(dates) - FORWARD_HORIZON):
        as_of = dates[idx]
        eval_dates = dates[idx + 1 : idx + FORWARD_HORIZON + 1]
        if len(eval_dates) != FORWARD_HORIZON:
            continue
        is_mon_fri = eval_dates[0].weekday() == 0 and eval_dates[-1].weekday() == 4
        no_calendar_gap = (eval_dates[-1] - eval_dates[0]).days == 4
        if is_mon_fri and no_calendar_gap:
            out.append((as_of, eval_dates[0], eval_dates[-1]))
    return out[-windows:]


def explicit_windows(trading_dates: np.ndarray, as_of_values: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    out = []
    for value in as_of_values:
        as_of = pd.to_datetime(value, format="%Y%m%d")
        if as_of not in dates:
            raise ValueError(f"as_of {value} is not in trading dates")
        idx = dates.index(as_of)
        if idx + FORWARD_HORIZON >= len(dates):
            raise ValueError(f"as_of {value} does not have {FORWARD_HORIZON} future trading days in local data")
        out.append((as_of, dates[idx + 1], dates[idx + FORWARD_HORIZON]))
    return out


def model_commands(as_of: pd.Timestamp, out_dir: Path, open_dir: Path) -> dict[str, tuple[list[str], Path]]:
    stamp = as_of.strftime("%Y%m%d")
    return {
        "baseline_xgb": (
            [str(PYTHON), "-u", "baseline_xgboost.py", "--as-of", stamp, "--out", str(out_dir / f"baseline_xgb_{stamp}.csv")],
            out_dir / f"baseline_xgb_{stamp}.csv",
        ),
        "tuned_xgb": (
            [
                str(PYTHON), "-u", str(LEGACY / "tuned_xgboost_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"tuned_xgb_{stamp}.csv"),
            ],
            out_dir / f"tuned_xgb_{stamp}.csv",
        ),
        "lightgbm": (
            [
                str(PYTHON), "-u", str(LEGACY / "lightgbm_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"lightgbm_{stamp}.csv"),
            ],
            out_dir / f"lightgbm_{stamp}.csv",
        ),
        "catboost_quantile": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_catboost_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--loss-function", "Quantile:alpha=0.55",
                "--out", str(out_dir / f"catboost_quantile_{stamp}.csv"),
                "--shape-out", str(out_dir / f"catboost_quantile_shape_{stamp}.csv"),
            ],
            out_dir / f"catboost_quantile_{stamp}.csv",
        ),
        "lstm_rank_weight": (
            [str(PYTHON), "-u", "lstm_rank_weight.py", "--as-of", stamp, "--out", str(out_dir / f"lstm_rank_weight_{stamp}.csv")],
            out_dir / f"lstm_rank_weight_{stamp}.csv",
        ),
        "layered_lstm": (
            [
                str(PYTHON), "-u", str(LEGACY / "layered_lstm_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--as-of", stamp,
                "--out", str(out_dir / f"layered_lstm_{stamp}.csv"),
                "--table-out", str(out_dir / f"layered_lstm_policy_{stamp}.csv"),
            ],
            out_dir / f"layered_lstm_{stamp}.csv",
        ),
        "gated_layered_lstm": (
            [
                str(PYTHON), "-u", str(LEGACY / "gated_layered_lstm.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--as-of", stamp,
                "--out", str(out_dir / f"gated_layered_lstm_{stamp}.csv"),
                "--table-out", str(out_dir / f"gated_layered_lstm_policy_{stamp}.csv"),
            ],
            out_dir / f"gated_layered_lstm_{stamp}.csv",
        ),
        "open_xgb_valuation": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_open_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--groups", "valuation",
                "--model-type", "xgb",
                "--as-of", stamp,
                "--out", str(out_dir / f"open_xgb_valuation_{stamp}.csv"),
                "--shape-out", str(out_dir / f"open_xgb_valuation_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"open_xgb_valuation_features_{stamp}.csv"),
            ],
            out_dir / f"open_xgb_valuation_{stamp}.csv",
        ),
        "open_lgb_valuation": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_open_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--groups", "valuation",
                "--model-type", "lightgbm",
                "--as-of", stamp,
                "--out", str(out_dir / f"open_lgb_valuation_{stamp}.csv"),
                "--shape-out", str(out_dir / f"open_lgb_valuation_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"open_lgb_valuation_features_{stamp}.csv"),
            ],
            out_dir / f"open_lgb_valuation_{stamp}.csv",
        ),
        "open_blend_valuation": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_open_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--groups", "valuation",
                "--model-type", "blend",
                "--as-of", stamp,
                "--out", str(out_dir / f"open_blend_valuation_{stamp}.csv"),
                "--shape-out", str(out_dir / f"open_blend_valuation_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"open_blend_valuation_features_{stamp}.csv"),
            ],
            out_dir / f"open_blend_valuation_{stamp}.csv",
        ),
        "open_lgb_val_mkt": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_open_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--groups", "valuation", "market_regime",
                "--model-type", "lightgbm",
                "--as-of", stamp,
                "--out", str(out_dir / f"open_lgb_val_mkt_{stamp}.csv"),
                "--shape-out", str(out_dir / f"open_lgb_val_mkt_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"open_lgb_val_mkt_features_{stamp}.csv"),
            ],
            out_dir / f"open_lgb_val_mkt_{stamp}.csv",
        ),
        "open_blend_val_mkt": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_open_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--groups", "valuation", "market_regime",
                "--model-type", "blend",
                "--as-of", stamp,
                "--out", str(out_dir / f"open_blend_val_mkt_{stamp}.csv"),
                "--shape-out", str(out_dir / f"open_blend_val_mkt_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"open_blend_val_mkt_features_{stamp}.csv"),
            ],
            out_dir / f"open_blend_val_mkt_{stamp}.csv",
        ),
        "open_lgb_market": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_open_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--groups", "market_regime",
                "--model-type", "lightgbm",
                "--as-of", stamp,
                "--out", str(out_dir / f"open_lgb_market_{stamp}.csv"),
                "--shape-out", str(out_dir / f"open_lgb_market_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"open_lgb_market_features_{stamp}.csv"),
            ],
            out_dir / f"open_lgb_market_{stamp}.csv",
        ),
        "baseline_flow_gate": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_fund_flow_gate.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--as-of", stamp,
                "--out", str(out_dir / f"baseline_flow_gate_{stamp}.csv"),
                "--table-out", str(out_dir / f"baseline_flow_gate_policy_{stamp}.csv"),
            ],
            out_dir / f"baseline_flow_gate_{stamp}.csv",
        ),
        "regime_gated_lstm": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_regime_gated_lstm.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--open-dir", str(open_dir),
                "--as-of", stamp,
                "--out", str(out_dir / f"regime_gated_lstm_{stamp}.csv"),
                "--decision-out", str(out_dir / f"regime_gated_lstm_decision_{stamp}.csv"),
            ],
            out_dir / f"regime_gated_lstm_{stamp}.csv",
        ),
        "enhanced_lgb_stock": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_enhanced_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--model-type", "lightgbm",
                "--as-of", stamp,
                "--out", str(out_dir / f"enhanced_lgb_stock_{stamp}.csv"),
                "--shape-out", str(out_dir / f"enhanced_lgb_stock_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"enhanced_lgb_stock_features_{stamp}.csv"),
            ],
            out_dir / f"enhanced_lgb_stock_{stamp}.csv",
        ),
        "enhanced_blend_stock": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_enhanced_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--model-type", "blend",
                "--as-of", stamp,
                "--out", str(out_dir / f"enhanced_blend_stock_{stamp}.csv"),
                "--shape-out", str(out_dir / f"enhanced_blend_stock_shape_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"enhanced_blend_stock_features_{stamp}.csv"),
            ],
            out_dir / f"enhanced_blend_stock_{stamp}.csv",
        ),
        "regularized_consensus": (
            [
                str(PYTHON), "-u", "stage2_regularized_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"regularized_consensus_{stamp}.csv"),
                "--feature-report-out", str(out_dir / f"regularized_consensus_features_{stamp}.csv"),
                "--shape-report-out", str(out_dir / f"regularized_consensus_shape_{stamp}.csv"),
            ],
            out_dir / f"regularized_consensus_{stamp}.csv",
        ),
        "tree_consensus": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"tree_consensus_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_{stamp}.csv",
        ),
        "tree_consensus_aggressive": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.25",
                "--rank-power", "1.2",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--out", str(out_dir / f"tree_consensus_aggressive_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_aggressive_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_aggressive_{stamp}.csv",
        ),
        "tree_consensus_guarded": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.25",
                "--rank-power", "1.2",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--defensive-equal-gate",
                "--out", str(out_dir / f"tree_consensus_guarded_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_guarded_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_guarded_{stamp}.csv",
        ),
        "tree_consensus_guarded_v2": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.10",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--defensive-equal-gate",
                "--out", str(out_dir / f"tree_consensus_guarded_v2_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_guarded_v2_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_guarded_v2_{stamp}.csv",
        ),
        "tree_consensus_ref_decay": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.10",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--feature-set", "reference",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--defensive-equal-gate",
                "--out", str(out_dir / f"tree_consensus_ref_decay_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_ref_decay_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_ref_decay_{stamp}.csv",
        ),
        "tree_consensus_reference": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.10",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--feature-set", "reference",
                "--defensive-equal-gate",
                "--out", str(out_dir / f"tree_consensus_reference_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_reference_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_reference_{stamp}.csv",
        ),
        "tree_consensus_decay": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.10",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--defensive-equal-gate",
                "--out", str(out_dir / f"tree_consensus_decay_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_decay_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_decay_{stamp}.csv",
        ),
        "tree_consensus_adaptive_decay": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.05",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--adaptive-time-decay",
                "--defensive-equal-gate",
                "--out", str(out_dir / f"tree_consensus_adaptive_decay_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_adaptive_decay_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_adaptive_decay_{stamp}.csv",
        ),
        "tree_consensus_drawdown_overlay": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.05",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--adaptive-time-decay",
                "--defensive-equal-gate",
                "--reweight-factor", "drawdown_20d",
                "--reweight-direction", "low",
                "--reweight-gamma", "1.5",
                "--reweight-power", "1.0",
                "--reweight-gate", "medium_move",
                "--out", str(out_dir / f"tree_consensus_drawdown_overlay_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_drawdown_overlay_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_drawdown_overlay_{stamp}.csv",
        ),
        "tree_consensus_quality": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.05",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--feature-set", "quality",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--adaptive-time-decay",
                "--defensive-equal-gate",
                "--reweight-factor", "drawdown_20d",
                "--reweight-direction", "low",
                "--reweight-gamma", "1.5",
                "--reweight-power", "1.0",
                "--reweight-gate", "medium_move",
                "--out", str(out_dir / f"tree_consensus_quality_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_quality_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_quality_{stamp}.csv",
        ),
        "tree_lstm_gate": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_tree_lstm_gate.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"tree_lstm_gate_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_lstm_gate_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_lstm_gate_{stamp}.csv",
        ),
        "hybrid_gate": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--out", str(out_dir / f"hybrid_gate_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_{stamp}.csv",
        ),
        "hybrid_gate_factor_ic": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--factor-ic-filter",
                "--out", str(out_dir / f"hybrid_gate_factor_ic_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_factor_ic_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_factor_ic_{stamp}.csv",
        ),
        "hybrid_gate_factor_ic_dampen": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--factor-ic-dampen",
                "--out", str(out_dir / f"hybrid_gate_factor_ic_dampen_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_factor_ic_dampen_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_factor_ic_dampen_{stamp}.csv",
        ),
        "hybrid_gate_quality": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--tree-feature-set", "quality",
                "--out", str(out_dir / f"hybrid_gate_quality_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_quality_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_quality_{stamp}.csv",
        ),
        "hybrid_gate_adaptive_cap": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--tree-base-max-weight", "-1",
                "--out", str(out_dir / f"hybrid_gate_adaptive_cap_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_adaptive_cap_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_adaptive_cap_{stamp}.csv",
        ),
        "hybrid_gate_no_alpha": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--alpha-mode", "none",
                "--out", str(out_dir / f"hybrid_gate_no_alpha_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_no_alpha_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_no_alpha_{stamp}.csv",
        ),
        "hybrid_gate_no_regime": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--alpha-mode", "no_regime",
                "--out", str(out_dir / f"hybrid_gate_no_regime_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_no_regime_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_no_regime_{stamp}.csv",
        ),
        "hybrid_gate_no_liquidity": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--alpha-mode", "no_liquidity",
                "--out", str(out_dir / f"hybrid_gate_no_liquidity_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_no_liquidity_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_no_liquidity_{stamp}.csv",
        ),
        "hybrid_gate_no_secondary": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--alpha-mode", "no_secondary",
                "--out", str(out_dir / f"hybrid_gate_no_secondary_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_no_secondary_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_no_secondary_{stamp}.csv",
        ),
        "hybrid_gate_no_route": (
            [
                str(PYTHON), "-u", "stage2_hybrid_gate.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--alpha-mode", "no_route",
                "--out", str(out_dir / f"hybrid_gate_no_route_{stamp}.csv"),
                "--meta-out", str(out_dir / f"hybrid_gate_no_route_meta_{stamp}.csv"),
            ],
            out_dir / f"hybrid_gate_no_route_{stamp}.csv",
        ),
        "meta_portfolio_ensemble": (
            [
                str(PYTHON), "-u", "stage2_meta_portfolio_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--rank-power", "2.5",
                "--mix-agg", "0.0",
                "--out", str(out_dir / f"meta_portfolio_ensemble_{stamp}.csv"),
                "--meta-out", str(out_dir / f"meta_portfolio_ensemble_meta_{stamp}.csv"),
            ],
            out_dir / f"meta_portfolio_ensemble_{stamp}.csv",
        ),
        "multiroute_consensus": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_multiroute_consensus.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--cache-mode", "auto",
                "--top-k", "30",
                "--rank-power", "128",
                "--current-votes", "5",
                "--out", str(out_dir / f"multiroute_consensus_{stamp}.csv"),
                "--meta-out", str(out_dir / f"multiroute_consensus_meta_{stamp}.csv"),
            ],
            out_dir / f"multiroute_consensus_{stamp}.csv",
        ),
        "fullweek_tree": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_fullweek_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--feature-set", "all",
                "--model-type", "blend",
                "--top-k", "30",
                "--rank-power", "2.0",
                "--score-blend", "0.5",
                "--max-weight", "0.08",
                "--corr-threshold", "0.94",
                "--half-life-weeks", "52",
                "--out", str(out_dir / f"fullweek_tree_{stamp}.csv"),
                "--meta-out", str(out_dir / f"fullweek_tree_meta_{stamp}.csv"),
            ],
            out_dir / f"fullweek_tree_{stamp}.csv",
        ),
        "fullweek_tree_aggressive": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_fullweek_tree_portfolio.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--feature-set", "all",
                "--model-type", "blend",
                "--top-k", "30",
                "--rank-power", "4.0",
                "--score-blend", "0.25",
                "--max-weight", "0.10",
                "--corr-threshold", "0.94",
                "--half-life-weeks", "52",
                "--out", str(out_dir / f"fullweek_tree_aggressive_{stamp}.csv"),
                "--meta-out", str(out_dir / f"fullweek_tree_aggressive_meta_{stamp}.csv"),
            ],
            out_dir / f"fullweek_tree_aggressive_{stamp}.csv",
        ),
        "weekly_alpha_overlay": (
            [
                str(PYTHON), "-u", "stage2_weekly_alpha_overlay.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--mode", "stable",
                "--out", str(out_dir / f"weekly_alpha_overlay_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_alpha_overlay_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_alpha_overlay_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_alpha_overlay_{stamp}.csv",
        ),
        "weekly_alpha_current": (
            [
                str(PYTHON), "-u", "stage2_weekly_alpha_overlay.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--mode", "current_regime",
                "--out", str(out_dir / f"weekly_alpha_current_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_alpha_current_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_alpha_current_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_alpha_current_{stamp}.csv",
        ),
        "weekly_alpha_floor": (
            [
                str(PYTHON), "-u", "stage2_weekly_alpha_overlay.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--mode", "floor",
                "--out", str(out_dir / f"weekly_alpha_floor_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_alpha_floor_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_alpha_floor_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_alpha_floor_{stamp}.csv",
        ),
        "weekly_alpha_auto": (
            [
                str(PYTHON), "-u", "stage2_weekly_alpha_overlay.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--mode", "auto",
                "--out", str(out_dir / f"weekly_alpha_auto_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_alpha_auto_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_alpha_auto_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_alpha_auto_{stamp}.csv",
        ),
        "weekly_cycle_tree": (
            [
                str(PYTHON), "-u", "stage2_weekly_cycle_tree.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "40",
                "--score-temperature", "0.80",
                "--rank-power", "3.0",
                "--score-rank-blend", "0.60",
                "--max-weight", "0.08",
                "--corr-threshold", "0.90",
                "--half-life-days", "180",
                "--fullweek-boost", "0.20",
                "--alpha-blend", "0.25",
                "--out", str(out_dir / f"weekly_cycle_tree_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_cycle_tree_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_cycle_tree_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_cycle_tree_{stamp}.csv",
        ),
        "weekly_cycle_tree_cat": (
            [
                str(PYTHON), "-u", "stage2_weekly_cycle_tree.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "40",
                "--score-temperature", "0.80",
                "--rank-power", "3.0",
                "--score-rank-blend", "0.60",
                "--max-weight", "0.08",
                "--corr-threshold", "0.90",
                "--half-life-days", "180",
                "--fullweek-boost", "0.20",
                "--model-set", "all",
                "--alpha-blend", "0.25",
                "--out", str(out_dir / f"weekly_cycle_tree_cat_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_cycle_tree_cat_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_cycle_tree_cat_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_cycle_tree_cat_{stamp}.csv",
        ),
        "weekly_consensus": (
            [
                str(PYTHON), "-u", "stage2_weekly_consensus_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--rank-power", "6.0",
                "--max-weight", "0.095",
                "--out", str(out_dir / f"weekly_consensus_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_consensus_meta_{stamp}.csv"),
            ],
            out_dir / f"weekly_consensus_{stamp}.csv",
        ),
        "baseline_guard": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "50",
                "--out", str(out_dir / f"baseline_guard_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_{stamp}.csv",
        ),
        "baseline_guard_top40": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "40",
                "--out", str(out_dir / f"baseline_guard_top40_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_top40_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_top40_{stamp}.csv",
        ),
        "baseline_guard_top30": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "30",
                "--out", str(out_dir / f"baseline_guard_top30_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_top30_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_top30_{stamp}.csv",
        ),
        "baseline_guard_adaptive": (
            [
                str(PYTHON), "-u", "stage2_baseline_guard_ensemble.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--baseline-top-k", "0",
                "--out", str(out_dir / f"baseline_guard_adaptive_{stamp}.csv"),
                "--meta-out", str(out_dir / f"baseline_guard_adaptive_meta_{stamp}.csv"),
            ],
            out_dir / f"baseline_guard_adaptive_{stamp}.csv",
        ),
        "weekly_ridge": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_weekly_ridge_ranker.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--feature-set", "compact",
                "--alpha", "25",
                "--corr-threshold", "0.92",
                "--half-life-days", "180",
                "--fullweek-weight", "2.5",
                "--nonfull-weight", "0.6",
                "--calendar-blend", "0.25",
                "--risk-penalty", "0.10",
                "--top-k", "35",
                "--rank-power", "4.0",
                "--max-weight", "0.095",
                "--score-mix", "0.10",
                "--out", str(out_dir / f"weekly_ridge_{stamp}.csv"),
                "--meta-out", str(out_dir / f"weekly_ridge_meta_{stamp}.csv"),
                "--diagnostics-out", str(out_dir / f"weekly_ridge_diag_{stamp}.csv"),
            ],
            out_dir / f"weekly_ridge_{stamp}.csv",
        ),
        "horizon_blend": (
            [
                str(PYTHON), "-u", str(ARCHIVED_STAGE2 / "stage2_horizon_blend.py"),
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--short-weight", "0.35",
                "--max-weight", "0.04",
                "--out", str(out_dir / f"horizon_blend_{stamp}.csv"),
                "--meta-out", str(out_dir / f"horizon_blend_meta_{stamp}.csv"),
            ],
            out_dir / f"horizon_blend_{stamp}.csv",
        ),
        "tree_consensus_target3": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.0",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--adaptive-time-decay",
                "--defensive-equal-gate",
                "--target-horizon", "3",
                "--shape-horizon", "3",
                "--out", str(out_dir / f"tree_consensus_target3_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_target3_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_target3_{stamp}.csv",
        ),
        "tree_consensus_excess": (
            [
                str(PYTHON), "-u", "stage2_tree_consensus.py",
                "--prices", "data/prices.parquet",
                "--index", "data/index.parquet",
                "--as-of", stamp,
                "--top-k", "30",
                "--alpha-xgb", "0.05",
                "--rank-power", "1.6",
                "--equal-mix", "0.0",
                "--max-weight", "0.04",
                "--time-decay-half-life", "120",
                "--time-decay-floor", "0.5",
                "--adaptive-time-decay",
                "--defensive-equal-gate",
                "--target-mode", "excess",
                "--out", str(out_dir / f"tree_consensus_excess_{stamp}.csv"),
                "--meta-out", str(out_dir / f"tree_consensus_excess_meta_{stamp}.csv"),
            ],
            out_dir / f"tree_consensus_excess_{stamp}.csv",
        ),
    }


def run_one(
    *,
    model: str,
    as_of: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    open_dir: Path,
    skip_existing: bool,
    timeout: int,
) -> dict:
    commands = model_commands(as_of, out_dir, open_dir)
    if model not in commands:
        raise ValueError(f"Unknown model {model}. Valid: {sorted(commands)}")
    cmd, submission_path = commands[model]
    status = "ok"
    error = ""
    if not (skip_existing and submission_path.exists()):
        try:
            run(cmd, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = exc.output[-4000:]
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = str(exc)
    if status == "ok":
        try:
            score_row = score(submission_path, start, end)
            n_names = len(pd.read_csv(submission_path))
        except Exception as exc:
            status = "score_failed"
            error = str(exc)
            score_row = {"portfolio_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan}
            n_names = np.nan
    else:
        score_row = {"portfolio_return": np.nan, "benchmark_return": np.nan, "excess_return": np.nan}
        n_names = np.nan
    return {
        "model": model,
        "as_of": as_of.date().isoformat(),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "submission": str(submission_path.relative_to(ROOT)) if submission_path.exists() else "",
        "status": status,
        "error": error,
        "portfolio_return": score_row["portfolio_return"],
        "benchmark_return": score_row["benchmark_return"],
        "excess_return": score_row["excess_return"],
        "n_names": n_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--models", nargs="+", default=["baseline_xgb", "tuned_xgb", "lightgbm", "lstm_rank_weight", "gated_layered_lstm"])
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--as-of", nargs="*", default=None)
    parser.add_argument(
        "--full-week-only",
        action="store_true",
        help="Use only complete Monday-Friday windows with no weekend/holiday gap inside the five trading days.",
    )
    parser.add_argument("--open-dir", default=str(ROOT / "history" / "common" / "data_unused" / "open"))
    parser.add_argument("--out-dir", default=str(ROOT / "submissions" / "stage2" / "backtests" / "baselines" / "stage2_5day_current"))
    parser.add_argument("--summary-out", default=str(ROOT / "submissions" / "stage2" / "final_report_materials" / "stage2_5day_current_summary.csv"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of window/model jobs to run in parallel. Each job is a subprocess with BLAS/Torch threads pinned to 1.",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    trading_dates = np.sort(prices["date"].unique())
    if args.as_of:
        windows = explicit_windows(trading_dates, args.as_of)
    elif args.full_week_only:
        windows = full_week_windows(trading_dates, args.windows)
    else:
        windows = rolling_windows(trading_dates, args.windows, args.step)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    open_dir = Path(args.open_dir)
    if not open_dir.is_absolute():
        open_dir = ROOT / open_dir
    rows = []
    jobs = [
        {
            "model": model,
            "as_of": as_of,
            "start": start,
            "end": end,
            "out_dir": out_dir,
            "open_dir": open_dir,
            "skip_existing": args.skip_existing,
            "timeout": args.timeout,
        }
        for as_of, start, end in windows
        for model in args.models
    ]
    for as_of, start, end in windows:
        print(f"## window as_of={as_of.date()} score={start.date()}..{end.date()}", flush=True)
    if args.jobs <= 1:
        for job in jobs:
            row = run_one(**job)
            if row["status"] != "ok":
                print(f"[warn] {row['model']} {row['as_of']} {row['status']}: {row['error']}", flush=True)
            rows.append(row)
    else:
        print(f">> running {len(jobs)} jobs with --jobs={args.jobs}", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_job = {executor.submit(run_one, **job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "model": job["model"],
                        "as_of": job["as_of"].date().isoformat(),
                        "start": job["start"].date().isoformat(),
                        "end": job["end"].date().isoformat(),
                        "submission": "",
                        "status": "failed",
                        "error": str(exc),
                        "portfolio_return": np.nan,
                        "benchmark_return": np.nan,
                        "excess_return": np.nan,
                        "n_names": np.nan,
                    }
                if row["status"] == "ok":
                    print(
                        f"[ok] {row['model']} {row['as_of']} excess={row['excess_return']:.4f}",
                        flush=True,
                    )
                else:
                    print(f"[warn] {row['model']} {row['as_of']} {row['status']}: {row['error']}", flush=True)
                rows.append(row)

    detail = pd.DataFrame(rows)
    ok = detail[detail["status"] == "ok"].copy()
    summary = (
        ok.groupby("model")["excess_return"]
        .agg(["mean", "sum", "median", "min", "max", "count"])
        .sort_values(["mean", "min"], ascending=False)
        .reset_index()
    )
    summary["negative_windows"] = summary["model"].map(ok.assign(neg=ok["excess_return"] < 0).groupby("model")["neg"].sum())

    summary_out = Path(args.summary_out)
    if not summary_out.is_absolute():
        summary_out = ROOT / summary_out
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    detail_out = summary_out.with_name(summary_out.stem + "_detail.csv")
    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    print("\n>> summary")
    print(summary.to_string(index=False))
    print(f">> wrote {summary_out}")
    print(f">> wrote {detail_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
