"""
LSTM rank/weight model with open-data sequence features.

This reuses the stable implementation in lstm_rank_weight.py but expands each
per-stock sequence with optional AKShare valuation and market-state features.
It is kept as a separate challenger so the previous best LSTM remains
reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lstm_rank_weight as base
from features import FORWARD_HORIZON, build_features
from open_data_features import add_open_data_features

DATA_DIR = ROOT / "data"


def fit_lstm_open(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: pd.Timestamp, open_dir: str | Path):
    min_date = as_of - pd.Timedelta(days=base.LOOKBACK_DAYS)
    prices = prices[prices["date"] >= min_date].copy()
    index_df = index_df[index_df["date"] >= min_date].copy()

    panel = build_features(prices, index_df)
    panel, open_cols = add_open_data_features(panel, open_dir=open_dir)
    base.SEQUENCE_FEATURES = list(dict.fromkeys(base.SEQUENCE_FEATURES + open_cols))
    panel = base.add_direct_targets(panel, index_df)

    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    train_cutoff = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    usable = panel.dropna(subset=base.SEQUENCE_FEATURES + ["target_rank_5d", "target_weight_5d"])
    train_pool = usable[usable["date"] <= train_cutoff].copy()
    train_dates, val_dates, train_end, val_start = base.date_splits(train_pool)
    train_records_raw = base.build_records(panel, train_dates)
    val_records_raw = base.build_records(panel, val_dates)
    normalizer = base.Normalizer(train_records_raw)
    train_records = normalizer.apply(train_records_raw)
    val_records = normalizer.apply(val_records_raw)
    print(f">> device={base.DEVICE} train_records={len(train_records):,} val_records={len(val_records):,}")
    print(f">> train_end={train_end.date()} val_start={val_start.date()}")
    print(f">> sequence_features={len(base.SEQUENCE_FEATURES)} open_features={len(open_cols)}")
    model = base.train_model(train_records, val_records)
    val_pred = base.predict_records(model, val_records)
    policy, policy_table = base.select_policy(val_pred, val_dates, trading_dates, prices, index_df)
    return {
        "panel": panel,
        "trading_dates": trading_dates,
        "normalizer": normalizer,
        "model": model,
        "policy": policy,
        "policy_table": policy_table,
        "prices": prices,
        "index_df": index_df,
        "open_cols": open_cols,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DATA_DIR / "open"))
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest full 5-day as-of")
    parser.add_argument("--out", default="submissions/lstm_open_rank_weight.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(prices["date"].unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])

    fit = fit_lstm_open(prices, index_df, as_of, args.open_dir)
    submission = base.generate_submission(fit, as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(">> selected policy")
    print(fit["policy"])
    print(">> validation policy table")
    print(fit["policy_table"].head(12).to_string(index=False))
    print(f">> wrote {len(submission)} names to {out_path}")
    print(
        f"   weight summary: min={submission['weight'].min():.4f} "
        f"max={submission['weight'].max():.4f} sum={submission['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()
