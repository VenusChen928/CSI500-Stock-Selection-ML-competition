"""
Download open auxiliary A-share datasets for portfolio experiments.

The core competition data already contains CSI500 OHLCV.  This script adds
public, reproducible AKShare datasets that can be aligned by date without an API
key:

  - 500ETF QVIX: market volatility / risk appetite state.
  - A-share market PB: broad valuation regime.
  - Eastmoney per-stock valuation: market cap, PE/PB/PS, share capital.

The output files live under data/open/ and are optional; models should degrade
gracefully if one of them is missing.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
OPEN_DIR = DATA_DIR / "open"


def _normalize_code(code: str) -> str:
    return str(code).zfill(6)


def _load_universe(path: Path) -> list[str]:
    cons = pd.read_csv(path, dtype={"stock_code": str})
    return sorted(cons["stock_code"].map(_normalize_code).unique())


def fetch_qvix() -> pd.DataFrame:
    df = ak.index_option_500etf_qvix().copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_market_pb() -> pd.DataFrame:
    df = ak.stock_a_all_pb().copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_stock_value(code: str, retries: int = 3) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_value_em(symbol=code)
            if df is None or df.empty:
                return None
            df = df.copy()
            df["stock_code"] = code
            df = df.rename(
                columns={
                    "数据日期": "date",
                    "当日收盘价": "value_close",
                    "当日涨跌幅": "value_pct_change",
                    "总市值": "total_mv",
                    "流通市值": "float_mv",
                    "总股本": "total_share",
                    "流通股本": "float_share",
                    "PE(TTM)": "pe_ttm",
                    "PE(静)": "pe_static",
                    "市净率": "pb",
                    "PEG值": "peg",
                    "市现率": "pcf",
                    "市销率": "ps",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            keep = [
                "date",
                "stock_code",
                "value_close",
                "value_pct_change",
                "total_mv",
                "float_mv",
                "total_share",
                "float_share",
                "pe_ttm",
                "pe_static",
                "pb",
                "peg",
                "pcf",
                "ps",
            ]
            return df[[c for c in keep if c in df.columns]].sort_values("date")
        except Exception as exc:  # pragma: no cover - network retry path
            last_err = exc
            time.sleep(1.0 + attempt)
    print(f"  [warn] stock_value_em({code}) failed: {last_err}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constituents", default=str(DATA_DIR / "constituents.csv"))
    parser.add_argument("--start", default="20230101", help="YYYYMMDD lower bound for stock valuation rows")
    parser.add_argument("--end", default=None, help="YYYYMMDD upper bound for stock valuation rows")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None, help="debug: only fetch first N stocks")
    args = parser.parse_args()

    OPEN_DIR.mkdir(parents=True, exist_ok=True)

    print(">> Fetching 500ETF QVIX")
    qvix = fetch_qvix()
    qvix.to_parquet(OPEN_DIR / "qvix_500etf.parquet", index=False)
    print(f"   saved {len(qvix):,} rows to {OPEN_DIR / 'qvix_500etf.parquet'}")

    print(">> Fetching A-share market PB")
    market_pb = fetch_market_pb()
    market_pb.to_parquet(OPEN_DIR / "market_pb.parquet", index=False)
    print(f"   saved {len(market_pb):,} rows to {OPEN_DIR / 'market_pb.parquet'}")

    codes = _load_universe(Path(args.constituents))
    if args.limit:
        codes = codes[: args.limit]
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else None

    existing_path = OPEN_DIR / "stock_value_em.parquet"
    existing = None
    fetched_codes: set[str] = set()
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing["stock_code"] = existing["stock_code"].astype(str).str.zfill(6)
        fetched_codes = set(existing["stock_code"].unique())
        print(f">> Existing stock valuation cache: {len(existing):,} rows, {len(fetched_codes)} stocks")

    frames = [] if existing is None else [existing]
    todo = [code for code in codes if code not in fetched_codes]
    print(f">> Fetching stock valuation data for {len(todo)} uncached stocks")
    for i, code in enumerate(tqdm(todo)):
        df = fetch_stock_value(code)
        if df is not None and not df.empty:
            df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]
            frames.append(df)
        if frames and (i + 1) % 50 == 0:
            pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["date", "stock_code"]
            ).sort_values(["stock_code", "date"]).to_parquet(existing_path, index=False)
        time.sleep(args.sleep)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
        out = out.drop_duplicates(subset=["date", "stock_code"]).sort_values(["stock_code", "date"])
        out.to_parquet(existing_path, index=False)
        print(f">> Saved {len(out):,} stock valuation rows for {out['stock_code'].nunique()} stocks")


if __name__ == "__main__":
    main()
