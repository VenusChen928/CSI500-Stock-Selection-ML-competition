"""
Download recent individual-stock fund-flow data via AKShare.

The Eastmoney fund-flow endpoint usually returns about 120 recent trading days
per stock, which is enough for the current rolling validation windows.  These
features are intended for portfolio-layer/gate experiments rather than direct
LSTM sequence input.
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


def _market(code: str) -> str:
    return "sh" if str(code).startswith("6") else "sz"


def _load_universe(path: Path) -> list[str]:
    cons = pd.read_csv(path, dtype={"stock_code": str})
    return sorted(cons["stock_code"].astype(str).str.zfill(6).unique())


def fetch_fund_flow(code: str, retries: int = 3) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=_market(code))
            if df is None or df.empty:
                return None
            df = df.copy()
            df["stock_code"] = code
            df = df.rename(
                columns={
                    "日期": "date",
                    "收盘价": "fund_close",
                    "涨跌幅": "fund_pct_change",
                    "主力净流入-净额": "main_net_amount",
                    "主力净流入-净占比": "main_net_pct",
                    "超大单净流入-净额": "super_net_amount",
                    "超大单净流入-净占比": "super_net_pct",
                    "大单净流入-净额": "big_net_amount",
                    "大单净流入-净占比": "big_net_pct",
                    "中单净流入-净额": "medium_net_amount",
                    "中单净流入-净占比": "medium_net_pct",
                    "小单净流入-净额": "small_net_amount",
                    "小单净流入-净占比": "small_net_pct",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            keep = [
                "date",
                "stock_code",
                "fund_close",
                "fund_pct_change",
                "main_net_amount",
                "main_net_pct",
                "super_net_amount",
                "super_net_pct",
                "big_net_amount",
                "big_net_pct",
                "medium_net_amount",
                "medium_net_pct",
                "small_net_amount",
                "small_net_pct",
            ]
            return df[[c for c in keep if c in df.columns]].sort_values("date")
        except Exception as exc:  # pragma: no cover - network retry path
            last_err = exc
            time.sleep(1.0 + attempt)
    print(f"  [warn] fund flow {code} failed: {last_err}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constituents", default=str(DATA_DIR / "constituents.csv"))
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="refetch even if cache exists")
    args = parser.parse_args()

    OPEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OPEN_DIR / "stock_fund_flow.parquet"
    existing = None
    fetched_codes: set[str] = set()
    if out_path.exists() and not args.force:
        existing = pd.read_parquet(out_path)
        existing["stock_code"] = existing["stock_code"].astype(str).str.zfill(6)
        fetched_codes = set(existing["stock_code"].unique())
        print(f">> Existing fund-flow cache: {len(existing):,} rows, {len(fetched_codes)} stocks")

    codes = _load_universe(Path(args.constituents))
    if args.limit:
        codes = codes[: args.limit]
    todo = [c for c in codes if c not in fetched_codes]
    frames = [] if existing is None or args.force else [existing]
    print(f">> Fetching fund-flow data for {len(todo)} uncached stocks")
    for i, code in enumerate(tqdm(todo)):
        df = fetch_fund_flow(code)
        if df is not None and not df.empty:
            frames.append(df)
        if frames and (i + 1) % 50 == 0:
            pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["date", "stock_code"]
            ).sort_values(["stock_code", "date"]).to_parquet(out_path, index=False)
        time.sleep(args.sleep)

    if not frames:
        raise RuntimeError("no fund-flow data fetched")
    out = pd.concat(frames, ignore_index=True)
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    out = out.drop_duplicates(subset=["date", "stock_code"]).sort_values(["stock_code", "date"])
    out.to_parquet(out_path, index=False)
    print(f">> Saved {len(out):,} fund-flow rows for {out['stock_code'].nunique()} stocks to {out_path}")
    print(f"   dates {out['date'].min().date()} to {out['date'].max().date()}")


if __name__ == "__main__":
    main()
