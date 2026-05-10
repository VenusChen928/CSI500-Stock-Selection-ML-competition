"""
Safely extend the cached CSI500 history to an earlier start date.

The original `download_data.py` can perform a full re-download, but this script
is designed for research iteration: it keeps the existing cache, backs it up,
downloads the missing older slice, and only then merges the result into the
standard data files used by the models.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from download_data import (
    DATA_DIR,
    fetch_constituents,
    fetch_index_hist,
    fetch_stock_hist,
)


def _backup(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    shutil.copy2(path, target)
    return target


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    return out.sort_values(["stock_code", "date"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20230101", help="earlier YYYYMMDD start date to add")
    parser.add_argument("--end", default=None, help="YYYYMMDD end date; defaults to existing max date")
    parser.add_argument("--sleep", type=float, default=0.08, help="seconds between stock requests")
    parser.add_argument("--backup-dir", default=str(DATA_DIR / "backup_before_history_extend"))
    parser.add_argument("--codes", default=None, help="optional comma-separated stock codes for a smoke test")
    parser.add_argument("--write-subset", action="store_true", help="allow --codes runs to overwrite main data files")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    prices_path = DATA_DIR / "prices.parquet"
    index_path = DATA_DIR / "index.parquet"
    constituents_path = DATA_DIR / "constituents.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"{prices_path} does not exist; run download_data.py first")

    DATA_DIR.mkdir(exist_ok=True)
    backup_dir = Path(args.backup_dir)
    print(f">> Backing up current data to {backup_dir}")
    for path in (prices_path, index_path, constituents_path):
        backed_up = _backup(path, backup_dir)
        if backed_up:
            print(f"   {path.name} -> {backed_up}")

    existing = _normalize_prices(pd.read_parquet(prices_path))
    current_min = existing["date"].min()
    current_max = existing["date"].max()
    requested_start = pd.Timestamp(args.start)
    end = args.end or current_max.strftime("%Y%m%d")
    print(
        f">> Existing price cache: {current_min.date()}..{current_max.date()} "
        f"({len(existing):,} rows, {existing['stock_code'].nunique()} stocks)"
    )
    if requested_start >= current_min:
        print(">> Requested start is not earlier than existing data; nothing to extend.")
        return

    slice_end = (current_min - pd.Timedelta(days=1)).strftime("%Y%m%d")
    print(f">> Downloading older slice {args.start}..{slice_end}")

    cons_full = fetch_constituents()
    cons = cons_full.copy()
    if args.codes:
        codes = [code.strip().zfill(6) for code in args.codes.split(",") if code.strip()]
        cons = cons[cons["stock_code"].isin(codes)].copy()
    else:
        # Keep the current model universe stable. The constituent list is still
        # refreshed for names, but only stocks already present in the cache are
        # extended backwards.
        cached_codes = set(existing["stock_code"].unique())
        cons = cons[cons["stock_code"].isin(cached_codes)].copy()
    codes = cons["stock_code"].tolist()
    print(f"   {len(codes)} stock histories to request.")

    print(">> Fetching older CSI500 index benchmark...")
    idx_old = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()
    idx_new = fetch_index_hist(args.start, end)
    idx = (
        pd.concat([idx_old, idx_new], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    frames = []
    failures: list[str] = []
    partial_path = DATA_DIR / "prices.history_extend.partial.parquet"
    for i, code in enumerate(tqdm(codes)):
        df = fetch_stock_hist(code, args.start, slice_end)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            failures.append(code)
        if frames and (i + 1) % args.checkpoint_every == 0:
            pd.concat(frames, ignore_index=True).to_parquet(partial_path, index=False)
        time.sleep(args.sleep)

    if not frames:
        raise RuntimeError("No older stock rows downloaded; current cache was left untouched.")

    older = _normalize_prices(pd.concat(frames, ignore_index=True))
    merged = (
        pd.concat([existing, older], ignore_index=True)
        .drop_duplicates(subset=["date", "stock_code"], keep="last")
        .sort_values(["stock_code", "date"])
        .reset_index(drop=True)
    )

    if args.codes and not args.write_subset:
        print(">> Smoke-test mode: downloaded and merged in memory; main data files left unchanged.")
        print(
            f"   would save prices: {merged['date'].min().date()}..{merged['date'].max().date()} "
            f"({len(merged):,} rows, {merged['stock_code'].nunique()} stocks)"
        )
        print(f"   downloaded older rows: {len(older):,}; failures: {len(failures)}")
        if partial_path.exists():
            partial_path.unlink()
        return

    print(">> Writing merged data files")
    idx.to_parquet(index_path, index=False)
    merged.to_parquet(prices_path, index=False)
    final_codes = set(merged["stock_code"].unique())
    cons_full[cons_full["stock_code"].isin(final_codes)].to_csv(constituents_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    print(
        f">> Saved prices: {merged['date'].min().date()}..{merged['date'].max().date()} "
        f"({len(merged):,} rows, {merged['stock_code'].nunique()} stocks)"
    )
    print(f">> Saved index rows: {len(idx):,}")
    print(f">> Downloaded older rows: {len(older):,}; failures: {len(failures)}")
    if failures:
        preview = ", ".join(failures[:12])
        suffix = "..." if len(failures) > 12 else ""
        print(f"   failed codes: {preview}{suffix}")


if __name__ == "__main__":
    main()
