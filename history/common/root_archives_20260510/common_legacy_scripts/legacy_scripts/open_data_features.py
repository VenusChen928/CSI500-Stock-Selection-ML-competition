"""
Feature joins for optional open-data augmentations.

These helpers intentionally live outside features.py so the original baseline
remains unchanged.  Missing auxiliary files are tolerated; the returned feature
list only includes columns that can be constructed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPEN_FEATURE_COLUMNS = [
    "qvix_close",
    "qvix_ret_5d",
    "qvix_ret_20d",
    "qvix_ma20_ratio",
    "market_middle_pb",
    "market_equal_pb",
    "market_pb_quantile",
    "log_total_mv",
    "log_float_mv",
    "float_mv_rank",
    "total_mv_rank",
    "pb",
    "pb_rank",
    "pe_ttm_safe",
    "pe_ttm_rank",
    "ps_safe",
    "ps_rank",
    "value_mv_ratio",
    "main_net_pct",
    "main_net_pct_ma3",
    "main_net_pct_ma5",
    "main_net_pct_ma10",
    "super_net_pct_ma5",
    "big_net_pct_ma5",
    "small_net_pct_ma5",
    "main_net_pct_rank",
    "main_net_pct_ma5_rank",
    "super_net_pct_ma5_rank",
    "big_net_pct_ma5_rank",
    "small_net_pct_ma5_rank",
]


def _safe_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _load_qvix(open_dir: Path) -> pd.DataFrame | None:
    path = open_dir / "qvix_500etf.parquet"
    if not path.exists():
        return None
    qvix = pd.read_parquet(path)
    qvix["date"] = pd.to_datetime(qvix["date"])
    qvix = _safe_numeric(qvix, ["close"])
    qvix = qvix.sort_values("date")
    close = qvix["close"]
    qvix["qvix_close"] = close
    qvix["qvix_ret_5d"] = close.pct_change(5)
    qvix["qvix_ret_20d"] = close.pct_change(20)
    qvix["qvix_ma20_ratio"] = close / close.rolling(20).mean() - 1.0
    return qvix[["date", "qvix_close", "qvix_ret_5d", "qvix_ret_20d", "qvix_ma20_ratio"]]


def _load_market_pb(open_dir: Path) -> pd.DataFrame | None:
    path = open_dir / "market_pb.parquet"
    if not path.exists():
        return None
    market = pd.read_parquet(path)
    market["date"] = pd.to_datetime(market["date"])
    market = _safe_numeric(
        market,
        [
            "middlePB",
            "equalWeightAveragePB",
            "quantileInRecent10YearsMiddlePB",
        ],
    )
    market = market.rename(
        columns={
            "middlePB": "market_middle_pb",
            "equalWeightAveragePB": "market_equal_pb",
            "quantileInRecent10YearsMiddlePB": "market_pb_quantile",
        }
    )
    keep = ["date", "market_middle_pb", "market_equal_pb", "market_pb_quantile"]
    return market[[c for c in keep if c in market.columns]].sort_values("date")


def _load_stock_value(open_dir: Path) -> pd.DataFrame | None:
    path = open_dir / "stock_value_em.parquet"
    if not path.exists():
        return None
    value = pd.read_parquet(path)
    value["date"] = pd.to_datetime(value["date"])
    value["stock_code"] = value["stock_code"].astype(str).str.zfill(6)
    value = _safe_numeric(
        value,
        ["total_mv", "float_mv", "pe_ttm", "pb", "ps", "value_close"],
    )
    value["log_total_mv"] = np.log1p(value["total_mv"].clip(lower=0))
    value["log_float_mv"] = np.log1p(value["float_mv"].clip(lower=0))
    value["pe_ttm_safe"] = value["pe_ttm"].where(value["pe_ttm"].between(0, 300))
    value["ps_safe"] = value["ps"].where(value["ps"].between(0, 100))
    value["value_mv_ratio"] = value["float_mv"] / value["total_mv"].replace(0, np.nan)
    for col in ["float_mv", "total_mv", "pb", "pe_ttm_safe", "ps_safe"]:
        value[f"{col}_rank"] = value.groupby("date")[col].rank(method="average", pct=True)
    keep = [
        "date",
        "stock_code",
        "log_total_mv",
        "log_float_mv",
        "float_mv_rank",
        "total_mv_rank",
        "pb",
        "pb_rank",
        "pe_ttm_safe",
        "pe_ttm_safe_rank",
        "ps_safe",
        "ps_safe_rank",
        "value_mv_ratio",
    ]
    out = value[[c for c in keep if c in value.columns]].copy()
    out = out.rename(
        columns={
            "pe_ttm_safe_rank": "pe_ttm_rank",
            "ps_safe_rank": "ps_rank",
        }
    )
    return out.sort_values(["stock_code", "date"])


def _load_fund_flow(open_dir: Path) -> pd.DataFrame | None:
    path = open_dir / "stock_fund_flow.parquet"
    if not path.exists():
        return None
    flow = pd.read_parquet(path)
    if flow.empty:
        return None

    flow["date"] = pd.to_datetime(flow["date"])
    flow["stock_code"] = flow["stock_code"].astype(str).str.zfill(6)
    pct_cols = [
        "main_net_pct",
        "super_net_pct",
        "big_net_pct",
        "small_net_pct",
    ]
    flow = _safe_numeric(flow, pct_cols)
    flow = flow.sort_values(["stock_code", "date"])

    for window in [3, 5, 10]:
        flow[f"main_net_pct_ma{window}"] = (
            flow.groupby("stock_code")["main_net_pct"]
            .transform(lambda s: s.rolling(window, min_periods=2).mean())
        )
    for col in ["super_net_pct", "big_net_pct", "small_net_pct"]:
        flow[f"{col}_ma5"] = (
            flow.groupby("stock_code")[col]
            .transform(lambda s: s.rolling(5, min_periods=2).mean())
        )

    for col in [
        "main_net_pct",
        "main_net_pct_ma5",
        "super_net_pct_ma5",
        "big_net_pct_ma5",
        "small_net_pct_ma5",
    ]:
        flow[f"{col}_rank"] = flow.groupby("date")[col].rank(method="average", pct=True)

    keep = [
        "date",
        "stock_code",
        "main_net_pct",
        "main_net_pct_ma3",
        "main_net_pct_ma5",
        "main_net_pct_ma10",
        "super_net_pct_ma5",
        "big_net_pct_ma5",
        "small_net_pct_ma5",
        "main_net_pct_rank",
        "main_net_pct_ma5_rank",
        "super_net_pct_ma5_rank",
        "big_net_pct_ma5_rank",
        "small_net_pct_ma5_rank",
    ]
    return flow[[c for c in keep if c in flow.columns]].sort_values(["stock_code", "date"])


def add_open_data_features(panel: pd.DataFrame, open_dir: str | Path = "data/open") -> tuple[pd.DataFrame, list[str]]:
    """Merge optional open-data features into an existing feature panel."""
    open_dir = Path(open_dir)
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)

    market_frames = [_load_qvix(open_dir), _load_market_pb(open_dir)]
    for frame in [f for f in market_frames if f is not None and not f.empty]:
        # Calendar gaps are forward-filled after aligning to panel dates.
        frame = frame.sort_values("date")
        dates = pd.DataFrame({"date": pd.to_datetime(np.sort(out["date"].unique())).astype("datetime64[ns]")})
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        frame = pd.merge_asof(dates, frame, on="date", direction="backward")
        out = out.merge(frame, on="date", how="left")

    value = _load_stock_value(open_dir)
    if value is not None and not value.empty:
        out = out.merge(value, on=["date", "stock_code"], how="left")

    flow = _load_fund_flow(open_dir)
    if flow is not None and not flow.empty:
        out = out.merge(flow, on=["date", "stock_code"], how="left")

    available = [c for c in OPEN_FEATURE_COLUMNS if c in out.columns]
    for col in available:
        if out[col].isna().any():
            if col.endswith("_rank"):
                out[col] = out[col].fillna(0.5)
            else:
                out[col] = out[col].fillna(out.groupby("date")[col].transform("median"))
                out[col] = out[col].fillna(out[col].median())
    return out, available
