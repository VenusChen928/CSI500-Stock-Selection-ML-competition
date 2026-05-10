"""
Clean open-data feature builder for stage2 experiments.

This module is intentionally model-agnostic.  It only handles:
  1. strict cleaning of optional open datasets,
  2. as-of-safe alignment to the competition feature panel,
  3. cross-sectional / time-series normalization, and
  4. coverage + correlation filtering.

The model scripts should import ``add_stage2_open_features`` and decide whether a
feature group improves 5-day multi-window excess return before keeping it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from features import build_features

ROOT = Path(__file__).resolve().parent
DEFAULT_OPEN_DIR = ROOT / "archive" / "data_unused" / "open"
DEFAULT_REPORT = ROOT / "submissions" / "stage2" / "reports" / "stage2_open_feature_quality.csv"

OPEN_GROUPS = ("valuation", "market_regime", "fund_flow")


@dataclass(frozen=True)
class FeatureReportRow:
    group: str
    feature: str
    coverage_before_fill: float
    missing_before_fill: float
    std_after_fill: float
    selected: bool
    drop_reason: str


def _parse_date(value) -> pd.Timestamp:
    if value is None:
        return value
    if isinstance(value, pd.Timestamp):
        return value
    parsed = pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value)
    return pd.Timestamp(parsed)


def _date_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).astype("datetime64[ns]")


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)


def _safe_numeric(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.loc[~np.isfinite(frame[col]), col] = np.nan
    return frame


def _winsorize_by_date(frame: pd.DataFrame, cols: list[str], q_low=0.01, q_high=0.99) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        if col not in out.columns:
            continue
        bounds = out.groupby("date")[col].quantile([q_low, q_high]).unstack()
        bounds.columns = ["lo", "hi"]
        out = out.merge(bounds, left_on="date", right_index=True, how="left")
        out[col] = out[col].clip(lower=out["lo"], upper=out["hi"])
        out = out.drop(columns=["lo", "hi"])
    return out


def robust_z_by_date(frame: pd.DataFrame, cols: list[str], clip: float = 4.0) -> pd.DataFrame:
    """Add robust daily cross-sectional z-scores for ``cols``."""
    out = frame.copy()
    for col in cols:
        median = out.groupby("date")[col].transform("median")
        mad = out.groupby("date")[col].transform(lambda s: (s - s.median()).abs().median())
        scale = (1.4826 * mad).replace(0, np.nan)
        z = ((out[col] - median) / scale).clip(-clip, clip)
        out[f"{col}_z"] = z
    return out


def _rank_by_date(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[f"{col}_rank"] = out.groupby("date")[col].rank(method="average", pct=True)
    return out


def _rolling_time_z(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - mean) / std).clip(-4, 4)


def _panel_keys(panel: pd.DataFrame) -> pd.DataFrame:
    keys = panel[["date", "stock_code"]].copy()
    keys["_row_id"] = np.arange(len(keys))
    keys["date"] = _date_ns(keys["date"])
    keys["stock_code"] = _normalize_code(keys["stock_code"])
    return keys


def _merge_stock_asof(
    panel: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    feature_group: str,
    max_lag_days: int,
) -> pd.DataFrame:
    """Backward as-of merge per stock; never forward-fills from future dates."""
    keys = _panel_keys(panel)
    aux = aux.copy()
    aux["date"] = _date_ns(aux["date"])
    aux["stock_code"] = _normalize_code(aux["stock_code"])
    aux = aux.sort_values(["stock_code", "date"]).drop_duplicates(["stock_code", "date"], keep="last")
    aux["__aux_date"] = aux["date"]

    merged_parts = []
    aux_groups = {code: df.drop(columns=["stock_code"]).sort_values("date") for code, df in aux.groupby("stock_code")}
    feature_cols = [c for c in aux.columns if c not in {"date", "stock_code", "__aux_date"}]
    empty_cols = feature_cols + ["__aux_date"]

    for code, left in keys.groupby("stock_code", sort=False):
        left = left.sort_values("date")
        right = aux_groups.get(code)
        if right is None or right.empty:
            chunk = left.copy()
            for col in empty_cols:
                chunk[col] = np.nan
        else:
            chunk = pd.merge_asof(
                left,
                right,
                on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=max_lag_days),
            )
        chunk["stock_code"] = code
        merged_parts.append(chunk)

    merged = pd.concat(merged_parts, ignore_index=True).sort_values("_row_id")
    lag_col = f"od_{feature_group}_lag_days"
    merged[lag_col] = (merged["date"] - merged["__aux_date"]).dt.days
    merged = merged.drop(columns=["__aux_date"])
    return merged.drop(columns=["date", "stock_code"]).set_index("_row_id")


def _merge_market_asof(panel: pd.DataFrame, aux: pd.DataFrame, *, max_lag_days: int) -> pd.DataFrame:
    keys = panel[["date"]].copy()
    keys["_row_id"] = np.arange(len(keys))
    keys["date"] = _date_ns(keys["date"])
    aux = aux.copy().sort_values("date")
    aux["date"] = _date_ns(aux["date"])
    aux = aux.drop_duplicates("date", keep="last")
    aux["__aux_date"] = aux["date"]
    merged = pd.merge_asof(
        keys.sort_values("date"),
        aux,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=max_lag_days),
    ).sort_values("_row_id")
    merged["od_market_lag_days"] = (merged["date"] - merged["__aux_date"]).dt.days
    merged = merged.drop(columns=["date", "__aux_date"])
    return merged.set_index("_row_id")


def _load_valuation(open_dir: Path) -> pd.DataFrame:
    path = open_dir / "stock_value_em.parquet"
    if not path.exists():
        return pd.DataFrame()
    value = pd.read_parquet(path)
    if value.empty:
        return pd.DataFrame()
    value["date"] = _date_ns(value["date"])
    value["stock_code"] = _normalize_code(value["stock_code"])
    value = _safe_numeric(
        value,
        ["value_close", "value_pct_change", "total_mv", "float_mv", "pe_ttm", "pb", "ps"],
    )
    value["od_val_log_total_mv"] = np.log1p(value["total_mv"].where(value["total_mv"] > 0))
    value["od_val_log_float_mv"] = np.log1p(value["float_mv"].where(value["float_mv"] > 0))
    value["od_val_float_ratio"] = (value["float_mv"] / value["total_mv"].replace(0, np.nan)).where(lambda s: s.between(0, 1.5))
    value["od_val_mv_gap"] = value["od_val_log_total_mv"] - value["od_val_log_float_mv"]
    value["od_val_pb"] = value["pb"].where(value["pb"].between(0, 50))
    value["od_val_pe_ttm"] = value["pe_ttm"].where(value["pe_ttm"].between(0, 300))
    value["od_val_ps"] = value["ps"].where(value["ps"].between(0, 100))
    value = value.sort_values(["stock_code", "date"])
    value["od_val_value_ret_5d"] = value.groupby("stock_code")["value_close"].pct_change(5)

    raw_cols = [
        "od_val_log_total_mv",
        "od_val_log_float_mv",
        "od_val_float_ratio",
        "od_val_mv_gap",
        "od_val_pb",
        "od_val_pe_ttm",
        "od_val_ps",
        "od_val_value_ret_5d",
    ]
    value = _winsorize_by_date(value, raw_cols)
    value = robust_z_by_date(value, raw_cols)
    value = _rank_by_date(value, raw_cols)
    keep = ["date", "stock_code"]
    keep += [c for c in value.columns if c.startswith("od_val_") and (c.endswith("_z") or c.endswith("_rank"))]
    return value[keep].sort_values(["stock_code", "date"])


def _load_market_regime(open_dir: Path, *, min_qvix_coverage: float = 0.6) -> pd.DataFrame:
    frames = []
    pb_path = open_dir / "market_pb.parquet"
    if pb_path.exists():
        pb = pd.read_parquet(pb_path)
        pb["date"] = _date_ns(pb["date"])
        pb = _safe_numeric(
            pb,
            [
                "middlePB",
                "equalWeightAveragePB",
                "quantileInRecent10YearsMiddlePB",
                "quantileInRecent10YearsEqualWeightAveragePB",
            ],
        ).sort_values("date")
        pb["od_mkt_middle_pb_z252"] = _rolling_time_z(pb["middlePB"])
        pb["od_mkt_equal_pb_z252"] = _rolling_time_z(pb["equalWeightAveragePB"])
        pb["od_mkt_pb_quantile"] = pb["quantileInRecent10YearsMiddlePB"].clip(0, 100) / 100.0
        pb["od_mkt_equal_pb_quantile"] = pb["quantileInRecent10YearsEqualWeightAveragePB"].clip(0, 100) / 100.0
        pb["od_mkt_pb_slope_20d"] = pb["middlePB"].pct_change(20).clip(-1, 1)
        frames.append(pb[[
            "date",
            "od_mkt_middle_pb_z252",
            "od_mkt_equal_pb_z252",
            "od_mkt_pb_quantile",
            "od_mkt_equal_pb_quantile",
            "od_mkt_pb_slope_20d",
        ]])

    qvix_path = open_dir / "qvix_500etf.parquet"
    if qvix_path.exists():
        qvix = pd.read_parquet(qvix_path)
        qvix["date"] = _date_ns(qvix["date"])
        qvix = _safe_numeric(qvix, ["close"]).sort_values("date")
        usable = qvix["close"].notna().mean()
        if usable >= min_qvix_coverage:
            qvix["od_mkt_qvix_z252"] = _rolling_time_z(qvix["close"])
            qvix["od_mkt_qvix_ret_5d"] = qvix["close"].pct_change(5).clip(-1, 1)
            frames.append(qvix[["date", "od_mkt_qvix_z252", "od_mkt_qvix_ret_5d"]])

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date")


def _load_fund_flow(open_dir: Path) -> pd.DataFrame:
    path = open_dir / "stock_fund_flow.parquet"
    if not path.exists():
        return pd.DataFrame()
    flow = pd.read_parquet(path)
    if flow.empty:
        return pd.DataFrame()
    flow["date"] = _date_ns(flow["date"])
    flow["stock_code"] = _normalize_code(flow["stock_code"])
    flow = _safe_numeric(
        flow,
        ["main_net_pct", "super_net_pct", "big_net_pct", "small_net_pct", "fund_pct_change"],
    ).sort_values(["stock_code", "date"])
    for col in ["main_net_pct", "super_net_pct", "big_net_pct", "small_net_pct", "fund_pct_change"]:
        flow[f"od_flow_{col}"] = flow[col].clip(-30, 30) / 100.0
    for window in [3, 5, 10]:
        flow[f"od_flow_main_net_pct_ma{window}"] = (
            flow.groupby("stock_code")["od_flow_main_net_pct"]
            .transform(lambda s: s.rolling(window, min_periods=2).mean())
        )
    flow["od_flow_main_net_pct_delta_3d"] = flow.groupby("stock_code")["od_flow_main_net_pct"].diff(3).clip(-1, 1)
    raw_cols = [
        "od_flow_main_net_pct",
        "od_flow_super_net_pct",
        "od_flow_big_net_pct",
        "od_flow_small_net_pct",
        "od_flow_fund_pct_change",
        "od_flow_main_net_pct_ma3",
        "od_flow_main_net_pct_ma5",
        "od_flow_main_net_pct_ma10",
        "od_flow_main_net_pct_delta_3d",
    ]
    flow = robust_z_by_date(flow, raw_cols)
    flow = _rank_by_date(flow, raw_cols)
    keep = ["date", "stock_code"]
    keep += [c for c in flow.columns if c.startswith("od_flow_") and (c.endswith("_z") or c.endswith("_rank"))]
    return flow[keep].sort_values(["stock_code", "date"])


def _fill_feature(frame: pd.DataFrame, col: str) -> pd.Series:
    if col.endswith("_rank"):
        return frame[col].fillna(0.5)
    if col.endswith("_lag_days"):
        return frame[col].fillna(999.0).clip(0, 999)
    return frame[col].fillna(0.0)


def correlation_prune(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    threshold: float = 0.92,
    method: str = "spearman",
    sample_days: int = 252,
    min_std: float = 1e-8,
) -> tuple[list[str], dict[str, str]]:
    """Greedy correlation pruning, preferring earlier columns in ``cols``."""
    if not cols:
        return [], {}

    dates = np.sort(pd.to_datetime(frame["date"]).unique())
    if len(dates) > sample_days:
        keep_dates = set(pd.Timestamp(d) for d in dates[-sample_days:])
        sample = frame[pd.to_datetime(frame["date"]).isin(keep_dates)]
    else:
        sample = frame

    drop_reasons: dict[str, str] = {}
    usable = []
    for col in cols:
        std = float(sample[col].std(skipna=True)) if col in sample else 0.0
        if not np.isfinite(std) or std <= min_std:
            drop_reasons[col] = "near_zero_variance"
        else:
            usable.append(col)

    selected: list[str] = []
    for col in usable:
        should_drop = False
        for prev in selected:
            corr = sample[[col, prev]].corr(method=method).iloc[0, 1]
            if np.isfinite(corr) and abs(float(corr)) >= threshold:
                drop_reasons[col] = f"corr_with:{prev}:{corr:.3f}"
                should_drop = True
                break
        if not should_drop:
            selected.append(col)
    return selected, drop_reasons


def _make_report(
    frame: pd.DataFrame,
    group_by_feature: dict[str, str],
    candidates: list[str],
    selected: list[str],
    drop_reasons: dict[str, str],
    coverage_before_fill: dict[str, float],
) -> pd.DataFrame:
    rows = []
    selected_set = set(selected)
    for col in candidates:
        coverage = float(coverage_before_fill.get(col, frame[col].notna().mean()))
        rows.append(FeatureReportRow(
            group=group_by_feature.get(col, "unknown"),
            feature=col,
            coverage_before_fill=coverage,
            missing_before_fill=1.0 - coverage,
            std_after_fill=float(frame[col].std(skipna=True)),
            selected=col in selected_set,
            drop_reason="" if col in selected_set else drop_reasons.get(col, "not_selected"),
        ))
    return pd.DataFrame([r.__dict__ for r in rows]).sort_values(["group", "selected", "feature"], ascending=[True, False, True])


def add_stage2_open_features(
    panel: pd.DataFrame,
    open_dir: str | Path = DEFAULT_OPEN_DIR,
    groups: Iterable[str] = OPEN_GROUPS,
    *,
    corr_threshold: float = 0.92,
    min_coverage: float = 0.55,
    valuation_max_lag_days: int = 10,
    market_max_lag_days: int = 10,
    fund_flow_max_lag_days: int = 7,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Return ``panel`` with cleaned open-data columns, selected columns, report."""
    open_dir = Path(open_dir)
    requested = tuple(dict.fromkeys(groups))
    if "all" in requested:
        requested = OPEN_GROUPS
    unknown = sorted(set(requested) - set(OPEN_GROUPS))
    if unknown:
        raise ValueError(f"Unknown open-data groups: {unknown}. Valid groups: {OPEN_GROUPS}")

    out = panel.copy()
    out["date"] = _date_ns(out["date"])
    out["stock_code"] = _normalize_code(out["stock_code"])

    group_by_feature: dict[str, str] = {}
    coverage_before_fill: dict[str, float] = {}
    candidate_cols: list[str] = []

    if "valuation" in requested:
        value = _load_valuation(open_dir)
        if not value.empty:
            merged = _merge_stock_asof(out, value, feature_group="val", max_lag_days=valuation_max_lag_days)
            for col in merged.columns:
                out[col] = merged[col].to_numpy()
            val_cols = [c for c in merged.columns if c.startswith("od_val_")]
            out["od_val_available"] = out[val_cols].notna().any(axis=1).astype(float)
            val_cols.append("od_val_available")
            for col in val_cols:
                group_by_feature[col] = "valuation"
                coverage_before_fill[col] = float(out[col].notna().mean())
            candidate_cols.extend(val_cols)

    if "market_regime" in requested:
        market = _load_market_regime(open_dir)
        if not market.empty:
            merged = _merge_market_asof(out, market, max_lag_days=market_max_lag_days)
            for col in merged.columns:
                out[col] = merged[col].to_numpy()
            mkt_cols = [c for c in merged.columns if c.startswith("od_mkt_") or c == "od_market_lag_days"]
            out["od_market_available"] = out[mkt_cols].notna().any(axis=1).astype(float)
            mkt_cols.append("od_market_available")
            for col in mkt_cols:
                group_by_feature[col] = "market_regime"
                coverage_before_fill[col] = float(out[col].notna().mean())
            candidate_cols.extend(mkt_cols)

    if "fund_flow" in requested:
        flow = _load_fund_flow(open_dir)
        if not flow.empty:
            merged = _merge_stock_asof(out, flow, feature_group="flow", max_lag_days=fund_flow_max_lag_days)
            for col in merged.columns:
                out[col] = merged[col].to_numpy()
            flow_cols = [c for c in merged.columns if c.startswith("od_flow_")]
            out["od_flow_available"] = out[flow_cols].notna().any(axis=1).astype(float)
            flow_cols.append("od_flow_available")
            for col in flow_cols:
                group_by_feature[col] = "fund_flow"
                coverage_before_fill[col] = float(out[col].notna().mean())
            candidate_cols.extend(flow_cols)

    candidate_cols = list(dict.fromkeys(candidate_cols))
    coverage_drops = {
        col: f"coverage_below_min:{coverage_before_fill.get(col, 0.0):.3f}"
        for col in candidate_cols
        if coverage_before_fill.get(col, 0.0) < min_coverage and not col.endswith("_available")
    }
    prunable_cols = [c for c in candidate_cols if c not in coverage_drops]

    for col in candidate_cols:
        out[col] = _fill_feature(out, col)

    selected, corr_drops = correlation_prune(out, prunable_cols, threshold=corr_threshold)
    drop_reasons = {**coverage_drops, **corr_drops}
    report = _make_report(out, group_by_feature, candidate_cols, selected, drop_reasons, coverage_before_fill)
    return out, selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--open-dir", default=str(DEFAULT_OPEN_DIR))
    parser.add_argument("--groups", nargs="+", default=list(OPEN_GROUPS), choices=list(OPEN_GROUPS) + ["all"])
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--min-coverage", type=float, default=0.55)
    parser.add_argument("--min-date", default=None)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--out-panel", default=None, help="Optional parquet path for the enriched panel")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    panel = build_features(prices, index_df)
    if args.min_date:
        panel = panel[panel["date"] >= _parse_date(args.min_date)]
    if args.max_date:
        panel = panel[panel["date"] <= _parse_date(args.max_date)]

    enriched, selected, report = add_stage2_open_features(
        panel,
        open_dir=args.open_dir,
        groups=args.groups,
        corr_threshold=args.corr_threshold,
        min_coverage=args.min_coverage,
    )

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    if args.out_panel:
        out_path = Path(args.out_panel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(out_path, index=False)

    print(f">> panel rows: {len(enriched):,}, dates {enriched['date'].min().date()} to {enriched['date'].max().date()}")
    print(f">> selected open features: {len(selected)}")
    for group in OPEN_GROUPS:
        group_selected = [c for c in selected if c in set(report.loc[report['group'] == group, 'feature'])]
        if group_selected:
            print(f"   {group}: {len(group_selected)}")
    print(f">> wrote quality report: {report_path}")
    if selected:
        print(">> selected columns:")
        for col in selected:
            print(f"   {col}")


if __name__ == "__main__":
    main()
