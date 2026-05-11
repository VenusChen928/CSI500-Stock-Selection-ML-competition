"""Recompute allocation IC for saved Stage2 backtest portfolios.

The script reads the per-window detail CSV produced by `stage2_backtest_5day.py`
and evaluates whether portfolio weights are positively aligned with realized
stock returns in the same scoring windows.  It does not train or generate
portfolios, so it is safe to run after the backtest records are frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _stock_returns(
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    rows: list[tuple[str, float]] = []
    for code, df in prices.groupby("stock_code", sort=False):
        df = df.sort_values("date")
        in_window = df[(df["date"] >= start) & (df["date"] <= end)]
        if in_window.empty:
            rows.append((code, 0.0))
            continue

        before = df[df["date"] < start]
        entry = before["close"].iloc[-1] if not before.empty else in_window["open"].iloc[0]
        exit_ = in_window["close"].iloc[-1]
        if entry <= 0 or pd.isna(entry) or pd.isna(exit_):
            rows.append((code, 0.0))
        else:
            rows.append((code, float(exit_ / entry - 1.0)))
    return pd.Series(dict(rows), dtype=float)


def _rank_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    ranks = aligned.rank(method="average")
    if ranks.iloc[:, 0].nunique() < 2 or ranks.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(ranks.iloc[:, 0], ranks.iloc[:, 1])[0, 1])


def compute_ic(detail: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prices = prices.copy()
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"])
    universe = pd.Index(sorted(prices["stock_code"].unique()))

    for row in detail.itertuples(index=False):
        submission = ROOT / row.submission
        portfolio = pd.read_csv(submission, dtype={"stock_code": str})
        portfolio["stock_code"] = portfolio["stock_code"].astype(str).str.zfill(6)
        weights = portfolio.set_index("stock_code")["weight"].astype(float)

        realized = _stock_returns(prices, pd.Timestamp(row.start), pd.Timestamp(row.end))
        realized = realized.reindex(universe).fillna(0.0)
        all_weights = pd.Series(0.0, index=universe)
        all_weights.loc[weights.index.intersection(universe)] = weights.reindex(universe).fillna(0.0)

        selected_returns = realized.reindex(weights.index)
        rows.append({
            "model": row.model,
            "as_of": row.as_of,
            "start": row.start,
            "end": row.end,
            "portfolio_return": row.portfolio_return,
            "benchmark_return": row.benchmark_return,
            "excess_return": row.excess_return,
            "n_names": row.n_names,
            "allocation_ic_all_universe": _rank_corr(all_weights, realized),
            "allocation_ic_selected_only": _rank_corr(weights, selected_returns),
        })
    return pd.DataFrame(rows)


def summarize(ic_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in ic_detail.groupby("model", sort=False):
        rows.append({
            "model": model,
            "mean_excess": group["excess_return"].mean(),
            "median_excess": group["excess_return"].median(),
            "min_excess": group["excess_return"].min(),
            "max_excess": group["excess_return"].max(),
            "negative_windows": int((group["excess_return"] < 0).sum()),
            "mean_allocation_ic_all_universe": group["allocation_ic_all_universe"].mean(),
            "median_allocation_ic_all_universe": group["allocation_ic_all_universe"].median(),
            "mean_allocation_ic_selected_only": group["allocation_ic_selected_only"].mean(),
            "median_allocation_ic_selected_only": group["allocation_ic_selected_only"].median(),
            "mean_n_names": group["n_names"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail",
        default="stage2_report/final_report_materials/03_full_week_12_window_performance_detail.csv",
    )
    parser.add_argument("--prices", default="data/prices.parquet")
    parser.add_argument(
        "--ic-detail-out",
        default="stage2_report/final_report_materials/08_full_week_12_window_ic_detail.csv",
    )
    parser.add_argument(
        "--ic-summary-out",
        default="stage2_report/final_report_materials/09_full_week_12_window_ic_summary.csv",
    )
    args = parser.parse_args()

    detail = pd.read_csv(ROOT / args.detail)
    prices = pd.read_parquet(ROOT / args.prices)
    ic_detail = compute_ic(detail, prices)
    ic_summary = summarize(ic_detail)

    detail_out = ROOT / args.ic_detail_out
    summary_out = ROOT / args.ic_summary_out
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    ic_detail.to_csv(detail_out, index=False)
    ic_summary.to_csv(summary_out, index=False)
    print(f"wrote {detail_out}")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
