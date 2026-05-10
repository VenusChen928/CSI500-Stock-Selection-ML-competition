"""
Blend multiple stock-weight submission files and enforce portfolio constraints.

This is useful for testing hybrid portfolios after each base model has produced
its own stock_code,weight file.  The blend is linear in weights, then clipped to
the 10% single-name cap and renormalized.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_xgboost import MAX_WEIGHT, MIN_STOCKS


def apply_cap(weights: pd.Series) -> pd.Series:
    w = weights[weights > 0].copy()
    if len(w) < MIN_STOCKS:
        raise ValueError(f"blended portfolio has {len(w)} names; need at least {MIN_STOCKS}")
    w = w / w.sum()
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def read_submission(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    return df.set_index("stock_code")["weight"].astype(float)


def blend(paths: list[str], alphas: list[float]) -> pd.DataFrame:
    if len(paths) != len(alphas):
        raise ValueError("--inputs and --alphas must have the same length")
    total_alpha = sum(alphas)
    if total_alpha <= 0:
        raise ValueError("alphas must sum to a positive value")
    alphas = [a / total_alpha for a in alphas]

    combined = pd.Series(dtype=float)
    for path, alpha in zip(paths, alphas):
        weights = read_submission(path)
        combined = combined.add(weights * alpha, fill_value=0.0)
    combined = apply_cap(combined)
    return pd.DataFrame({"stock_code": combined.index, "weight": combined.values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = blend(args.inputs, args.alphas)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f">> wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
