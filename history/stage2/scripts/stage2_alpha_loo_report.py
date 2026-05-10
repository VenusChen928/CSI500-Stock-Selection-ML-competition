"""Leave-one-window-out audit for stage2 hybrid alpha variants."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_detail(path: Path, model: str | None, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if model and "model" in df.columns:
        df = df[df["model"] == model].copy()
    keep = ["as_of", "start", "end", "portfolio_return", "benchmark_return", "excess_return"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[keep].copy()
    out["as_of"] = out["as_of"].astype(str)
    out = out.rename(columns={"excess_return": label})
    return out[["as_of", label]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-detail",
        default="submissions/stage2/reports/stage2_hybrid_alpha_ablation_12w_20260509_summary_detail.csv",
    )
    parser.add_argument(
        "--final-detail",
        default="submissions/stage2/reports/stage2_hybrid_gate_secondary_alpha_12w_20260509_summary_detail.csv",
    )
    parser.add_argument("--out", default="submissions/stage2/reports/stage2_alpha_loo_audit_20260509.csv")
    parser.add_argument("--summary-md", default="submissions/stage2/reports/stage2_alpha_loo_audit_20260509.md")
    args = parser.parse_args()

    ablation_path = Path(args.ablation_detail)
    final_path = Path(args.final_detail)
    variants = {
        "no_alpha": ("hybrid_gate_no_alpha", ablation_path),
        "no_regime": ("hybrid_gate_no_regime", ablation_path),
        "no_liquidity": ("hybrid_gate_no_liquidity", ablation_path),
        "no_secondary": ("hybrid_gate_no_secondary", ablation_path),
        "no_route": ("hybrid_gate_no_route", ablation_path),
        "final": (None, final_path),
    }

    merged: pd.DataFrame | None = None
    for label, (model, path) in variants.items():
        part = load_detail(path, model, label)
        merged = part if merged is None else merged.merge(part, on="as_of", how="inner")
    if merged is None or merged.empty:
        raise RuntimeError("no variant rows loaded")

    variant_cols = list(variants)
    rows = []
    for heldout in merged["as_of"].tolist():
        train = merged[merged["as_of"] != heldout]
        test = merged[merged["as_of"] == heldout].iloc[0]
        train_means = train[variant_cols].mean().sort_values(ascending=False)
        selected = str(train_means.index[0])
        rows.append(
            {
                "heldout_as_of": heldout,
                "selected_by_other_windows": selected,
                "selected_train_mean": float(train_means.iloc[0]),
                "final_train_mean": float(train["final"].mean()),
                "heldout_selected_excess": float(test[selected]),
                "heldout_final_excess": float(test["final"]),
                "final_rank_on_train": int(list(train_means.index).index("final") + 1),
                "final_would_be_selected": selected == "final",
                **{f"heldout_{c}": float(test[c]) for c in variant_cols},
            }
        )

    loo = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loo.to_csv(out_path, index=False)

    summary = merged[variant_cols].agg(["mean", "min", "max"]).T
    summary["negative_windows"] = (merged[variant_cols] < 0).sum()
    summary = summary.sort_values(["mean", "min"], ascending=False)

    # Compare each ablation with final. Positive delta means the final alpha
    # layer combination improved that held-out window.
    deltas = []
    for col in variant_cols:
        if col == "final":
            continue
        diff = merged["final"] - merged[col]
        deltas.append(
            {
                "comparison": f"final_minus_{col}",
                "mean_delta": float(diff.mean()),
                "min_delta": float(diff.min()),
                "negative_delta_windows": int((diff < 0).sum()),
            }
        )
    delta_summary = pd.DataFrame(deltas).sort_values("mean_delta", ascending=False)

    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Stage2 Alpha LOO Audit\n\n")
        f.write("## Variant Summary\n\n")
        f.write(summary.to_markdown(floatfmt=".6f"))
        f.write("\n\n## Final vs Ablations\n\n")
        f.write(delta_summary.to_markdown(index=False, floatfmt=".6f"))
        f.write("\n\n## Leave-One-Window-Out Selection\n\n")
        f.write(loo[[
            "heldout_as_of",
            "selected_by_other_windows",
            "final_rank_on_train",
            "heldout_selected_excess",
            "heldout_final_excess",
            "final_would_be_selected",
        ]].to_markdown(index=False, floatfmt=".6f"))
        f.write("\n")

    print("VARIANT SUMMARY")
    print(summary.to_string())
    print("\nFINAL VS ABLATIONS")
    print(delta_summary.to_string(index=False))
    print("\nLOO")
    print(loo[[
        "heldout_as_of",
        "selected_by_other_windows",
        "final_rank_on_train",
        "heldout_selected_excess",
        "heldout_final_excess",
        "final_would_be_selected",
    ]].to_string(index=False))
    print(f">> wrote {out_path}")
    print(f">> wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
