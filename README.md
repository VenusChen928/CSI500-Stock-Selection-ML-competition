# CSI500 Stage2 Portfolio Submission

This repository is organized for the Stage2 five-trading-day submission.

Upload exactly this file:

```text
submissions/portfolio.csv
```

All duplicate portfolio CSVs were moved to `history/` so the final upload file
is unambiguous.

## Environment

```bash
conda activate mlcomp-sp26
pip install -r requirements.txt
```

The project was last checked with data through `2026-05-08` in
`data/prices.parquet`.

## Submission Rules

The competition validator expects a UTF-8 CSV with exactly two columns:

```text
stock_code,weight
000001,0.0333
600000,0.0250
```

The active constraints are: current CSI500 universe only, at least 30 positive
weight stocks, non-negative weights summing to `1.0 +/- 1e-4`, and max single
stock weight `0.10`.

## Reproduce Final Portfolio

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out submissions/portfolio.csv \
  --meta-out submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv
python validate_submission.py submissions/portfolio.csv
```

## Final Model Route

`stage2_baseline_guard_ensemble.py` is the final route.  It uses only data up
to the requested `as_of` date and combines:

- official XGBoost-style baseline fallback;
- LightGBM/XGBoost tree consensus;
- weekly alpha overlay;
- full-week cycle tree route;
- as-of-observable regime guards.

The guard chooses the safest route for the current market regime and does not
read realized future returns or cached historical score results.  For
`as_of=20260508`, the route falls back to the official-style XGBoost baseline
with 30 rank-weighted names.  A final top-k sanity check kept top30 because the
closest historical baseline-guard windows showed better mean and minimum excess
than top35/top40/top50/top60.

## Validation Commands

Run official format validation:

```bash
python validate_submission.py submissions/portfolio.csv
```

Run a leakage audit:

```bash
python stage2_leakage_audit.py \
  --as-of 20260109 20260227 20260313 20260327 20260410 20260508 \
  --models baseline_guard_adaptive \
  --out submissions/stage2/final_report_materials/05_final_leakage_audit_dynamic.csv \
  --static-out submissions/stage2/final_report_materials/06_final_leakage_audit_static_scan.csv
```

Run full-week comparison:

```bash
python stage2_backtest_5day.py \
  --models baseline_xgb baseline_guard_adaptive \
  --full-week-only \
  --windows 12 \
  --jobs 4 \
  --out-dir submissions/stage2/backtests/final_check \
  --summary-out submissions/stage2/final_report_materials/02_full_week_12_window_performance_summary.csv
```

Historical reports are archived under `history/stage2/reports_archive/`.
