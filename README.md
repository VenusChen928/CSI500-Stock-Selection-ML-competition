# CSI500 Stage2 Portfolio Submission

This repository is organized for the Stage2 five-trading-day submission.

Upload exactly this file:

```text
stage2_final_portfolio.csv
```

Stage1 and Stage2 materials are separated so the final upload file is
unambiguous.

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
  --out stage2_final_portfolio.csv \
  --meta-out stage2_report/final_report_materials/01_final_portfolio_metadata.csv
python validate_submission.py stage2_final_portfolio.csv
```

## Final Model Route

`stage2_baseline_guard_ensemble.py` is the 12-window best route, named
`baseline_guard_adaptive` in the validation reports.  It uses only data up to
the requested `as_of` date and combines:

- official XGBoost-style baseline fallback;
- LightGBM/XGBoost tree consensus;
- weekly alpha overlay;
- full-week cycle tree route;
- meta/hybrid/LSTM fallback used by the weekly alpha base when no cache is
  provided;
- as-of-observable regime guards.

The guard chooses the safest route for the current market regime and does not
read realized future returns or cached historical score results.  For
`as_of=20260508`, the route falls back to the official-style XGBoost baseline
with 30 rank-weighted names.  A final top-k sanity check kept top30 because the
closest historical baseline-guard windows showed better mean and minimum excess
than top35/top40/top50/top60.

The full-week validation summary used for the final choice is:

```text
baseline_guard_adaptive: mean excess +4.001%, min +0.907%, 0 negative windows / 12
baseline_xgb:            mean excess +0.527%, min -1.221%, 5 negative windows / 12
```

## Validation Commands

Run official format validation:

```bash
python validate_submission.py stage2_final_portfolio.csv
```

Run a leakage audit:

```bash
python tools/stage2_validation/stage2_leakage_audit.py \
  --as-of 20260109 20260227 20260313 20260327 20260410 20260508 \
  --models baseline_guard_adaptive \
  --out stage2_report/final_report_materials/05_final_leakage_audit_dynamic.csv \
  --static-out stage2_report/final_report_materials/06_final_leakage_audit_static_scan.csv
```

Run full-week comparison:

```bash
python tools/stage2_validation/stage2_backtest_5day.py \
  --models baseline_xgb baseline_guard_adaptive \
  --full-week-only \
  --windows 12 \
  --jobs 4 \
  --out-dir stage2_report/backtests/final_check \
  --summary-out stage2_report/final_report_materials/02_full_week_12_window_performance_summary.csv
```

## Report Materials

The final write-up materials are organized in:

```text
stage2_report/final_report_materials/
```

Start with `00_stage2_final_report_summary.md` for the main narrative and
`07_stage2_experiment_history.md` for the detailed experiment trail.

Stage1 materials are archived separately in `stage1_report/` and are not part
of the Stage2 submission path.
