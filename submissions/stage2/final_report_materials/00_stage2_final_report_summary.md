# Stage2 Final Report Summary

## Final Submission

- Upload file: `submissions/portfolio.csv`
- As-of date: `2026-05-08`
- Data cache max date: `2026-05-08`
- Final generator: `stage2_baseline_guard_ensemble.py`
- Final selected route: `baseline_xgb`
- Guard reason: `baseline_guard_overheated_high_breadth`
- Holdings: 30 stocks
- Max weight: 6.452%
- Effective number of names: 22.87

## Model Design

The final route is a baseline-guarded Stage2 ensemble.  It first computes
as-of-observable market regime statistics, then chooses among:

- official-style XGBoost baseline fallback;
- LightGBM/XGBoost tree consensus;
- weekly alpha overlay;
- full-week cycle tree route;
- broad defensive tilt.

For `2026-05-08`, the short- and medium-term market regime was overheated with
high breadth, so the guard selected the official-style XGBoost baseline route
with 30 rank-weighted names.  This is deliberately conservative: the strongest
more complex routes were useful in other regimes, but this regime favored the
baseline fallback in leakage-safe historical checks.

## Validation Result

Full-week 12-window comparison:

| model | mean excess | min excess | max excess | negative windows |
|---|---:|---:|---:|---:|
| baseline-guarded ensemble | +4.001% | +0.907% | +12.308% | 0 / 12 |
| original XGBoost baseline | +0.527% | -1.221% | +2.916% | 5 / 12 |

Detailed numbers are in:

- `02_full_week_12_window_performance_summary.csv`
- `03_full_week_12_window_performance_detail.csv`

## Portfolio Shape

The final top-k sanity check compared top30/top35/top40/top50/top60 on
historical baseline-routed windows.  Top30 had the best mean and minimum excess
among these checks, so the final portfolio keeps 30 names instead of diluting
the rank signal.

Evidence file:

- `04_final_topk_and_weighting_check.md`

## Leakage And Overfitting Controls

The final dynamic leakage audit compares portfolio weights generated from full
data versus data truncated to the as-of date.  The final rerun passed on
`2026-01-23`, `2026-03-20`, and `2026-05-08` with:

- `max_abs_diff = 0`
- `l1_diff = 0`
- `changed_names = 0`

Evidence files:

- `05_final_leakage_audit_dynamic.csv`
- `06_final_leakage_audit_static_scan.csv`

## Reproduce

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out submissions/portfolio.csv \
  --meta-out submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv
python validate_submission.py submissions/portfolio.csv
```
