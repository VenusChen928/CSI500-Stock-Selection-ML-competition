# Stage1 Report Archive

This folder keeps the Stage1 materials separate from the active Stage2
submission path.  Stage1 used a 3-trading-day target, so these scripts and CSVs
are retained only for report traceability and comparison.

## Layout

- `final_portfolios/`: retained Stage1 final/candidate portfolios.
- `scripts/`: reproducible Stage1 model routes that were discussed in the
  report trail.
- `report_materials/`: compact CSV evidence used to summarize Stage1
  experiments.

## Reproduce A Stage1 Candidate

Run from the repository root:

```bash
python stage1_report/scripts/stage1_guarded_ensemble.py \
  --as-of 20260430 \
  --horizon 3 \
  --out stage1_report/generated/stage1_guarded_ensemble_20260430.csv
python validate_submission.py stage1_report/final_portfolios/stage1_guarded_ensemble_20260430.csv
```

Stage1 files are intentionally not used by the final Stage2 portfolio builder.
