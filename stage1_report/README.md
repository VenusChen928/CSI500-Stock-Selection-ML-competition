# Stage1 Report Archive

This folder keeps the Stage1 materials separate from the active Stage2
submission path.  Stage1 used a 3-trading-day target, so these scripts and CSVs
are retained only for report traceability and comparison.

## Layout

- `stage1_final_portfolio.csv`: the single retained Stage1 submission CSV,
  restored from the archived guarded-ensemble final route.
- `scripts/`: reproducible Stage1 model routes that were discussed in the
  report trail.
- `report_materials/`: compact CSV evidence used to summarize Stage1
  experiments.

## Validate Stage1 Final Submission

Run from the repository root:

```bash
python validate_submission.py stage1_report/stage1_final_portfolio.csv
```

The scripts are retained only to explain the Stage1 experiment path; the
retained Stage1 submission itself is the single CSV above.

Historical note: the archived Stage1 README identified
`stage1_guarded_ensemble_20260430.csv` as the submitted route and
`best_current_submission.csv` as its mirror.  This folder keeps that guarded
ensemble portfolio under the simpler final filename above.

Stage1 files are intentionally not used by the final Stage2 portfolio builder.
