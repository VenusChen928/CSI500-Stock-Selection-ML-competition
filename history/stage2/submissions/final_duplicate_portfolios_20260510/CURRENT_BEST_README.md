# Current Best Stage2 Portfolio

This folder contains the promoted Stage2 portfolio generated from
`stage2_baseline_guard_ensemble.py` with `as_of=20260508`.

## Files

- `stage2_current_best_20260508.csv`: promoted final candidate.
- `stage2_current_best_20260508_meta.csv`: promoted candidate metadata.
- `stage2_baseline_guard_route_v2_20260508.csv`: same final route under its
  explicit route-v2 name.
- `stage2_baseline_guard_route_v2_20260508_meta.csv`: route-v2 metadata.

The root `portfolio.csv` is copied from `stage2_current_best_20260508.csv` for
direct upload convenience.

## Reproduce

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out submissions/stage2/current_best/stage2_current_best_20260508.csv \
  --meta-out submissions/stage2/current_best/stage2_current_best_20260508_meta.csv
python validate_submission.py submissions/stage2/current_best/stage2_current_best_20260508.csv
```

## Final Checks

- Submission constraints: validated with `validate_submission.py`.
- Leakage audit: see `submissions/stage2/reports/`.
- Full-week 12-window report: see
  `submissions/stage2/reports/stage2_guard_route_v2_nocache_fullweek12_20260510_summary.csv`.
