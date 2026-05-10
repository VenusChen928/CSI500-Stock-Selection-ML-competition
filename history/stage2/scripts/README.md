# Archived Stage2 Scripts

These scripts were moved out of the root because they are not the promoted
Stage2 route.  They remain available as evidence for the report and as
reproducible historical experiments.

Extra root scripts from the final cleanup were archived under
`archived_root_extras_20260510/`.

The active final entry point is:

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --out submissions/portfolio.csv \
  --meta-out submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv
```
