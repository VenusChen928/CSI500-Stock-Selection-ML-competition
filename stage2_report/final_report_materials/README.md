# Stage2 Final Report Materials

This folder contains the compact evidence set for the final Stage2 write-up.
It is intentionally small and ordered by report-writing flow.

## Files

- `00_stage2_final_report_summary.md`: full Stage2 report narrative, including
  objective, data, features, models, validation, final route, and self-tests.
- `01_final_portfolio_metadata.csv`: final portfolio route and training
  metadata for `stage2_final_portfolio.csv`.
- `02_full_week_12_window_performance_summary.csv`: aggregate 12-window
  performance versus the original XGBoost baseline.
- `03_full_week_12_window_performance_detail.csv`: per-window performance
  details.
- `04_final_topk_and_weighting_check.md`: why the final route keeps 30
  rank-weighted names.
- `05_final_leakage_audit_dynamic.csv`: full-data versus as-of-truncated
  dynamic leakage audit.
- `06_final_leakage_audit_static_scan.csv`: static scan for leakage-prone code
  references.
- `07_stage2_experiment_history.md`: chronological optimization log, rejected
  directions, window-definition correction, and final decision trail.

Older redundant checks are archived in
`history/stage2/reports_archive/redundant_final_checks_20260510/`.
