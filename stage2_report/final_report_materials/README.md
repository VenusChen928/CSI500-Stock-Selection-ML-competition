# Stage2 Final Report Materials

This folder contains the compact evidence set for the final Stage2 write-up.
It is intentionally small and ordered by report-writing flow.

## Files

- `00_stage2_final_report_summary.md`: compact Stage2 report narrative kept as
  a concise companion to root-level `final_report.md`.
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
- `08_full_week_12_window_ic_detail.csv`: per-window allocation IC diagnostics
  used in the expanded final report.
- `09_full_week_12_window_ic_summary.csv`: aggregate allocation IC summary.
- `10_full_week_gate_route_detail.csv`: as-of regime gate route selected for
  each complete full-week self-test window.

The root-level `final_report.md` is the expanded Markdown report source that
combines these materials into the final narrative.

Per-window generated portfolio CSVs from the final 12-window self-test are
kept under `stage2_report/backtests/`, with companion adaptive-route meta CSVs.
`03_full_week_12_window_performance_detail.csv` preserves the numeric results
and links each row to the corresponding portfolio file.  To refresh those CSVs,
use the full-week self-test command in the root `README.md`.  To refresh
allocation IC after a backtest rerun, use
`tools/stage2_validation/stage2_allocation_ic.py`.
