# Stage2 Reports

This folder contains the final compact reports kept near the submission.

## Current Reports

- `stage2_guard_route_v2_nocache_fullweek12_20260510_summary.csv`: aggregate
  full-week 12-window comparison.
- `stage2_guard_route_v2_nocache_fullweek12_20260510_summary_detail.csv`:
  per-window details for that comparison.
- `portfolio_meta_20260508.csv`: metadata for the final root `portfolio.csv`.
- `stage2_final_topk_sanity_20260510.md`: final top-k sanity check explaining
  why the submitted route keeps 30 names for the `2026-05-08` regime.
- `stage2_leakage_audit_baseline_guard_adaptive_route_v2_20260510.csv`:
  dynamic full-data versus truncated-data leakage audit.
- `stage2_leakage_static_scan_route_v2_20260510.csv`: static source-pattern
  scan for leakage-prone references.
- `stage2_leakage_audit_final_layout_20260510.csv`: same dynamic audit rerun
  after final directory cleanup.
- `stage2_leakage_static_scan_final_layout_20260510.csv`: static scan rerun
  after final directory cleanup.
- `stage2_leakage_audit_final_single_portfolio_20260510.csv`: final rerun
  after duplicate portfolio CSVs were archived.
- `stage2_leakage_static_scan_final_single_portfolio_20260510.csv`: matching
  final static scan.

Older reports are archived under `history/stage2/reports_archive/`.
