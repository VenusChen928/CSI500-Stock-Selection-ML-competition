# Stage2 Report Layout

This folder contains active Stage2 report materials and reproducibility
artifacts.  The actual portfolio to upload is the root-level
`stage2_final_portfolio.csv`.

## Active Folders

- `final_report_materials/`: final score summaries, leakage audits, top-k
  sanity check, metadata, and report-writing index.
- `backtests/`: saved per-window portfolio CSVs from the final 12-window
  validation run, adaptive-route meta CSVs, and a README explaining how to
  regenerate them.

Rejected variants and large transient backtest outputs were removed from the
clean submission layout; the retained report evidence is in
`final_report_materials/`, while `backtests/` keeps the final per-window
portfolio records and route metadata used by the self-test.

Active model helper code lives at the repository root, not inside this report
folder.  In particular, `lightgbm_portfolio.py` and
`tuned_xgboost_portfolio.py` are root-level helpers used by
`stage2_tree_consensus.py`.
