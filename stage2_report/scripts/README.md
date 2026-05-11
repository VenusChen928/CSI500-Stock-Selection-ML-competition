# Stage2 Helper Scripts

These scripts are imported by active Stage2 root routes but are not final
portfolio entry points.

- `lightgbm_portfolio.py`: validation-shaped LightGBM helper used by
  `stage2_tree_consensus.py`.
- `tuned_xgboost_portfolio.py`: validation-shaped XGBoost helper used by
  `stage2_tree_consensus.py`.

Run final reproduction from the repository root with
`stage2_baseline_guard_ensemble.py`; do not upload outputs generated directly
from this helper folder.
