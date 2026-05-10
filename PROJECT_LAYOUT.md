# Project Layout

This repository is trimmed for the Stage2 submission.  The root directory keeps
the complete model chain required to build the final portfolio, plus official
download/scoring/format validation scripts.  Multi-window research utilities
live under `tools/`, and historical experiments are archived under `history/`.

## Root

- `requirements.txt`: Python package requirements for `mlcomp-sp26`.
- `download_data.py`: official data updater.
- `features.py`: shared OHLCV feature engineering.
- `baseline_xgboost.py`: official-style baseline and shared portfolio helpers.
- `lstm_rank_weight.py`: sequence learner used by the active hybrid fallback
  chain.
- `stage2_baseline_guard_ensemble.py`: current final Stage2 generator.
- `stage2_weekly_consensus_ensemble.py`: weekly consensus sub-route.
- `stage2_weekly_alpha_overlay.py`: weekly alpha overlay sub-route.
- `stage2_weekly_cycle_tree.py`: full-week cycle tree sub-route.
- `stage2_tree_consensus.py`: LightGBM/XGBoost tree consensus sub-route.
- `stage2_hybrid_gate.py`: hybrid fallback used by the meta route.
- `stage2_meta_portfolio_ensemble.py`: meta portfolio base used by weekly
  alpha when no cache is supplied.
- `stage2_regularized_consensus.py`: regularized tree/CatBoost route used by
  the hybrid fallback.
- `score_submission.py`: realized-return scoring script.
- `validate_submission.py`: official submission format/rule validator.

## Active Directories

- `data/`: current official data cache used by the final generator.
- `submissions/portfolio.csv`: the only final Stage2 upload candidate.
- `submissions/stage2/final_report_materials/`: final metadata, validation,
  leakage, performance, and portfolio-shape evidence for the written report.
- `common/`: LightGBM/XGBoost helper scripts imported by the active tree
  consensus route.
- `tools/stage2_validation/`: multi-window backtest and leakage-audit scripts
  used for final verification, kept out of the root model chain.
- `history/common/`: data backups, old IDE config, old utilities, and generic
  legacy reports.
- `history/stage1/`: Stage1 scripts, submissions, and reports.
- `history/stage2/`: Stage2 experiments, rejected scripts, old reports, and
  old generated portfolios.

## Final Reproduction

```bash
conda activate mlcomp-sp26
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out submissions/portfolio.csv \
  --meta-out submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv
python validate_submission.py submissions/portfolio.csv
```

`submissions/portfolio.csv` is the only active portfolio CSV.  Older duplicates
are archived under `history/stage2/submissions/final_duplicate_portfolios_20260510/`.
