# Project Layout

This repository is trimmed for the Stage2 submission.  The root directory keeps
the complete model chain required to build the final portfolio, plus official
download/scoring/format validation scripts.  Multi-window research utilities
live under `tools/`, while Stage1 and Stage2 report evidence are separated into
their own report folders.

## Root

- `requirements.txt`: Python package requirements for `mlcomp-sp26`.
- `download_data.py`: official data updater.
- `features.py`: shared OHLCV feature engineering.
- `baseline_xgboost.py`: official-style baseline and shared portfolio helpers.
- `tuned_xgboost_portfolio.py`: tuned XGBoost helper used by the Stage2 tree
  consensus branch.
- `lightgbm_portfolio.py`: LightGBM helper used by the Stage2 tree consensus
  branch.
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
- `final_report.md`: expanded Markdown source for the final written report,
  including factors, models, self-test, IC diagnostics, and analysis.

## Active Directories

- `data/`: current official data cache used by the final generator.
- `stage2_final_portfolio.csv`: the only final Stage2 upload candidate.
- `stage2_report/final_report_materials/`: final metadata, validation,
  leakage, performance, and portfolio-shape evidence for the written report.
- `stage2_report/backtests/`: saved per-window portfolio CSVs from the final
  12-window Stage2 self-test, plus adaptive-route meta CSVs and a README
  explaining how to regenerate them.
- `stage1_report/`: Stage1 final portfolio, scripts, and report evidence kept
  out of the active Stage2 path.
- `tools/stage2_validation/`: multi-window backtest and leakage-audit scripts
  used for final verification, kept out of the root model chain.

## Final Reproduction

```bash
conda activate mlcomp-sp26
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out stage2_final_portfolio.csv \
  --meta-out stage2_report/final_report_materials/01_final_portfolio_metadata.csv
python validate_submission.py stage2_final_portfolio.csv
```

`stage2_final_portfolio.csv` is the only active Stage2 portfolio CSV at the
repository root.
