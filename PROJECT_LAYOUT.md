# Project Layout

This repository is now trimmed for the Stage2 submission.  The root directory
contains active data, feature, model, scoring, and validation scripts.  The
single upload CSV lives at `submissions/portfolio.csv`.  Historical experiments
are archived under `history/`.

## Root

- `requirements.txt`: Python package requirements for `mlcomp-sp26`.
- `download_data.py`: official data updater.
- `features.py`: shared OHLCV feature engineering.
- `baseline_xgboost.py`: official-style baseline and shared portfolio helpers.
- `lstm_rank_weight.py`: retained sequence learner used by active historical
  route code.
- `stage2_baseline_guard_ensemble.py`: current final Stage2 generator.
- `stage2_weekly_consensus_ensemble.py`: weekly consensus sub-route.
- `stage2_weekly_alpha_overlay.py`: weekly alpha overlay sub-route.
- `stage2_weekly_cycle_tree.py`: full-week cycle tree sub-route.
- `stage2_tree_consensus.py`: LightGBM/XGBoost tree consensus sub-route.
- `stage2_hybrid_gate.py`: earlier active hybrid route kept for reproducible
  comparison.
- `stage2_meta_portfolio_ensemble.py`: earlier active meta route kept for
  reproducible comparison.
- `stage2_regularized_consensus.py`: regularized challenger route retained for
  ablations.
- `score_submission.py`: realized-return scoring script.
- `validate_submission.py`: official submission format/rule validator.
- `stage2_backtest_5day.py`: slower but comprehensive five-trading-day
  multi-window evaluator.
- `stage2_fast_backtest_numpy.py`: faster parallel evaluator for repeated
  Stage2 checks.
- `stage2_leakage_audit.py`: full-data versus as-of-truncated leakage audit.
- `stage2_validation_audit.py`: additional route validation/audit utility.

## Active Directories

- `data/`: current official data cache used by the final generator.
- `submissions/portfolio.csv`: the only final Stage2 upload candidate.
- `submissions/stage2/final_report_materials/`: final metadata, validation,
  leakage, performance, and portfolio-shape evidence for the written report.
- `common/`: shared legacy scripts still imported by active Stage2 code.
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
