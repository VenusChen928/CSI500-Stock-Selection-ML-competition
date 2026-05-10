# Project Layout

This repository is trimmed for the final Stage2 submission.  The root directory
contains only the compact portfolio-generation path plus scoring/validation
helpers.  The single upload CSV lives at `submissions/portfolio.csv`.
Historical experiments are archived under `history/`.

## Root

- `requirements.txt`: Python package requirements for `mlcomp-sp26`.
- `download_data.py`: official data updater.
- `features.py`: shared OHLCV feature engineering.
- `baseline_xgboost.py`: official-style baseline and shared portfolio helpers.
- `stage2_baseline_guard_ensemble.py`: final compact Stage2 generator.
- `score_submission.py`: realized-return scoring script.
- `validate_submission.py`: official submission format/rule validator.

## Active Directories

- `data/`: current official data cache used by the final generator.
- `submissions/portfolio.csv`: the only final Stage2 upload candidate.
- `submissions/stage2/final_report_materials/`: final metadata, validation,
  leakage, performance, and portfolio-shape evidence for the written report.
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
