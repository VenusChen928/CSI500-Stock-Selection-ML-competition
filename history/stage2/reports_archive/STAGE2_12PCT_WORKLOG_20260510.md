# Stage2 12% Worklog - 2026-05-10

Goal: improve five-trading-day multi-window mean excess toward `+12%` while
keeping leakage and overfit checks explicit.

## Current Anchor

- Active route: `stage2_meta_portfolio_ensemble.py`
- Current 12-window result:
  mean excess `+8.098%`, min `+3.799%`, max `+16.841%`, negatives `0 / 12`.
- Recent monitor `as_of=20260428`:
  portfolio `+16.284%`, benchmark `+5.975%`, excess `+10.309%`.
- Leakage audit passed:
  `stage2_leakage_audit_meta_portfolio_ensemble_adaptive_power_20260510.csv`.

## Cleaning

- Deleted Python `__pycache__` directories.
- Archived pre-12%-round reports under
  `submissions/stage2/reports/archive_pre_12pct_20260510/`.
- Archived old probe/candidate backtests under
  `submissions/stage2/backtests/archive_pre_12pct_20260510/`.
- Archived rejected probe scripts under
  `archive/rejected_stage2_scripts_20260510/`.

## Research Takeaways

- Ranking and cross-sectional ordering matter more than raw point prediction.
- Regime-aware walk-forward validation is safer than one global recipe.
- Graph/relationship models are promising but need enough data and careful
  regularization; a quick deep model is not automatically better.
- Feature redundancy should be controlled, but blindly removing rank/raw pairs
  previously hurt this project.

## New Probes

- Dynamic factor portfolio:
  selected a factor only from historical realized windows before each `as_of`.
  Rejected because the 20260428 selection history had mean only `+0.554%` with
  `10 / 24` negative historical windows.
- Forest rank portfolio:
  used de-correlated rank-normalized features with ExtraTrees/RandomForest.
  Rejected because validation IC was `0.0018` and 20260428 known-window excess
  was `-0.218%`.
- Current-pool reweight sweep:
  tested low-vol, low-downside-vol, low-market-correlation, low-amount-spike,
  trend-quality, OBV, intraday, and residual-return tilts on the active 30-name
  pool. Rejected because best variants did not beat the active baseline.
- Cached stock-pool catalog:
  rescored 676 cached submissions across 51 model prefixes. The best
  all-cache oracle excluding the active meta reweighting was only `+6.699%`
  mean, confirming the bottleneck is new stock-pool quality rather than simply
  mixing old candidates.

## Decision

Do not promote any new model from this round. Keep the current `+8.098%` route
as active best. The next viable path to `+12%` needs a genuinely new stock-pool
generator, likely one of:

- relationship-aware graph/cluster alpha using historical correlation groups;
- a properly regularized pairwise/listwise ranker with walk-forward IC checks;
- benchmark-aware residual alpha model that improves stock pool, not only
  weight concentration;
- carefully cleaned external data joined by stock/date with strict ablation.
