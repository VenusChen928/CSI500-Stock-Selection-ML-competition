# Stage1 Archive

Stage1 targeted the 3-trading-day evaluation window. It is now archived so the
root project can focus on stage2.

## Final Route

The submitted route was the guarded ensemble:

```bash
/opt/anaconda3/envs/mlcomp-sp26/bin/python stage1_guarded_ensemble.py \
  --as-of 20260430 \
  --horizon 3 \
  --out submissions/stage1/final/stage1_guarded_ensemble_20260430.csv
```

## Important Files

- `final/stage1_guarded_ensemble_20260430.csv`: final stage1 submission.
- `final/best_current_submission.csv`: mirror of the final stage1 submission.
- `final/stage1_lstm_lgb_confidence_risk_balanced_20260430.csv`: strong direct
  LSTM + LightGBM confidence candidate.
- `reports/`: recent-window summaries and policy tables used to choose the
  final route.
- `backtests/`: generated portfolios from the stage1 multi-window tests.
- `experiments/`: non-final model trials retained for report evidence.
