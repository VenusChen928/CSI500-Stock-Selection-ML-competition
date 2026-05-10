# Tools Layout

- `backtests/`: rolling-window evaluation scripts.  `backtest_layered_compare.py`
  is the current multi-window runner for baseline, tuned XGB, LSTM, and layered
  LSTM.
- `reports/`: analysis scripts that create ablation or diagnostic CSVs.
- `research_models/`: models that were useful for learning but are not part of
  the current production route, including the old hybrid LSTM/XGB, direct
  open-data experiments, and the lightweight Transformer rank/weight challenger.
  They are retained for reference rather than kept in the project root.
- `utilities/`: small helpers for manipulating submissions.

The project root is reserved for the active data pipeline, scoring helpers, and
current production model route.
