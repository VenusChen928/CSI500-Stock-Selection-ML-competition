# Submission Folder Layout

Root-level file:

- `gated_layered_lstm_20260421.csv`: current production submission for the latest full 5-trading-day window.  It selects layered LSTM only when validation mean and validation worst-window evidence are both strong; otherwise it falls back to tuned XGB.

Folders:

- `current_best/`: copies of the current gated/layered LSTM submission and the previous LSTM/hybrid best for safe reference.
- `benchmarks/`: baseline and tuned-XGB comparison submissions.
- `backtests/`: generated submissions from rolling-window model comparisons.
- `reports/`: score summaries, ablation tables, and backtest comparison CSVs.
- `experiments/`: representative experiment outputs that may still be useful.
- `archive_experiments/`: older experiments and compact historical records.

The noisy per-parameter temporary grids from the open-data blending/reweighting
round were removed; only the compact grid summary is retained in the archive.

Current best score window:

- As-of: `2026-04-21`
- Evaluation: `2026-04-22` to `2026-04-28`
- Portfolio return: `+3.354%`
- Benchmark return: `-0.802%`
- Excess return: `+4.156%`

Recent 9-window backtest:

- Gated layered LSTM mean excess: `+1.668%`
- Layered LSTM mean excess: `+0.766%`
- Tuned XGB mean excess: `+0.724%`
- Pure LSTM mean excess: `+0.658%`
- Baseline mean excess: `-0.151%`

Gate threshold note:

- Production gate uses `layer_mean >= 0.010` and `layer_min >= -0.020`.
- This improved the 9-window mean from the previous gate's `+1.491%` to
  `+1.668%`, while reducing worst-window excess from `-2.771%` to `-0.853%`.

Fund-flow experiment note:

- Added AKShare stock fund-flow data under `data/open/stock_fund_flow.parquet`.
- Added fund-flow features to `open_data_features.py`, but fund-flow portfolio
  layers are experimental and require `--include-flow-policies`.
- Recent 5-window test with fund-flow layers improved over baseline
  (`+0.913%` mean excess vs `-0.655%`) but did not beat the validated production
  route, so the default production scripts intentionally keep fund-flow layers
  disabled.
- Direct open-data XGB ranker/regressor and lightweight Transformer challengers
  were tested as research models.  Both showed validation-window overfit on the
  latest window and are retained under `tools/research_models/` rather than
  promoted to production.
