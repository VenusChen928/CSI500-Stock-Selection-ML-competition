# Stage2 Experiment History And Decision Log

This file records the actual optimization path.  It is intentionally narrative:
the project did not move in a straight line, and several high-looking results
were later rejected after better validation.

## A. External Reporting Reference

We reviewed the public reference repository
`JunjiaZhangMax/CSI500-Stock-Selection-Machine-Learning-final-project`
(`https://github.com/JunjiaZhangMax/CSI500-Stock-Selection-Machine-Learning-final-project`)
for reporting style, not for code reuse.  The useful reporting pattern was:

| report element | why it helps |
|---|---|
| objective and constraints first | makes the scoring target explicit |
| data and feature table | shows what information the model is allowed to use |
| model and target explanation | clarifies whether the model predicts return, rank, or weights |
| walk-forward validation | makes leakage and time ordering visible |
| portfolio construction section | separates prediction from weight allocation |
| experiment log table | records failed attempts instead of hiding them |
| key findings and limitations | explains why the final choice is reasonable |

We adapted that structure to our own Stage2 route.

## B. Chronological Optimization Log

| phase | experiment | observation | decision |
|---|---|---|---|
| 1 | Baseline analysis | `score_submission.py` scores a submitted portfolio using future realized stock returns minus benchmark return; model only creates stock weights | keep scoring and prediction separated |
| 2 | Official XGBoost baseline | stable reference, but weak mean excess and multiple negative windows | retain as fallback |
| 3 | Direct rank/weight learning | attempted to learn future rank and target weight directly | rejected because weight learning was unstable and sometimes degenerated toward equal-looking weights |
| 4 | Feature selection | tested core OHLCV, ranks, volatility, liquidity, trend, RSI | kept compact core features for production |
| 5 | LSTM sequence model | captured short-term bursts and looked strong on early rolling tests | rejected as standalone due regime instability |
| 6 | Layered/gated LSTM | combined LSTM with tree confirmation and confidence weights | useful for Stage1 exploration, not robust enough for final Stage2 |
| 7 | LightGBM route | faster tree model with similar tabular capacity | retained as tree-consensus ingredient |
| 8 | Tuned XGBoost route | stronger XGB parameters and portfolio layer | retained as tree-consensus ingredient |
| 9 | Open data | valuation, market regime, and fund-flow data were lagged, aligned, normalized, coverage-checked, and correlation-filtered | rejected because final mean/floor did not improve |
| 10 | Transformer/deep sequence consideration | data size and instability made deep models risky | not promoted |
| 11 | Stage1 three-day optimization | target horizon changed to 3 days; tested confidence score, LSTM/LightGBM consensus, top-k and weight sweeps | useful but overfit to Stage1-like horizon |
| 12 | Phase1 result review | our Stage1 return was below class average; high-scoring classmates likely used simpler regularized tree ensembles aligned to 3-day horizon | reset emphasis toward regularization and validation |
| 13 | CatBoost robust-loss challenger | Quantile/MAE losses tested as robust alternatives | not competitive enough |
| 14 | Regularized consensus | Ridge, LightGBM, XGBoost, CatBoost-style route with stricter feature filtering | useful in narrow regimes, weak globally |
| 15 | Drawdown/risk overlay | reweighted selected names by drawdown and volatility confidence | improved some route floors but not sufficient alone |
| 16 | Meta/hybrid gate | route-specific choice among tree, regularized, and LSTM routes | improved old rolling validation |
| 17 | Multi-route consensus | broad class-ensemble-inspired aggregation of candidate portfolios | high old rolling mean, later not accepted as final due window-definition issue |
| 18 | Full-workweek audit | discovered old 9/12-window tests were not strict Monday-Friday complete weeks | reset validation standard |
| 19 | Weekly alpha design | added Monday/Friday, month-start/end, post-gap reopening, weekly risk-appetite features | useful as an expert, not universally best |
| 20 | Final regime gate | choose among experts using only as-of market state | promoted as final Stage2 route |

## C. Important Mistakes We Found

### C.1 Misreading The Test Window

Early tests used "next five trading days" after each as-of date.  That is a
normal holding-period definition, but many windows crossed weekends or holidays.
Stage2 is better represented by uninterrupted full workweeks.  After correcting
to complete Monday-Friday windows, several previously strong results dropped
substantially.

The correction changed the project direction:

| before correction | after correction |
|---|---|
| optimize any five trading days | optimize complete workweek behavior |
| short burst features looked stronger | weekly/cycle/regime features became more relevant |
| old multi-route mean looked around `+8.7%` | final strict 12-window mean is `+4.001%` |
| validation was optimistic | validation became report-safe |

### C.2 Feature Stuffing

Adding more features often improved one window but hurt the floor.  Open data,
graph features, and broad experimental feature sets were useful diagnostics but
not production improvements.  The final pipeline therefore uses compact core
features in the fallback and richer features only inside specialized experts.

### C.3 LSTM Overconfidence

LSTM variants produced impressive local windows because they can capture
short-term serial structure.  But when the horizon moved between three-day,
generic five-day, and full-week five-day definitions, their performance shifted
too much.  LSTM remains in the fallback dependency chain, but not as the primary
submission route.

### C.4 Over-Optimization Of Portfolio Weights

We repeatedly tested top-k and max-weight changes.  More aggressive caps can
increase upside in one window but reduce stability.  The final top30 choice was
made only after checking similar baseline-guard windows, not by tuning directly
on the final as-of return.

## D. Model Families Tried

| model family | target tested | status |
|---|---|---|
| official XGBoost | raw 5-day return | final fallback |
| tuned XGBoost | raw/rank/excess variants | tree-consensus ingredient |
| LightGBM | raw/rank/excess variants | tree-consensus ingredient |
| CatBoost | quantile/MAE-style robust objective | rejected as standalone |
| Ridge/regularized consensus | rank/excess labels | narrow-regime fallback |
| LSTM rank/weight | future rank and portfolio confidence | fallback dependency only |
| layered/gated LSTM | LSTM plus confirmation gates | Stage1 exploration |
| weekly cycle tree | full-week excess return | final expert |
| weekly alpha overlay | flow/vol/OBV/gap confidence | final expert |
| regime-gated ensemble | as-of route selection | final production design |

## E. Feature Engineering Decisions

| decision | evidence |
|---|---|
| keep rank features with raw features | removing rank/raw pairs hurt tree routes |
| keep compact core for fallback | broad feature sets overfit and hurt floors |
| use target excess for weekly cycle experts | full-week expert should learn stock alpha relative to index |
| include calendar weekly features only in weekly route | they match full-week objective but are not universally useful |
| reject noisy open data | strict cleaning did not improve mean and min together |
| reject graph peer factors | standalone peer signals had weak robustness |
| avoid blind de-correlation | high correlation alone did not imply a useless tree feature |

## F. Final Validation Evidence

The final strict 12 full-week validation is:

| model | mean excess | min excess | max excess | negative windows |
|---|---:|---:|---:|---:|
| `baseline_guard_adaptive` | `+4.001%` | `+0.907%` | `+12.308%` | `0 / 12` |
| original `baseline_xgb` | `+0.527%` | `-1.221%` | `+2.916%` | `5 / 12` |

The per-window file is:

```text
03_full_week_12_window_performance_detail.csv
```

## G. Final Route For `2026-05-08`

The final route selected `baseline_xgb` for the final as-of date because the
market regime was overheated and broad:

| variable | value |
|---|---:|
| `idx_ret_5d` | `+5.975%` |
| `idx_ret_20d` | `+14.833%` |
| `breadth_ret_5d_pos` | `70.741%` |
| selected holdings | 30 |
| max weight | `6.452%` |
| effective names | 22.87 |

This is a branch selected by the full gated model.  It is not a claim that the
entire project ended at the original baseline.

## H. Self-Test Checklist

Before treating a route as final, we require:

| check | command or file | status |
|---|---|---|
| format validation | `python validate_submission.py submissions/portfolio.csv` | passed |
| portfolio regeneration | `stage2_baseline_guard_ensemble.py --as-of 20260508` | exact match |
| dynamic leakage audit | `05_final_leakage_audit_dynamic.csv` | passed |
| static scan | `06_final_leakage_audit_static_scan.csv` | reviewed |
| canonical backtest | `02` and `03` full-week CSVs | completed |
| root layout check | active model chain only in root; validation tools in `tools/` | completed |

## I. Files To Cite In The Written Report

| file | use |
|---|---|
| `00_stage2_final_report_summary.md` | main report narrative |
| `01_final_portfolio_metadata.csv` | final route and as-of regime metadata |
| `02_full_week_12_window_performance_summary.csv` | aggregate result table |
| `03_full_week_12_window_performance_detail.csv` | per-window evidence |
| `04_final_topk_and_weighting_check.md` | portfolio-size and weight-shape rationale |
| `05_final_leakage_audit_dynamic.csv` | dynamic no-leak evidence |
| `06_final_leakage_audit_static_scan.csv` | static audit evidence |
