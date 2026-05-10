# Stage2 Final Report: CSI500 Stock Selection

## 1. Objective

The competition task is a long-only CSI500 stock-selection problem.  For each
submission date, the model must output a `stock_code,weight` portfolio with at
least 30 stocks, non-negative weights summing to 1, and no single weight above
10%.  The scoring script evaluates realized portfolio return over the official
future holding window and subtracts the CSI500 benchmark return.

For Stage2, the practical target became a five-trading-day portfolio that is
not merely high on a few windows, but robust across full Monday-Friday weeks.
This changed the modeling emphasis from short-horizon burst chasing toward
stable five-day stock selection, regime-aware routing, and leakage-safe
validation.

## 2. Data

The final production route uses the official local data cache:

| item | value |
|---|---|
| stock data | `data/prices.parquet` |
| index data | `data/index.parquet` |
| latest local trading date | `2026-05-08` |
| final as-of date | `2026-05-08` |
| final evaluation target | next five trading days after as-of |

The data contains forward-adjusted daily OHLCV-style stock data and CSI500 index
prices.  During development we also tested external/open data, but did not
promote it to the final route because the cleaned versions did not improve both
mean excess and downside stability.

## 3. Baseline And Scoring Understanding

The official XGBoost baseline trains a five-day forward-return regressor using
features available at `as_of`.  It does not output excess return directly.  It
outputs a cross-sectional score for every available stock, then converts the top
ranked scores into portfolio weights.  The realized excess return is computed
only by `score_submission.py`, using future prices in the evaluation window.

This distinction mattered throughout the project.  A model can have reasonable
rank ordering but poor weight concentration, or it can pick stocks with high raw
return while still failing to beat the benchmark if the whole market rises even
faster.

## 4. Feature Engineering

The final code keeps a compact production core and several experimental feature
groups for route-specific models.

### 4.1 Production Core Features

`features.py` defines `CORE_FEATURE_COLUMNS` as the stable baseline feature set:

| group | examples | purpose |
|---|---|---|
| short/medium returns | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` | momentum and reversal state |
| risk | `vol_20d` | realized risk filter |
| liquidity | `volume_z_20d`, `turnover_ma_20d` | trading activity confirmation |
| trend location | `close_over_ma20`, `close_over_ma60`, `rsi_14` | trend strength and overextension |
| cross-sectional ranks | `ret_5d_rank`, `ret_20d_rank`, `vol_20d_rank` | daily relative normalization |

The core set stayed production-facing because broad feature stuffing repeatedly
hurt out-of-sample excess.  A key finding was that raw features and ranked
versions can both be useful to tree models; mechanically removing correlated
rank/raw pairs reduced performance.

### 4.2 Experimental Feature Groups

We tested additional feature families before deciding what to keep:

| feature family | examples | final decision |
|---|---|---|
| reference amplitude features | `amplitude_ma_20d`, rank variants | kept as reference only; not production |
| momentum/risk features | `ret_3d`, `vol_ratio_5_20`, `drawdown_20d`, `amount_z_20d` | useful in selected routes |
| quality/residual features | `beta_60d`, `market_corr_60d`, `downside_vol_20d`, `residual_ret_20d` | useful for defensive routing |
| microstructure-like alpha | gap, intraday, OBV, price-volume correlation | useful in weekly alpha overlays |
| calendar weekly cycle | Monday/Friday, month-start/end, post-gap reopening features | used in full-week route experiments |
| external/open data | valuation, market regime, fund flow | cleaned and tested, but rejected from final |
| graph/peer features | trailing return-correlation peer signals | rejected from final due weak robustness |

The most important feature-engineering lesson was not "more features is better."
The best production route uses richer features only in route-specific experts
and keeps a conservative core for the fallback XGBoost path.

## 5. Model Development Timeline

The project evolved through several distinct stages.

| stage | idea | result | decision |
|---|---|---|---|
| official baseline | XGBoost five-day raw return score, rank-weighted portfolio | simple, stable reference | kept as fallback |
| direct rank/weight learning | train model to learn future rank and target weight directly | produced unstable/equal-looking weights in some submissions | not final |
| tuned XGB / LightGBM | stronger tree learners, feature pruning, top-k sweeps | improved some windows but not enough alone | kept as ingredients |
| LSTM route | sequence model for short-term price dynamics | strong on some early rolling windows, weak after 3-day/5-day and full-week correction | kept only as fallback dependency |
| Stage1 short horizon | 3-day target, confidence weighting, LSTM/LightGBM consensus | reached reasonable return but below class average | diagnosed horizon/regime overfit |
| open-data enhancement | strict cleaning, lagging, correlation filtering for valuation/regime/fund-flow data | noisy; did not improve mean and floor together | rejected from final |
| CatBoost/regularized consensus | robust losses and multi-tree consensus | helpful in narrow regimes, weak globally | kept as fallback |
| meta/hybrid route | route-specific tree, regularized, and LSTM selection | improved old rolling validation | retained as dependency |
| full-week rebase | corrected validation to strict Monday-Friday windows | old high results no longer valid | reset final selection standard |
| final regime-gated ensemble | choose route by as-of market regime | best strict 12 full-week mean/floor | final route |

## 6. The Window Definition Error And Correction

One of the most important discoveries was that early "five-day" backtests used
the next five trading days after an as-of date, even when those dates crossed a
weekend or holiday.  That is a valid generic holding-period test, but it does
not match the intended Stage2 full-workweek target.

After auditing the windows, we found:

| old validation set | complete Mon-Fri windows | issue |
|---|---:|---|
| old 12-window setup | 2 / 12 | most windows crossed weekend/holiday gaps |
| old 9-window setup | 0 / 9 | none were complete workweeks |

This invalidated the earlier apparent high-water mark, including the old
multi-route consensus result around `+8.7%` mean excess.  That result was useful
for research, but it was not the final Stage2 selection standard.  We therefore
rebased validation to strict complete Monday-Friday windows and rebuilt the
final route around that corrected objective.

## 7. Final Model: Regime-Gated Ensemble

The final Stage2 generator is:

```text
stage2_baseline_guard_ensemble.py
```

It is named `baseline_guard_adaptive` in the validation reports.  The design is
a mixture-of-experts route selector, not a simple average of all models.

### 7.1 Experts Inside The Ensemble

| expert | implementation | role |
|---|---|---|
| official-style XGBoost baseline | `baseline_xgboost.py` and fallback inside `stage2_baseline_guard_ensemble.py` | stable defensive route |
| LightGBM/XGBoost tree consensus | `stage2_tree_consensus.py` plus helper scripts in `common/legacy_scripts/` | two-model tabular consensus |
| weekly alpha overlay | `stage2_weekly_alpha_overlay.py` | combines base portfolio with flow/volatility/OBV style weekly alpha |
| weekly consensus | `stage2_weekly_consensus_ensemble.py` | aggregates weekly alpha and weekly cycle tree |
| weekly cycle tree | `stage2_weekly_cycle_tree.py` | learns full-week excess-return features including calendar cycle terms |
| meta/hybrid fallback | `stage2_meta_portfolio_ensemble.py`, `stage2_hybrid_gate.py`, `stage2_regularized_consensus.py`, `lstm_rank_weight.py` | live fallback chain when no cached base is used |
| broad defensive tilt | internal function in `stage2_baseline_guard_ensemble.py` | low-risk broad route for mild positive flat tape |

### 7.2 Regime Gate

The gate uses only information observable at `as_of`, including:

| variable | meaning |
|---|---|
| `idx_ret_5d` | recent CSI500 index return |
| `idx_ret_20d` | medium-term index trend |
| `breadth_ret_5d_pos` | fraction of stocks with positive recent five-day return |
| `median_ret_5d` | median stock return over the recent five-day lookback |

The gate then decides whether the market is overheated, weak/choppy, broadly
selling off, stabilizing after a selloff, or safe enough for weekly consensus.
This avoided forcing one global model onto all regimes.

## 8. Why The Final Submission Uses `baseline_xgb`

The final as-of date was `2026-05-08`.  The route metadata is:

| field | value |
|---|---:|
| selected route | `baseline_xgb` |
| guard reason | `baseline_guard_overheated_high_breadth` |
| `idx_ret_5d` | `+5.975%` |
| `idx_ret_20d` | `+14.833%` |
| `breadth_ret_5d_pos` | `70.741%` |
| holdings | 30 |
| max weight | `6.452%` |
| effective names | 22.87 |

This does not mean the final model is the unmodified baseline.  It means the
full regime-gated ensemble chose its defensive XGBoost fallback because the
current market looked like a broad overheated rebound.  In similar historical
states, the more aggressive ensemble routes were more likely to chase crowded
winners, while the baseline fallback had a better floor.

## 9. Portfolio Construction

The final portfolio is not equal-weighted.  It uses a rank-weighted top-k
portfolio layer:

| design choice | final setting | rationale |
|---|---:|---|
| selected stocks | 30 | preserves rank signal in the fallback regime |
| max single weight | `6.452%` realized, below 10% cap | concentrated but compliant |
| effective names | 22.87 | avoids hidden equal weighting |
| top-5 weight | 30.11% | allows conviction |
| top-10 weight | 54.84% | more aggressive than broad diversification |

We explicitly swept top30/top35/top40/top50/top60 on similar baseline-guard
windows.  Top30 had the best mean and minimum excess among those checks, so the
final route did not dilute the signal into a larger stock set.

## 10. Validation Results

The final comparison uses 12 strict complete Monday-Friday full-week windows.

| model | mean excess | median excess | min excess | max excess | negative windows |
|---|---:|---:|---:|---:|---:|
| `baseline_guard_adaptive` | `+4.001%` | `+3.001%` | `+0.907%` | `+12.308%` | `0 / 12` |
| original `baseline_xgb` | `+0.527%` | `+0.264%` | `-1.221%` | `+2.916%` | `5 / 12` |

Detailed per-window evidence is in:

```text
03_full_week_12_window_performance_detail.csv
```

## 11. Self-Test And Leakage Controls

The final pipeline includes several self-tests.

| test | purpose | result |
|---|---|---|
| `validate_submission.py` | checks CSV format, stock count, non-negative weights, sum-to-one, max-weight cap | passed |
| dynamic leakage audit | compares full-data generation with physically truncated `date <= as_of` generation | passed |
| static scan | searches active code for leakage-prone references and cache/report dependencies | reviewed |
| regeneration check | regenerates final portfolio from script and compares against `submissions/portfolio.csv` | exact match |
| py_compile check | ensures active model and validation scripts import cleanly | passed |
| canonical score script | scores all backtest portfolios using `score_submission.py` logic | used for final tables |

The dynamic leakage audit passed for representative historical and final dates
with `max_abs_diff = 0`, `l1_diff = 0`, and `changed_names = 0`.  The active
route trims stock and index data to `date <= as_of` before feature construction.

## 12. Main Lessons

The most important lessons from the project were:

| lesson | implication |
|---|---|
| horizon alignment matters | Stage1 three-day winners did not automatically transfer to Stage2 five-day windows |
| validation-window definition matters | old rolling five-trading-day tests overstated performance for the final full-week target |
| more data is not automatically better | external data had to pass strict cleaning and ablation; it did not improve the final objective |
| LSTM can find short-term bursts but overfits easily | sequence routes were useful as candidates, not as the production backbone |
| simple models can be the right expert in some regimes | the final route selected XGBoost fallback for an overheated high-breadth market |
| self-tests are part of the model | leakage checks, as-of truncation, and regeneration tests became required before promotion |

## 13. Reproduction Commands

Generate the final portfolio:

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out submissions/portfolio.csv \
  --meta-out submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv
```

Validate the final CSV:

```bash
python validate_submission.py submissions/portfolio.csv
```

Run the leakage audit:

```bash
python tools/stage2_validation/stage2_leakage_audit.py \
  --as-of 20260123 20260320 20260508 \
  --models baseline_guard_adaptive \
  --out submissions/stage2/final_report_materials/05_final_leakage_audit_dynamic.csv \
  --static-out submissions/stage2/final_report_materials/06_final_leakage_audit_static_scan.csv
```

Run the full-week comparison:

```bash
python tools/stage2_validation/stage2_backtest_5day.py \
  --models baseline_xgb baseline_guard_adaptive \
  --full-week-only \
  --windows 12 \
  --jobs 4 \
  --out-dir submissions/stage2/backtests/final_check \
  --summary-out submissions/stage2/final_report_materials/02_full_week_12_window_performance_summary.csv
```
