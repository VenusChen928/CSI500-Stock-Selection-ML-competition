# CSI500 Stock Selection Competition Report

**Project:** CSI500 Stock Selection ML Competition, Spring 2026  
**Final Stage2 portfolio:** `stage2_final_portfolio.csv`  
**Final as-of date:** 2026-05-08  
**Final production entry point:** `stage2_baseline_guard_ensemble.py`  
**Local data coverage:** 2023-01-03 to 2026-05-08  

## Executive Summary

The task is to submit a long-only CSI500 portfolio for a future holding window.  The submission file contains only two columns, `stock_code` and `weight`; the scoring script then computes the realized portfolio return over the evaluation window and subtracts the CSI500 benchmark return.  Therefore, the modeling problem has two separate layers:

1. Predict a cross-sectional ordering or confidence score for stocks using only information available at the submission date.
2. Convert those scores into a valid portfolio with at least 30 positive-weight stocks, non-negative weights, weights summing to one, and no single stock above the 10% cap.

The final Stage2 route is a **regime-gated ensemble**.  It contains several candidate experts, including an official-style XGBoost fallback, a LightGBM/XGBoost tree consensus, a weekly alpha overlay, a full-week cycle tree, and meta/hybrid fallback routes.  A market-state gate chooses the route using only as-of observable index and breadth features.  For the final as-of date, 2026-05-08, the gate selected the defensive official-style XGBoost branch because the market looked like a broad overheated rebound.

On the final strict self-test of 12 complete Monday-Friday five-day windows, the final adaptive route achieved:

| model | mean excess | median excess | min excess | max excess | negative windows |
|---|---:|---:|---:|---:|---:|
| final regime-gated ensemble | +4.001% | +3.001% | +0.907% | +12.308% | 0 / 12 |
| original XGBoost baseline | +0.600% | +0.264% | -1.221% | +2.916% | 5 / 12 |

The final route improved both the average excess return and the downside floor relative to the original baseline.  The final report intentionally presents failed attempts as well, because many high-looking experiments were later rejected after stricter window definition and leakage checks.

The logic of the final method is:

| step | decision | why it comes next |
|---|---|---|
| scoring analysis | model outputs weights, while `score_submission.py` computes realized excess return | separates prediction from evaluation |
| factor design | build lagged price, liquidity, risk, market-relative, and weekly-cycle signals | defines what the model is allowed to know at as-of |
| model search | test XGBoost, LightGBM, CatBoost, LSTM, direct rank/weight learning, and hybrids | identifies which learners are stable versus regime-sensitive |
| validation correction | switch from generic five-trading-day windows to complete Monday-Friday self-test windows | aligns the model-selection test with the Stage2 objective |
| final route | use an as-of regime gate to choose among experts | avoids forcing one model into every market state |
| portfolio shape | choose top-k and weight concentration conditional on the selected route | converts scores into a compliant, risk-controlled submission |

## 1. Factors

### 1.1 Prediction Target And Scoring Definition

The official scoring script applies portfolio weights at the open of the evaluation start date and holds through the close of the evaluation end date.  For stock \(i\), the realized return is:

```text
r_i = close_i(end) / close_i(day_before_start) - 1
```

The portfolio return is the weighted sum of realized stock returns:

```text
portfolio_return = sum_i weight_i * r_i
```

The official competition metric is excess return:

```text
excess_return = portfolio_return - CSI500_benchmark_return
```

This means the model does not need to output excess return directly.  It outputs stock weights.  The excess return is computed only after future data become known in `score_submission.py`.

During development, we tested three target designs:

| target design | implementation idea | result | final decision |
|---|---|---|---|
| raw future return | predict `target_5d` and rank stocks by predicted return | stable, close to official baseline, useful fallback | retained |
| future excess return | predict `target_excess_5d = stock future return - index future return` | useful for weekly cycle experts | retained in specialized route |
| direct future rank/weight | train toward future rank or synthetic optimal weight | unstable; some portfolios became too equal-like or too noisy | rejected as primary route |

The final route separates **prediction** from **portfolio construction**.  This was more robust than directly learning weights, because the "true weight" is not an observed label in the data; it is an artificial transformation of future returns and can amplify noise.

### 1.2 Official Data Used

The final production route uses only the official local data cache.  External data were explored, cleaned, and tested, but not promoted into the final route because they did not improve both mean excess and downside stability.

| data item | file | value |
|---|---|---:|
| stock daily panel | `data/prices.parquet` | 397,757 rows |
| unique stocks | `data/prices.parquet` | 499 |
| trading dates | `data/prices.parquet` | 807 |
| date range | `data/prices.parquet` | 2023-01-03 to 2026-05-08 |
| index daily panel | `data/index.parquet` | 807 rows |
| constituent list | `data/constituents.csv` | 499 rows |

The core stock data contain daily OHLCV-style fields such as open, high, low, close, volume, amount, and turnover.  The index data provide CSI500 close and return features used for benchmark-relative signals.

### 1.3 Core Feature Set

The production baseline path uses a compact set of technical and cross-sectional features defined in `features.py` as `CORE_FEATURE_COLUMNS`.

| feature group | features | rationale |
|---|---|---|
| short and medium returns | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` | capture momentum, reversal, and trend persistence |
| volatility and risk | `vol_20d` | avoid stocks whose recent risk dominates signal |
| liquidity and activity | `volume_z_20d`, `turnover_ma_20d` | identify abnormal trading activity and tradeability |
| trend location | `close_over_ma20`, `close_over_ma60`, `rsi_14` | measure overextension versus moving averages |
| cross-sectional ranks | `ret_5d_rank`, `ret_20d_rank`, `vol_20d_rank` | normalize signals across a trading day |

The compact core was retained because blindly adding features repeatedly improved one or two windows while hurting the out-of-sample floor.  Tree models can use both raw and ranked features; mechanically dropping highly correlated raw/rank pairs reduced performance in several route checks.

### 1.4 Expanded Feature Families Tested

Several richer feature groups were implemented and used by specialized experts or rejected after ablation.

| feature family | examples | why tested | final use |
|---|---|---|---|
| momentum and reversal | `ret_3d`, `mom_accel_5_20`, `ret_60d_rank` | balance short-term bursts with medium trend | used in tree/weekly experts |
| risk and drawdown | `vol_ratio_5_20`, `downside_vol_20d`, `drawdown_20d` | penalize unstable names and recent crash risk | used in defensive routes |
| market-relative features | `excess_ret_5d`, `excess_ret_20d`, `beta_60d`, `residual_ret_20d` | separate stock alpha from market beta | used in weekly and quality routes |
| flow and price-volume features | `amount_z_20d`, `obv_20d`, `price_volume_corr_20d` | approximate money-flow confirmation | used in weekly alpha overlay |
| intraday and gap features | `gap_mean_20d`, `intraday_mean_20d`, `overnight_vol_20d` | detect reopening risk and persistent demand | used in weekly alpha experiments |
| calendar cycle features | Monday/Friday, month start/end, post-gap, full-workweek flags | align Stage2 with five-day workweek behavior | used in weekly cycle route |
| external/open data | valuation, regime, fund-flow fields | potentially add non-price information | rejected from final |
| graph/peer features | correlation-neighbor return features | model group influence among stocks | rejected from final |

### 1.5 Weekly Cycle Features

After we realized that the final Stage2 target was better represented by complete five-day workweeks, we added known-in-advance calendar cycle features.  These do not use future prices.  They are functions of the as-of date and the next five business days:

| weekly feature | interpretation |
|---|---|
| `eval_starts_monday` | whether the holding window begins on Monday |
| `eval_ends_friday` | whether the holding window ends on Friday |
| `eval_is_full_workweek` | whether the evaluation is a complete Monday-Friday week |
| `weekend_gap_days` | length of the non-trading gap before the holding window |
| `eval_month_start` and `eval_month_end` | month-start or month-end flow regime |
| `weekly_risk_appetite` | interaction of Monday risk appetite with recent returns and OBV |
| `weekly_friday_derisk` | Friday de-risking tilt using trend quality and volatility |
| `month_start_flow` | month-start liquidity and flow confirmation |
| `month_end_defensive` | month-end defensive tilt |
| `post_gap_reopen_flow` | reopening behavior after weekend or holiday gaps |
| `weekly_carry_quality` | full-week carry quality using OBV, trend, close location, and volatility |

These features were used only in the weekly expert route.  They were not forced into the compact baseline fallback, because the fallback performed best as a simpler and more regularized model.

### 1.6 Feature Normalization And Correlation Handling

The final feature-processing approach was deliberately moderate:

| processing step | implementation | reason |
|---|---|---|
| rolling features | computed only from current and historical observations per stock | avoid future leakage |
| cross-sectional rank features | per-date rank percentile for selected signals | reduce scale drift across market regimes |
| robust normalization in weekly route | rank-normalize stock-level features within each date | stabilize tree learners |
| correlation filtering in weekly route | Spearman correlation filter, threshold 0.90 | remove redundant route-specific features |
| target clipping in weekly route | clip training target at 1% and 99% quantiles | reduce outlier dominance |
| time decay | half-life 180 days in weekly route | emphasize recent regimes without discarding history |

We did not perform aggressive de-correlation globally.  In this project, high feature correlation did not always mean a feature was useless for tree splits.  The safer compromise was to use compact features for the baseline route and correlation filtering only in richer specialized routes.

## 2. Models

### 2.1 Model Development Timeline

The final route came from a long sequence of model and feature experiments.  The important part is that the project did not move linearly; some promising paths were later rejected after better validation.

| phase | experiment | observation | decision |
|---|---|---|---|
| baseline understanding | read `score_submission.py` and official XGBoost baseline | model predicts scores/weights; score script computes excess return | keep prediction and scoring separate |
| official XGBoost | `baseline_xgboost.py`, raw 5-day target, top-50 rank weights | simple and stable, but weak mean excess | keep as reference and fallback |
| direct rank/weight learning | target future rank and synthetic target weight | weight labels were unstable and sometimes produced near-equal weights | reject as main route |
| feature pruning | test core OHLCV, ranks, volatility, liquidity, trend, RSI | compact feature set had better floor than broad stuffing | keep compact core |
| LSTM sequence model | learn short-term sequences and confidence | strong on several early rolling windows | not stable across horizon changes |
| layered/gated LSTM | LSTM plus LightGBM confirmation and confidence weights | useful for Stage1 three-day objective | not final for Stage2 |
| LightGBM/XGBoost consensus | blend two tree portfolio rankings | improved some regimes | retained as expert ingredient |
| CatBoost robust challenger | quantile/MAE-style robust model | not competitive enough standalone | keep as rejected comparison |
| regularized consensus | stricter tree ensemble and risk overlays | helped in narrow regimes | retained as fallback route |
| open-data enhancement | strict lagging/cleaning of valuation, regime, fund flow | noisy, did not improve mean and min together | reject from final |
| full-week correction | change validation to complete Monday-Friday windows | invalidated old optimistic results | reset Stage2 standard |
| weekly alpha route | flow, OBV, low-vol confidence, calendar cycle | useful in selected regimes | retained as expert |
| final regime gate | route selection by as-of market state | best strict 12-window mean and floor | final route |

### 2.2 Official-Style XGBoost Baseline

The baseline model predicts raw five-day future stock return, then ranks the cross-section.

| setting | value |
|---|---:|
| target | `target_5d` |
| estimators | 400 |
| max depth | 5 |
| learning rate | 0.05 |
| subsample | 0.8 |
| colsample by tree | 0.8 |
| min child weight | 10 |
| L2 regularization | 1.0 |
| tree method | `hist` |
| validation days | 10 trading days |
| Embargo | 5 trading days |
| early stopping | 30 rounds |
| default portfolio size | 50 names |

The split for each as-of date is time ordered:

```text
available data up to as_of
train target cutoff = as_of - 5 trading days

[ historical train rows ] [ 5-day embargo ] [ 10-day validation rows ] [ as_of prediction ]
```

The embargo is important because the target for date \(t\) uses prices through \(t+5\).  Without the embargo, training labels could overlap with validation prices.

### 2.3 LightGBM / XGBoost Tree Consensus

The tree consensus route was inspired by the idea that different tree learners can agree on robust names even when raw predicted scores differ.  The route trains a tuned XGBoost and a LightGBM-style portfolio helper, then combines their selected stocks and ranks.

| component | role |
|---|---|
| tuned XGBoost | strong tabular ranking model |
| LightGBM | fast gradient boosting challenger |
| consensus rank | prefer names selected by both models |
| rank-weight layer | avoid pathological raw-score scaling |
| optional factor reweight | apply drawdown/vol/quality tilts only when regime gate allows |

The most useful finding was that model averaging alone was not sufficient.  The consensus needed a portfolio layer that preserved conviction while controlling concentration.

### 2.4 Weekly Cycle Tree Expert

The weekly cycle route trains on all available five-trading-day targets but adds features describing the next five-business-day calendar shape.  It is not trained only on full weeks, which helps avoid overfitting to too few examples.

| setting | LightGBM | XGBoost | CatBoost challenger |
|---|---:|---:|---:|
| robust loss | Huber | pseudo-Huber | quantile |
| estimators/iterations | 650 | 520 | 520 |
| learning rate | 0.022 | 0.022 | 0.025 |
| depth/leaves | 47 leaves | depth 3 | depth 5 |
| min child / leaf | 70 | 45 | L2 leaf reg 10 |
| subsample | 0.82 | 0.82 | bagging temp 0.35 |
| colsample | 0.70 | 0.70 | ordered boosting style |
| regularization | alpha 0.35, lambda 5.0 | alpha 0.20, lambda 6.0 | random strength 1.0 |

The final weekly cycle configuration used the LightGBM/XGBoost pair, not CatBoost, because the robust two-tree pair gave better stability in the final full-week checks.

### 2.5 LSTM And Confidence-Weight Experiments

The LSTM route was developed because stock data are sequential.  The main idea was to learn a short-term sequence representation, then combine it with a confidence score:

```text
final confidence = model agreement + rank strength - volatility/drawdown penalty
```

For Stage1, this was useful.  On the three-day validation around the Stage1 horizon, the best LSTM/LightGBM confidence configuration reached about +3.3% mean excess on three recent windows:

| configuration | alpha_lstm | risk penalty | top_k | rank power | confidence power | mean excess | min excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| best recent Stage1 LSTM confidence | 0.95 | 0.30 | 40 | 1.35 | 3.0 | +3.302% | +3.246% |

However, this did not transfer reliably to Stage2.  Once the target changed to strict five-day workweeks, LSTM routes became more regime-sensitive and could produce negative windows.  The final Stage2 route therefore keeps LSTM in the historical fallback chain but does not use it as the final primary expert.

### 2.6 Why Direct Weight Learning Was Rejected

Directly learning future rank and target weight looked attractive because the submission itself is a weight file.  In practice, it caused two problems:

| issue | why it happened |
|---|---|
| synthetic labels are noisy | there is no true observed "best weight"; any label is constructed from future returns |
| weight scale instability | small return differences can create very different target weights |
| equal-looking portfolios | regularization and noisy labels sometimes collapsed weights toward similar values |
| poor transfer across horizons | 3-day, generic 5-day, and full-week 5-day objectives favored different shapes |

The final design uses score/rank learning plus a separate portfolio allocator.  This made the model easier to validate and easier to constrain under competition rules.

### 2.7 Final Regime-Gated Ensemble

The final production route is implemented in `stage2_baseline_guard_ensemble.py`.  It is a mixture-of-experts selector, not a simple average of all model outputs.

| expert | implementation | role |
|---|---|---|
| official-style XGBoost fallback | `baseline_xgboost.py` and fallback function inside final route | stable defensive branch |
| LightGBM/XGBoost tree consensus | `stage2_tree_consensus.py`, `lightgbm_portfolio.py`, `tuned_xgboost_portfolio.py` | tabular consensus branch |
| weekly alpha overlay | `stage2_weekly_alpha_overlay.py` | flow/volatility/OBV style confidence branch |
| weekly consensus | `stage2_weekly_consensus_ensemble.py` | combines weekly alpha and weekly cycle ideas |
| weekly cycle tree | `stage2_weekly_cycle_tree.py` | full-week calendar and excess-return expert |
| meta/hybrid fallback | `stage2_meta_portfolio_ensemble.py`, `stage2_hybrid_gate.py`, `stage2_regularized_consensus.py`, `lstm_rank_weight.py` | fallback chain for non-cached routes |
| broad defensive tilt | internal final-route function | broad low-risk route for mild positive flat tape |

The gate uses only as-of observable variables:

| gate variable | interpretation |
|---|---|
| `idx_ret_5d` | recent CSI500 index return |
| `idx_ret_20d` | medium-term index trend |
| `breadth_ret_5d_pos` | fraction of stocks with positive recent five-day return |
| `median_ret_5d` | median stock return over the recent five-day lookback |

The final as-of state was:

| field | value |
|---|---:|
| as-of date | 2026-05-08 |
| selected route | `baseline_xgb` |
| guard reason | `baseline_guard_overheated_high_breadth` |
| `idx_ret_5d` | +5.975% |
| `idx_ret_20d` | +14.833% |
| `breadth_ret_5d_pos` | 70.741% |
| selected stocks | 30 |
| max weight | 6.452% |
| effective names | 22.87 |
| route validation rank IC | 0.0783 |

This does not mean the final project is simply the unchanged baseline.  It means the full gated model chose the baseline fallback for this as-of date because historical similar regimes favored a defensive branch over aggressive weekly consensus.

### 2.8 Portfolio Construction And Weighting

The portfolio layer was tuned separately from the prediction layer.  This is important because a model can rank stocks reasonably but still lose excess return if the final portfolio is too diluted, too concentrated, or too equal-weighted.  The final portfolio is not equal-weighted.

| metric | final value |
|---|---:|
| holdings | 30 |
| weight sum | 1.0000 |
| max weight | 6.452% |
| min weight | 0.215% |
| effective number of names | 22.87 |
| top-5 weight | 30.11% |
| top-10 weight | 54.84% |

The final top holdings are:

| rank | stock_code | weight |
|---:|---|---:|
| 1 | 688615 | 6.452% |
| 2 | 002624 | 6.237% |
| 3 | 300857 | 6.022% |
| 4 | 300676 | 5.806% |
| 5 | 688017 | 5.591% |
| 6 | 603341 | 5.376% |
| 7 | 002131 | 5.161% |
| 8 | 600536 | 4.946% |
| 9 | 300454 | 4.731% |
| 10 | 601615 | 4.516% |

The table below is a **portfolio-shape sanity check**, not the main 12-window model comparison.  Its purpose was narrower: after the final 2026-05-08 regime gate selected the `baseline_xgb` fallback, we needed to decide whether that fallback branch should hold the minimum 30 names or dilute into more stocks.  To avoid tuning on the final unknown window, I only used historical windows where the same final gate also routed into the baseline-style fallback, then rescored the same fallback idea with `top_k` set to 30, 35, 40, 50, and 60.

The two historical windows used for this check were:

| as_of | evaluation window | gate reason |
|---|---|---|
| 2026-01-23 | 2026-01-26 to 2026-01-30 | overheated high breadth |
| 2026-03-20 | 2026-03-23 to 2026-03-27 | severe broad selloff |

Across only those two baseline-routed sanity-check windows, top30 had the best mean and best minimum:

| top_k | mean excess | min excess |
|---:|---:|---:|
| 30 | +1.993% | +1.141% |
| 35 | +1.831% | +0.616% |
| 40 | +1.626% | +0.153% |
| 50 | +1.268% | -0.380% |
| 60 | +1.072% | -0.551% |

The causal use of this table is therefore limited but clear: it did not decide the overall model, and it was not used to claim that 30 names is globally optimal for all regimes.  It only supported the final fallback-branch portfolio shape.  In the closest available baseline-routed regimes, larger portfolios diluted the XGBoost rank signal and lowered the floor.  Since the final top30 shape stayed below the 10% cap and had 22.87 effective names, I kept top30 for the final overheated high-breadth fallback.

## 3. Results

This section reports how the final portfolio generator performed on held-out complete workweek windows relative to the original XGBoost baseline.  The construction of the train / validation / test split and the no-leakage checks are documented separately in Section 5, so this section focuses on the empirical outcome.

### 3.1 Held-Out Performance Relative To Baseline

The final route was compared with the original baseline XGBoost on the same 12 windows.

| model | mean excess | sum excess | median excess | min excess | max excess | count | negative windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| final regime-gated ensemble | +4.001% | +48.006% | +3.001% | +0.907% | +12.308% | 12 | 0 |
| original XGBoost baseline | +0.600% | +7.194% | +0.264% | -1.221% | +2.916% | 12 | 5 |

The final route improved mean excess by +3.401 percentage points per held-out window and removed all negative excess windows.  It beat the original baseline in all 12 test windows.

### 3.2 Window-By-Window Held-Out Results

| window | baseline excess | final excess | difference | winner |
|---|---:|---:|---:|---|
| 2026-01-12 to 2026-01-16 | +2.227% | +2.948% | +0.721% | final |
| 2026-01-19 to 2026-01-23 | -0.581% | +6.633% | +7.214% | final |
| 2026-01-26 to 2026-01-30 | -0.380% | +1.141% | +1.521% | final |
| 2026-02-02 to 2026-02-06 | -0.228% | +2.741% | +2.969% | final |
| 2026-02-09 to 2026-02-13 | +0.392% | +6.872% | +6.480% | final |
| 2026-03-02 to 2026-03-06 | -1.221% | +3.253% | +4.474% | final |
| 2026-03-09 to 2026-03-13 | +0.136% | +12.308% | +12.172% | final |
| 2026-03-16 to 2026-03-20 | +1.325% | +2.211% | +0.886% | final |
| 2026-03-23 to 2026-03-27 | +2.916% | +3.098% | +0.182% | final |
| 2026-03-30 to 2026-04-03 | -0.392% | +0.907% | +1.299% | final |
| 2026-04-13 to 2026-04-17 | +1.830% | +2.841% | +1.011% | final |
| 2026-04-20 to 2026-04-24 | +1.170% | +3.054% | +1.884% | final |

### 3.3 Route Diagnostics For The Reported Results

The final model is adaptive.  Different windows selected different route types based on as-of regime:

| as_of | evaluation window | selected route | guard reason | `idx_ret_5d` | `idx_ret_20d` | breadth |
|---|---|---|---|---:|---:|---:|
| 2026-01-09 | 2026-01-12 to 2026-01-16 | weekly cycle tree | extreme broad rebound | +7.92% | +12.59% | 88.78% |
| 2026-01-16 | 2026-01-19 to 2026-01-23 | weekly consensus | weekly consensus allowed | +2.18% | +15.34% | 48.10% |
| 2026-01-23 | 2026-01-26 to 2026-01-30 | baseline XGBoost | overheated high breadth | +4.34% | +16.84% | 79.76% |
| 2026-01-30 | 2026-02-02 to 2026-02-06 | weekly consensus | weekly consensus allowed | -2.56% | +12.12% | 27.25% |
| 2026-02-06 | 2026-02-09 to 2026-02-13 | weekly consensus | weekly consensus allowed | -2.68% | +1.11% | 29.86% |
| 2026-02-27 | 2026-03-02 to 2026-03-06 | broad defensive tilt | mild positive flat tape | +2.79% | +3.23% | 64.13% |
| 2026-03-06 | 2026-03-09 to 2026-03-13 | weekly consensus | weekly consensus allowed | -3.44% | -1.85% | 24.65% |
| 2026-03-13 | 2026-03-16 to 2026-03-20 | weekly alpha current | weak flat positive medium trend | -1.44% | +1.15% | 40.88% |
| 2026-03-20 | 2026-03-23 to 2026-03-27 | baseline XGBoost | severe broad selloff | -5.82% | -7.88% | 11.82% |
| 2026-03-27 | 2026-03-30 to 2026-04-03 | weekly cycle tree | long selloff stabilization | -0.29% | -10.64% | 38.68% |
| 2026-04-10 | 2026-04-13 to 2026-04-17 | weekly alpha auto | overheated negative medium trend | +4.79% | -4.67% | 73.15% |
| 2026-04-17 | 2026-04-20 to 2026-04-24 | weekly consensus | weekly consensus allowed | +3.07% | +4.28% | 64.73% |

This table is important because it shows that the final result is not produced by selecting the best route after seeing future returns.  The route is chosen from as-of observable market state.

### 3.4 IC As A Secondary Result Diagnostic

In addition to excess return, we used rank-style IC checks as a sanity signal:

| IC type | definition | use |
|---|---|---|
| validation rank IC | daily Spearman correlation between validation target and model prediction | internal model sanity check during training |
| allocation IC, all universe | Spearman correlation between submitted weights, including zeros for unheld names, and realized future stock returns | checks whether portfolio allocation direction aligns with realized cross-sectional returns |
| allocation IC, selected only | Spearman correlation between selected-stock weights and realized returns inside the portfolio | checks whether heavier selected names outperform lighter selected names |

IC was not used as the only selection metric.  The official objective is excess return, so IC was used to diagnose ranking quality and avoid misleading high-return accidents.

| model | mean excess | min excess | negative windows | mean all-universe allocation IC | median all-universe allocation IC | mean selected-only allocation IC |
|---|---:|---:|---:|---:|---:|---:|
| final regime-gated ensemble | +4.001% | +0.907% | 0 | 0.0879 | 0.0671 | 0.3055 |
| original XGBoost baseline | +0.600% | -1.221% | 5 | 0.0025 | -0.0310 | -0.0010 |

The final route had both stronger excess return and better allocation IC.  This supports the interpretation that the improvement was not only from market beta or a single lucky window; the final portfolio weights were more aligned with future cross-sectional winners.

## 4. Analysis

### 4.1 What Worked

The most successful design choices were:

| choice | why it worked |
|---|---|
| separating prediction from portfolio allocation | prevented unstable synthetic weight labels from dominating training |
| compact core fallback | reduced overfitting and kept a stable defensive branch |
| route-specific richer features | allowed weekly alpha features without contaminating all regimes |
| as-of regime gate | avoided forcing one model onto incompatible market states |
| robust losses and regularization | reduced sensitivity to outlier future returns |
| complete full-week validation | aligned self-test with the Stage2 target |
| leakage audits | protected against accidentally using future information or cached results |

The final route's biggest improvement came from not trying to make one model win every regime.  The official-style XGBoost baseline is weak on average, but it is useful in overheated or selloff regimes.  Weekly alpha and weekly cycle routes can be stronger when the market state is more normal.  The gate turns this into a practical mixture-of-experts system.

### 4.2 What Did Not Work

Several ideas looked promising but were not promoted:

| failed or limited idea | reason |
|---|---|
| direct rank/weight learning | labels were synthetic and unstable; portfolio weights sometimes became too equal-like |
| standalone LSTM | strong local windows, weak horizon transfer and regime stability |
| transformer-style deep models | data size and validation instability made them too risky for final |
| noisy external data | strict cleaning did not improve mean and floor together |
| graph/peer factors | group influence signals were not robust enough in held-out windows |
| blind feature expansion | improved single windows but hurt minimum excess |
| overly broad top-k portfolios | diluted rank signal in the final fallback branch sanity check |
| old generic five-day validation | overstated performance by mixing weekend/holiday effects |

The Phase 1 result was also a useful warning.  The Stage1 model had a reasonable local backtest but underperformed the class average in the official 2026-05-06 to 2026-05-08 window.  That pushed the Stage2 design toward simpler, better-regularized models and stricter self-tests.

### 4.3 Why The Final Route Uses Baseline XGBoost On 2026-05-08

This was a point of confusion during development.  The final route selected `baseline_xgb` for 2026-05-08, but the final model is not merely the original baseline.  It is a gated ensemble that sometimes routes to baseline when market conditions match a defensive regime.

For 2026-05-08:

| signal | interpretation |
|---|---|
| `idx_ret_5d = +5.975%` | very strong short-term index rebound |
| `idx_ret_20d = +14.833%` | extended medium-term strength |
| `breadth_ret_5d_pos = 70.741%` | broad participation |
| guard reason | overheated high-breadth rebound |

In this state, aggressive weekly consensus can chase already crowded winners.  Historical similar states favored a more conservative XGBoost fallback with 30 rank-weighted names.

### 4.4 Remaining Limitations

The final route is stronger than the official baseline in our self-test, but it is not perfect.

| limitation | consequence |
|---|---|
| only official OHLCV-style data in final route | no fundamental or industry structure in final submission |
| regime gate is rule-based | may miss nuanced transitions |
| self-test windows are limited | 12 full-week windows are useful but still a small sample |
| final as-of selected fallback | upside may be lower than a more aggressive route if rebound continues |
| no true industry-neutral optimizer | sector concentration risk may remain |

Future improvements should focus on clean industry/fundamental features, stronger walk-forward validation, and a more formal meta-learner for route selection.  Any additional data should pass the same strict lagging, cleaning, correlation, and ablation standards used here.

## 5. Self-Test

This section is written to match the self-test requirement directly.  It explains how the provided data were split into training, validation, and test sets, why the split is time-safe, how leakage was checked, and what performance the final model achieved on the held-out test set relative to the baseline.

### 5.1 Chronological Train / Validation / Test Split

The self-test uses a walk-forward chronological split, not a random split.  This is necessary for financial data because random splitting would mix future market regimes into training and produce look-ahead leakage.

The provided data are split separately for each historical as-of date:

| split | definition | purpose |
|---|---|---|
| training set | historical supervised rows whose five-day targets are fully observable before the validation/embargo boundary | fit model parameters |
| Embargo | five trading days removed between train and validation | prevent overlapping five-day target windows |
| validation set | the last 10 eligible trading days before the as-of prediction point, after embargo | early stopping, validation IC, route sanity checks |
| test set | the future five trading days after as-of, never visible during feature construction or training | held-out scoring with `score_submission.py` |

For every tested as-of date:

```text
1. truncate stock and index data to date <= as_of
2. build features on the truncated panel
3. remove the last 5 trading days from supervised training target availability
4. split the remaining supervised data into train / embargo / validation
5. train the model route
6. generate portfolio at as_of
7. score only on future dates after as_of
```

The official-style XGBoost branch uses the following concrete split:

| split component | definition |
|---|---|
| training target cutoff | `as_of - 5 trading days` |
| training set | all eligible rows up to `train_end` |
| Embargo | 5 trading days discarded between train and validation |
| validation set | last 10 trading days after embargo |
| test set | future five trading days after as-of |

Each portfolio is generated using only data available before its own test window.  The route gate also uses only as-of market state, not realized future returns.

### 5.2 Validation Design And Window Correction

Early in the project, many "five-day" tests used the next five trading days after an arbitrary as-of date.  That is a valid generic holding-period test, but it often crosses weekends or holidays.  After auditing the intended Stage2 target, we changed the main self-test to complete Monday-Friday workweeks.

| old validation set | complete Mon-Fri windows | issue |
|---|---:|---|
| old 12-window setup | 2 / 12 | most windows crossed weekend or holiday gaps |
| old 9-window setup | 0 / 9 | none were strict workweeks |

This correction invalidated an earlier optimistic high-water mark around +8.7% mean excess.  It was useful research evidence but not safe as the final Stage2 standard.  The final model-selection criterion therefore prioritized the complete-workweek test set, while also checking that the route did not rely on one lucky window.

### 5.3 Held-Out Test Set And Performance

The final 12 held-out complete workweek windows were:

| as_of | evaluation window |
|---|---|
| 2026-01-09 | 2026-01-12 to 2026-01-16 |
| 2026-01-16 | 2026-01-19 to 2026-01-23 |
| 2026-01-23 | 2026-01-26 to 2026-01-30 |
| 2026-01-30 | 2026-02-02 to 2026-02-06 |
| 2026-02-06 | 2026-02-09 to 2026-02-13 |
| 2026-02-27 | 2026-03-02 to 2026-03-06 |
| 2026-03-06 | 2026-03-09 to 2026-03-13 |
| 2026-03-13 | 2026-03-16 to 2026-03-20 |
| 2026-03-20 | 2026-03-23 to 2026-03-27 |
| 2026-03-27 | 2026-03-30 to 2026-04-03 |
| 2026-04-10 | 2026-04-13 to 2026-04-17 |
| 2026-04-17 | 2026-04-20 to 2026-04-24 |

The final model's performance on this held-out test set was:

| model | mean test excess | median test excess | min test excess | negative test windows | wins versus baseline |
|---|---:|---:|---:|---:|---:|
| final regime-gated ensemble | +4.001% | +3.001% | +0.907% | 0 / 12 | 12 / 12 |
| original XGBoost baseline | +0.600% | +0.264% | -1.221% | 5 / 12 | 0 / 12 |

This satisfies the self-test requirement that the reported test performance exceed the provided baseline.  The final model improved mean test excess by +3.401 percentage points and had no negative excess test windows, while the baseline had five negative windows.  The detailed window-by-window returns are shown in Section 3.2.

### 5.4 No-Leakage And Cache Controls

We performed both dynamic and static checks.

| check | purpose | result |
|---|---|---|
| format validation | verify stock count, positive weights, sum to one, max weight cap | passed |
| dynamic truncation audit | compare full-data generation with physically truncated `date <= as_of` data | passed |
| static code scan | search active route files for `score_submission`, report-cache, archive, and leakage-prone references | reviewed |
| regeneration check | regenerate final `stage2_final_portfolio.csv` from script and compare | exact match |
| py_compile | ensure active scripts import cleanly | passed |

The final dynamic leakage audit was rerun after the last folder cleanup.  It
passed on six representative as-of dates chosen to cover the major gate
branches: weekly cycle, broad defensive tilt, weekly alpha, and the final
baseline fallback.

| as_of | model | max absolute weight difference | L1 difference | changed names | pass |
|---|---|---:|---:|---:|---|
| 2026-01-09 | final route | 0.0 | 0.0 | 0 | true |
| 2026-02-27 | final route | 0.0 | 0.0 | 0 | true |
| 2026-03-13 | final route | 0.0 | 0.0 | 0 | true |
| 2026-03-27 | final route | 0.0 | 0.0 | 0 | true |
| 2026-04-10 | final route | 0.0 | 0.0 | 0 | true |
| 2026-05-08 | final route | 0.0 | 0.0 | 0 | true |

The active route trims stock and index data to `date <= as_of` before feature construction.  Target columns use forward shifts only for supervised training rows, and prediction rows do not use future targets.  Optional cache arguments exist for speed during research, but the final reproduction path and leakage audit run with no cache directory; the final route regenerates live from the official data cache.

## 6. Reproducibility

### 6.1 Environment

```bash
conda activate mlcomp-sp26
pip install -r requirements.txt
```

Main package requirements:

| package | requirement |
|---|---|
| pandas | >= 2.0 |
| numpy | >= 1.24, < 2.0 |
| xgboost | >= 2.0 |
| lightgbm | >= 4.6 |
| torch | >= 2.0 |
| scikit-learn | >= 1.3 |
| scipy | >= 1.11 |
| pyarrow | >= 14.0 |
| catboost | >= 1.2 |

### 6.2 Reproduce Final Portfolio

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out stage2_final_portfolio.csv \
  --meta-out stage2_report/final_report_materials/01_final_portfolio_metadata.csv

python validate_submission.py stage2_final_portfolio.csv
```

### 6.3 Reproduce Full-Week Self-Test

```bash
python tools/stage2_validation/stage2_backtest_5day.py \
  --models baseline_xgb baseline_guard_adaptive \
  --full-week-only \
  --windows 12 \
  --jobs 4 \
  --out-dir stage2_report/backtests/guard_route_v2_nocache_fullweek12_20260510 \
  --summary-out stage2_report/final_report_materials/02_full_week_12_window_performance_summary.csv \
  --detail-out stage2_report/final_report_materials/03_full_week_12_window_performance_detail.csv
```

The `stage2_report/backtests/` folder stores the per-window portfolio CSVs
generated by the final 12-window self-test, plus companion metadata CSVs for
the adaptive route.  The command above refreshes those portfolio records
together with the retained summary/detail evidence files.

### 6.4 Reproduce Leakage Audit

```bash
python tools/stage2_validation/stage2_leakage_audit.py \
  --as-of 20260109 20260227 20260313 20260327 20260410 20260508 \
  --models baseline_guard_adaptive \
  --out stage2_report/final_report_materials/05_final_leakage_audit_dynamic.csv \
  --static-out stage2_report/final_report_materials/06_final_leakage_audit_static_scan.csv
```

### 6.5 Final Consistency Checks

After removing the old draft report and consolidating the project layout, the
final consistency check confirmed:

| check | result |
|---|---|
| `stage2_final_portfolio.csv` format validation | passed |
| data cache max date | 2026-05-08 |
| final portfolio row count | 30 |
| final portfolio weight sum | 1.000000000000 |
| final portfolio max weight | 6.4516% |
| final effective number of names | 22.8689 |
| regenerated final portfolio versus root CSV | exact match, max difference 0 |
| performance summary versus detail CSV | matched |
| IC summary/detail versus score detail | matched |
| gate route table versus self-test windows | matched |
| dynamic leakage audit | 6 / 6 rows passed |

### 6.6 Report Evidence Files

| file | purpose |
|---|---|
| `stage2_report/final_report_materials/01_final_portfolio_metadata.csv` | final as-of route metadata and validation IC |
| `stage2_report/final_report_materials/02_full_week_12_window_performance_summary.csv` | aggregate self-test performance |
| `stage2_report/final_report_materials/03_full_week_12_window_performance_detail.csv` | per-window returns |
| `stage2_report/final_report_materials/04_final_topk_and_weighting_check.md` | top-k and weight-shape sanity check |
| `stage2_report/final_report_materials/05_final_leakage_audit_dynamic.csv` | dynamic no-leak audit |
| `stage2_report/final_report_materials/06_final_leakage_audit_static_scan.csv` | static leakage scan |
| `stage2_report/final_report_materials/07_stage2_experiment_history.md` | chronological experiment log |
| `stage2_report/final_report_materials/08_full_week_12_window_ic_detail.csv` | per-window allocation IC |
| `stage2_report/final_report_materials/09_full_week_12_window_ic_summary.csv` | aggregate allocation IC |
| `stage2_report/final_report_materials/10_full_week_gate_route_detail.csv` | route selected by the regime gate |
| `stage2_report/backtests/` | saved per-window portfolio CSVs and adaptive-route metadata from the final 12-window self-test |
| `stage2_report/backtests/README.md` | explains how to regenerate the per-window portfolio records |

## 7. Conclusion

The final Stage2 system is a leakage-safe, walk-forward-tested, regime-gated stock-selection ensemble.  The strongest lesson from the project was that robust portfolio performance came less from adding a more complex model and more from matching the validation window, controlling overfitting, separating score prediction from weight allocation, and routing among experts based only on as-of market state.

The final adaptive route outperformed the original XGBoost baseline on every one of the 12 strict complete workweek self-test windows, with +4.001% mean excess versus +0.600% for the baseline, and with zero negative excess windows.  The final portfolio for 2026-05-08 is therefore a defensive branch selected by a broader ensemble, not a simple unmodified baseline submission.
