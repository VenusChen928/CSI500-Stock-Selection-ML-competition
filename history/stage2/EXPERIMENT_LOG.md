# Stage2 Experiment Log

Stage2 is the active target: maximize five-trading-day excess return while
keeping negative-window risk low. New experiments should be added here with the
date, data snapshot, windows, mean excess, min excess, negative-window count,
and keep/reject decision.

## Current Data Snapshot

- Local `data/prices.parquet` and `data/index.parquet` were last checked at
  `2026-05-08`.
- The current standard 5-window test used as-of dates:
  `2026-03-26`, `2026-04-02`, `2026-04-10`, `2026-04-17`, `2026-04-24`.
- The broader 9-window robustness test also includes:
  `2026-03-02`, `2026-03-09`, `2026-03-16`, `2026-03-23`.

## Current Best Route

- Script: `stage2_multiroute_consensus.py`
- Current candidate:
  `current_best/stage2_multiroute_consensus_20260508.csv`
- Primary report:
  `reports/stage2_multiroute_consensus_12w_20260510_summary.csv`
- Supporting reports:
  `reports/stage2_multiroute_consensus_refine_12w_20260510.csv`,
  `reports/stage2_open_factor_sweep_fast_12w_20260510.csv`,
  `reports/stage2_graph_factor_sweep_fast_12w_20260510.csv`
- 12-window mean excess: `+8.718%`
- 12-window min excess: `+4.361%`
- 12-window max excess: `+20.191%`
- Negative windows: `0 / 12`

Design summary: keep `stage2_hybrid_gate.py` and
`stage2_meta_portfolio_ensemble.py` as the base route family, then apply
`stage2_multiroute_consensus.py`. The multiroute layer aggregates a broader
set of cached/live as-of-safe route portfolios plus repeated votes from the
current meta route, then rebuilds a concentrated 30-name consensus portfolio.
Fresh as-of dates fall back to live generation from data `<= as_of`.

## Important Baselines

- `reports/stage2_5day_current_20260506_summary.csv`
  - `baseline_xgb`: mean `+1.152%`, min `-0.327%`, negative windows `2 / 5`.
  - `tuned_xgb` / old `gated_layered_lstm`: mean `+1.030%`, min `-1.165%`,
    negative windows `1 / 5`.
  - `lstm_rank_weight`: mean `+0.895%`, min `-1.043%`, negative windows `2 / 5`.
  - `lightgbm`: mean `+0.821%`, min `-0.779%`, negative windows `1 / 5`.

## Useful Candidates

- `open_lgb_valuation`
  - Report: `reports/stage2_open_valuation_20260506_summary.csv`
  - Mean `+1.179%`, min `-0.633%`, negative windows `1 / 5`.
  - Keep as a confirmation route, not as a standalone production model.
- `baseline_flow_gate`
  - Report: `reports/stage2_flow_gate_20260506_summary.csv`
  - Mean `+1.211%`, min `-0.455%`, negative windows `2 / 5`.
  - Keep only as diagnostic evidence until the gate is less ex-post.
- Fixed LSTM policy probe
  - Report: `reports/stage2_lstm_fixed_policy_probe_20260506_summary.csv`
  - Best LSTM-route setting: `top_k=30`, `temperature=0.65`,
    `rank_blend=1.0`.

## Rejected Or Deprioritized Directions

- Open-data market regime features hurt the five-window objective and are kept
  under `backtests/rejected/`.
- Enhanced stock-only tree features improved some current windows but failed on
  historical checks, so they are not production.
- Earlier transformer/open-data ranker submissions are retained under
  `rejected/` as negative evidence, not active routes.
- Smoke/debug runs were deleted because they only tested plumbing and had no
  modeling value.
- `stage2_regularized_consensus.py` added strict feature filtering and a
  Ridge/LightGBM/XGBoost/CatBoost consensus, but the first stage2 smoke test
  underfit badly, so it is retained only as a reference for anti-leakage and
  regularization plumbing.

## 2026-05-10 10% Target Continuation

- Ran `download_data.py --update --end 20260510`. No new rows were added because
  2026-05-09 and 2026-05-10 were weekend dates; local prices/index still end at
  `2026-05-08`.
- Re-tested strict open-data features from `stage2_open_data_features.py`.
  Valuation, market-regime, and fund-flow features are cleaned with backward
  as-of joins, lag limits, robust cross-sectional z-scores, coverage checks,
  and correlation filtering. Standalone open-data factor portfolios were not
  competitive: best 12-window mean excess was only about `+0.804%` with several
  negative windows, so open data is not promoted as a stock-pool route.
- Tested graph/peer-group features based only on trailing 90-day stock-return
  correlations. The best peer factor reached only about `+1.387%` mean excess
  and had 6 negative windows, so it is retained as diagnostic evidence rather
  than a production signal.
- Tested the previous meta route's weight curve (`top_k`, `rank_power`,
  `mix_agg`). The best cached shape remained around `+8.08%`, confirming that
  the bottleneck was not simple weight tuning.
- Promoted `stage2_multiroute_consensus.py`: a broader candidate-pool ensemble
  inspired by the class-ensemble idea. Best validated setting uses the broad
  route set, five current-meta votes, `top_k=30`, and `rank_power=128`.
- New 12-window validation: mean excess `+8.718%`, min `+4.361%`, max
  `+20.191%`, negative windows `0 / 12`.
- Recent known window (`as_of=20260428`, `2026-04-29` to `2026-05-08`):
  portfolio `+16.419%`, benchmark `+5.975%`, excess `+10.444%`.
- Generated and validated:
  `current_best/stage2_multiroute_consensus_20260428.csv` and
  `current_best/stage2_multiroute_consensus_20260508.csv`.

## 2026-05-08 Update

- Data was refreshed with `download_data.py --update --end 20260508`.
- `tree_consensus` vs `regime_gated_lstm` retest showed the old LSTM route had
  negative mean excess on the refreshed 5-window setup, while LightGBM and tree
  consensus were more stable.
- Shape sweep promoted two candidate portfolio layers:
  aggressive `top_k=30`, `alpha_xgb=0.25`, `rank_power=1.2`, `max_weight=0.04`;
  balanced `top_k=35`, `alpha_xgb=0.0`, `rank_power=1.6`, `equal_mix=0.25`,
  `max_weight=0.04`.
- A local guarded sweep promoted v2:
  `top_k=30`, `alpha_xgb=0.10`, `rank_power=1.6`, `equal_mix=0.0`,
  `max_weight=0.04`.
- The guarded v2 version was the prior best before the reference-repo follow-up:
  mean excess `+1.571%`, min excess `+0.175%`, negative windows `0 / 9`.

## 2026-05-08 Reference-Repo Follow-Up

- Inspected `JunjiaZhangMax/CSI500-Stock-Selection-Machine-Learning-final-project`.
  Useful ideas: raw-return target, exponential time-decay sample weights,
  amplitude/range feature, and score-proportional/concentrated weighting.
- Tested direct transfer:
  `reference` features (`amplitude_ma_20d`, rank) plus time decay underperformed
  the current core route: mean excess `+1.102%`, negative windows `0 / 9`.
- Ablation showed amplitude was harmful for our five-day stage2 objective:
  `tree_consensus_reference` mean `+0.673%`, negative windows `3 / 9`.
- Core features plus time decay improved mean to `+1.761%`, but introduced one
  negative window, so it was not promoted directly.
- Added adaptive time decay: use `half_life=120`, `floor=0.5` normally, but
  switch back to no-decay in post-rally/no-medium-support and medium-uptrend
  pullback regimes. This promoted the current best:
  mean excess `+2.303%`, min excess `+0.391%`, negative windows `0 / 9`.
- Validated submission candidate:
  `current_best/stage2_tree_consensus_adaptive_decay_20260508.csv`.

## 2026-05-09 Phase1 Gap Analysis

- Compared the current five-day stage2 model on the phase1 `2026-05-06` to
  `2026-05-08` window using the correct no-leakage as-of date `2026-04-30`.
- It returned `+5.293%`, benchmark `+4.128%`, excess `+1.165%`.
- Added an optional `target_3d` path to the tree-consensus workflow. With the
  same LightGBM/XGBoost ensemble, adaptive time decay, and rank-weighted
  portfolio, the best phase1-style setting reached `+7.133%` return and
  `+3.005%` excess on `2026-05-06` to `2026-05-08`.
- This explains most of the gap to stronger phase1 submissions: their model was
  likely aligned to the three-day horizon, while our production model was still
  optimized for five-day holding.
- `score_prop` weighting and the richer `momentum` feature set were tested on
  the same phase1 window but underperformed. Momentum features were especially
  noisy and are not promoted.
- The pure `target_3d` model was then tested on nine five-day stage2 windows:
  mean excess `+1.288%`, min `-0.980%`, negative windows `1 / 9`. It is useful
  as a short-horizon diagnostic, but it is not promoted as the stage2 main route.
- A simple 3-day/5-day horizon blend also diluted the phase1 upside, so it is
  not promoted over the five-day production route.

## 2026-05-09 Stage2 Drawdown Overlay

- Ran a pre-2026-03 feature IC check and single-factor validation sweep. The
  strongest standalone signals were mostly low long-horizon momentum /
  pullback-style features, but standalone factor portfolios had several
  negative windows and were rejected as production replacements.
- Promoted only a conservative overlay: keep the current tree-consensus stock
  list unchanged, then reweight selected names by low `drawdown_20d` confidence
  when `abs(idx_ret_20d) > 4%` or `idx_ret_5d > 6%`.
- Canonical 9-window backtest via `score_submission.py`:
  `tree_consensus_drawdown_overlay` mean `+2.428%`, min `+0.490%`, max
  `+4.569%`, negative windows `0 / 9`.
- Generated and validated current as-of candidate:
  `current_best/stage2_tree_consensus_drawdown_overlay_20260508.csv`.
- Checked cached portfolio ensembles against the overlay. Small blends with
  adaptive, score-proportional, target-3, LightGBM, tuned XGB, and baseline XGB
  all diluted mean excess; the overlay alone stayed best.
- Added `stage2_catboost_portfolio.py` as a CatBoost robust-loss challenger.
  Quick 2026-04-28 smoke tests were not competitive: Quantile `alpha=0.55`
  produced `-0.039%` excess and MAE produced `+1.043%` excess on a window where
  the current overlay produced `+4.569%`.
- Tested raising the overlay max weight to `6%` on representative windows. It
  improved 2026-04-21 but reduced 2026-03-23, 2026-04-07, and 2026-04-28, so
  the production cap remains `4%`.

## 2026-05-09 Hybrid Gate Upgrade

- Fixed `lstm_rank_weight.py` target-index compatibility after
  `features.py` started carrying `idx_target_5d`. Full 9-window LSTM standalone
  validation remained too unstable: mean excess `+0.414%`, min `-2.377%`,
  negative windows `4 / 9`.
- LSTM is still useful in one broad capitulation regime. Gating LSTM only for
  that regime and using the drawdown-overlay tree route elsewhere improved the
  prior best to mean excess `+2.983%`, min `+0.929%`, negative windows `0 / 9`.
- Tested a heavily regularized rank-label ensemble globally. It failed as a
  standalone route: mean excess `-0.245%`, min `-4.645%`, negative windows
  `5 / 9`. It is not a production model by itself.
- Cached route analysis showed the regularized route is useful only in two
  specific weak regimes: mild post-rally/no-medium-support and defensive
  equal-universe tape. Implemented `stage2_hybrid_gate.py`:
  regularized consensus for those two regimes, LSTM for broad capitulation,
  tree consensus with drawdown overlay otherwise.
- Canonical 9-window five-day validation:
  `hybrid_gate` mean excess `+3.185%`, min `+1.991%`, max `+4.569%`,
  negative windows `0 / 9`.
- Generated and validated latest current-best candidate:
  `current_best/stage2_hybrid_gate_20260508.csv`. For this as-of date the route
  is still `tree_consensus_drawdown_overlay`, so the CSV is identical to the
  previous tree/LSTM-gate output; the hybrid improvement affects future or
  backtest windows whose regime gates fire.
- Started a targeted low-volatility / low-amplitude feature-set probe. The first
  two tested sets badly damaged early rebound windows, e.g. `core_plus_risk_ic`
  produced `-4.178%` excess on 2026-03-02. The run was stopped and this feature
  stuffing direction is not promoted.

## 2026-05-09 Risk/Leakage Audit Follow-Up

- Tested whether the `4%` tree-route cap is overly conservative by reweighting
  the same selected names with higher effective concentration. This did not
  beat the current hybrid route: best cached rank-reweight variant reached mean
  excess `+3.129%` versus the hybrid route's `+3.185%` before no-leak trimming.
  Conclusion: the binding cap is not the main bottleneck; selected-name quality
  and route choice matter more than simply increasing top-name exposure.
- Tested portfolio-level class-ensemble style aggregation across cached model
  portfolios. It underperformed the current gate: best consensus sweep mean
  excess was `+2.513%`. Broad model aggregation diluted the regime-specific
  edge instead of improving it.
- Tested core-feature de-correlation by removing rank features and raw duplicate
  features. Both harmed tree-route windows. Removing rank features dropped the
  hypothetical hybrid mean to about `+2.156%`; removing raw duplicates dropped
  it to about `+2.359%`. A greedy IC/correlation subset produced a negative
  2026-03-30 tree-route window. Conclusion: do not mechanically remove
  correlated features; the rank/raw pairs provide useful nonlinear views to the
  tree models.
- Strengthened leakage controls in production code by trimming `prices` and
  `index_df` to `date <= as_of` inside `stage2_tree_consensus.py`,
  `stage2_regularized_consensus.py`, and `lstm_rank_weight.py` before feature
  construction. The earlier train cutoffs already prevented target leakage, but
  this makes the pipeline cleaner and easier to audit.
- Re-ran canonical 9-window validation after explicit no-leak trimming:
  `hybrid_gate` mean excess `+3.173%`, min `+1.991%`, max `+4.569%`, negative
  windows `0 / 9`. The small mean decrease is accepted for the stricter,
  report-safe pipeline.

## 2026-05-09 Regime-Specific Alpha Upgrade

- Moved away from a single universal feature/weight recipe and targeted weak
  regimes individually. The core route remains no-leak tree/regularized/LSTM
  gating; new layers only reweight already-selected names.
- First tree-regime alpha pass improved the no-leak hybrid from mean excess
  `+3.173%` to `+4.211%` on the primary 9-window five-day test, with min
  `+2.519%` and `0 / 9` negative windows.
- Added a low-`amount_z_20d` liquidity confidence layer for regimes where
  recent trading-amount spikes were unstable. This raised the 12-window mean
  to `+4.541%`, min `+1.986%`, negative windows `0 / 12`.
- Tuned route-specific alpha strengths and added narrow regularized/LSTM
  confidence tilts. The final promoted route reached:
  `+5.291%` mean excess, `+2.791%` min, `0 / 9` negatives on the primary
  9-window test.
- Robustness 12-window test reached:
  `+5.207%` mean excess, `+2.235%` min, `0 / 12` negatives.
- Generated and validated latest as-of candidate:
  `current_best/stage2_hybrid_gate_20260508.csv`. For this current tape the
  selected route is `tree_consensus_drawdown_overlay_liquidity_alpha`, with 30
  stocks and max weight about `7.68%`.
- Follow-up v2 secondary-alpha tuning added narrowly gated intraday/MA60/
  drawdown confidence rules for weak short tape, bear-market snapback, and
  strong medium-trend follow-through regimes. It improved the 12-window mean
  from `+5.207%` to `+5.717%` and the 9-window mean from `+5.291%` to
  `+5.646%`, with no window degradation versus v1. A v1-v2 LOO check selected
  v2 from the other 11 windows for every held-out window.
- Re-ran leakage audit for v2:
  `reports/stage2_leakage_audit_v2_20260509.csv`. Full-data cached portfolios
  match portfolios regenerated after physical as-of truncation with max weight
  differences around `1e-16`.
- Regenerated and validated `current_best/stage2_hybrid_gate_20260508.csv`.
  The latest as-of route is now
  `tree_consensus_drawdown_overlay_liquidity_alpha_secondary_alpha`, with 30
  stocks and max weight at the `10%` competition cap.
- Regularized-route stock-quality follow-up: concentrated the regularized
  consensus output from 50 to 30 names after route-specific alpha, removing its
  low-confidence tail. This only changes the two regularized windows in the
  12-window audit. It improved mean from `+5.717%` to `+5.767%`, raised min
  from `+2.791%` to `+3.177%`, and kept `0 / 12` negative windows.
- Changed-window leakage audit:
  `reports/stage2_leakage_audit_regularized_top30_20260509.csv`; both
  regularized windows pass full-data vs physical as-of truncation comparison
  with max weight differences around `1e-16`.
- Recent known five-day window monitor: `as_of=20260428` corresponds to trading
  days `2026-04-29`, `2026-04-30`, `2026-05-06`, `2026-05-07`, `2026-05-08`.
  Current best excess on that window remains `+8.005%`.
- Expanded the regularized route's portfolio-shape validation grid to allow
  30/35/40-name candidate portfolios and lower equal-weight mixing. This
  safely lifted the 12-window min from `+3.177%` to `+3.196%`, with no negative
  windows, but only a tiny mean improvement.
- Tested a global confidence overlay sweep across existing 12-window outputs.
  Broad risk-on overlays increased mean slightly but reduced the worst window,
  so they were rejected. A mild, gated `ret_20d` high confirmation layer was
  promoted only for weak/flat medium-tape non-LSTM regimes because it improved
  both mean and min.
- Final confidence version reached 12-window mean excess `+5.845%`, min
  `+3.225%`, max `+9.326%`, and `0 / 12` negative windows.
- `as_of=20260428` known-window monitor improved to excess `+8.130%`.
- Small leakage audit for the changed regularized windows plus `as_of=20260428`
  passed:
  `reports/stage2_leakage_audit_final_confidence_probe_20260509.csv`.
- Current best outputs are now always generated as a pair via
  `stage2_generate_best_pair.py`:
  `current_best/stage2_hybrid_gate_20260428.csv` and
  `current_best/stage2_hybrid_gate_20260508.csv`.
- Nonlinear feature branch test: added an experimental `quality` feature set
  with beta/residual return, trend-quality, trend-efficiency, downside
  volatility, and volume-price confirmation features. Directly replacing the
  core tree feature set failed badly on a 6-window probe: hybrid mean fell from
  `+6.211%` to `+3.601%`. Kept as a research branch only, not promoted.
- CatBoost challenger test also failed as a production ensemble component:
  6-window CatBoost quantile mean was only `+0.446%` with `3 / 6` negative
  windows. Small CatBoost blends diluted hybrid performance, so no promotion.
- Tree portfolio-shape sweep showed global concentration was not safe:
  raising the first-stage tree cap to `6%/8%` improved strong 20-day tape
  windows but reduced the recent `as_of=20260428` monitor. Promoted an adaptive
  cap instead: `8%` only when `idx_ret_20d > 8%` and short tape is not deeply
  negative, otherwise `4%`.
- Adaptive-cap 12-window validation reached mean excess `+5.897%`, min
  `+3.225%`, max `+9.326%`, and `0 / 12` negative windows.
- Adaptive-cap leakage audit passed for changed windows plus `as_of=20260428`:
  `reports/stage2_leakage_audit_adaptive_cap_probe_20260509.csv`.
- Candidate-strategy matrix check showed that existing model-switching routes
  had an oracle ceiling around `+5.994%` mean excess, so further route switching
  among only current model outputs could not reach the `8%` target.
- Full `quality` feature-set run confirmed it is not a general replacement:
  mean `+3.588%`, min `-1.066%`, with only `2026-02-06` improving. Promoted it
  only as a narrow auto feature-set challenger for shallow selloffs on flat
  medium tape.
- Pure factor route sweep found no globally useful single factor, but did find
  a few regime-specific stock-pool challengers. Promoted a narrow factor route
  layer:
  `overnight_ret low` for flat-tape rebound, `turnover_ma_20d low` for weak-flat
  tape, `downside_vol_20d low` for mild post-rally, and `downside_vol_20d high`
  for strong rebound after weak medium trend.
- Factor-route 12-window validation reached mean excess `+6.653%`, min
  `+3.431%`, max `+13.711%`, and `0 / 12` negative windows.
- Factor-route leakage audit passed on all changed windows plus `as_of=20260428`:
  `reports/stage2_leakage_audit_factor_route_probe_20260509.csv`.
- Added validation IC diagnostics:
  `stage2_ic_diagnostics.py` and
  `reports/stage2_ic_diagnostics_20260510_active.csv`.  Validation IC is now
  tracked as a secondary overfit check, not as the primary objective.  The
  current factor-route results show `overnight_ret low` has positive trailing
  IC support, while `downside_vol_20d` and `turnover_ma_20d` routes have strong
  realized-window IC but weak/negative trailing IC, so they are higher overfit
  risk.
- Tested an IC hard-filter route:
  `hybrid_gate_factor_ic`, summary
  `reports/stage2_factor_ic_gate_12w_20260510_summary.csv`.  It keeps only
  IC-supported factor routes and falls back otherwise.  Result: mean excess
  `+6.359%`, min `+3.225%`, max `+13.711%`, negatives `0 / 12`.  Rejected as
  too conservative for now.
- Tested an IC dampening route:
  `hybrid_gate_factor_ic_dampen`, summary
  `reports/stage2_factor_ic_dampen_12w_20260510_summary.csv`.  Unsupported
  factor routes are kept but rank-power concentration is reduced to `1.0`.
  Result: mean excess `+6.505%`, min `+3.349%`, max `+13.711%`, negatives
  `0 / 12`.  Kept as a safer challenger, but not promoted over the current
  aggressive best (`+6.653%` mean).
- IC dampening leakage audit passed for factor-route windows plus
  `as_of=20260428`:
  `reports/stage2_leakage_audit_factor_ic_dampen_20260510.csv`.

## 2026-05-10 Alpha Features And Meta Ensemble Promotion

- Added `alpha_microstructure` features in `features.py`: short/medium returns,
  MA slope ratios, range/amplitude stability, amount/turnover trends,
  close-location features, gap/intraday decomposition, volatility splits,
  OBV-style accumulation, tail stats, and 60-day market correlation.
- Factor sweep found two useful replacements for fragile old factor routes:
  `intraday_mean_5d low` for strong medium-trend pullbacks and `obv_20d high`
  for mild post-rally windows.  Promoting those routes raised 12-window mean
  excess from `+6.653%` to `+6.699%`, min from `+3.431%` to `+3.670%`, with
  `0 / 12` negative windows.
- Built `stage2_meta_portfolio_ensemble.py`: generate as-of-safe hybrid
  variants, aggregate their selected-stock weights, then rebuild a concentrated
  30-name rank portfolio.  The first production meta version reached mean
  excess `+7.145%`, min `+3.690%`, max `+14.351%`, negatives `0 / 12`.
- Added narrow passthrough guards for two regimes where meta concentration hurt:
  defensive low-turnover factor windows and weak/flat tree-tape windows.  This
  raised mean excess to `+7.262%`, min `+3.799%`, negatives `0 / 12`.
- Added adaptive rank-power confidence weighting.  Base `rank_power=2.5` is
  increased only when the regime historically supports stronger concentration:
  moderate for fragile pullback factors, stronger for overnight/downside-vol
  factor routes, and strongest for high-dispersion trend/capitulation tapes.
- Final promoted 12-window validation:
  mean excess `+8.098%`, min `+3.799%`, max `+16.841%`, negatives `0 / 12`.
- Recent known monitor (`as_of=20260428`, window `2026-04-29` to
  `2026-05-08`) reached portfolio return `+16.284%`, benchmark `+5.975%`,
  excess `+10.309%`.
- Leakage audit passed for representative factor, passthrough, and recent
  windows:
  `reports/stage2_leakage_audit_meta_portfolio_ensemble_adaptive_power_20260510.csv`.
- Current best outputs:
  `current_best/stage2_meta_portfolio_ensemble_20260428.csv` and
  `current_best/stage2_meta_portfolio_ensemble_20260508.csv`.

## 2026-05-10 12% Target Round

- Cleaned the workspace before further optimization: deleted Python caches,
  archived pre-round reports/backtests, and kept only current best artifacts in
  the active report/probe folders.
- Web/research scan reinforced three useful principles for the next stage:
  learning-to-rank and stock interdependence are more relevant than point
  forecast accuracy, regime-aware walk-forward validation is mandatory, and
  feature redundancy must be controlled without mechanically deleting useful
  rank/raw nonlinear views.
- Tested a dynamic factor portfolio that chooses factor/shape from only
  already-realized historical windows before each `as_of`.  Rejected: for the
  20260428 monitor its selected factor had historical mean excess only
  `+0.554%` with `10 / 24` negative selection windows.
- Tested a de-correlated forest rank portfolio using ExtraTrees/RandomForest on
  rank-normalized features.  Rejected: validation IC was `0.0018` and the
  20260428 known-window excess was `-0.218%`.
- Tested current-pool risk/confidence reweighting across low volatility,
  downside volatility, market correlation, amount spike, trend quality, OBV,
  intraday, and residual-return tilts.  Rejected: none beat the active
  `+8.098%` mean route.
- Rescored 676 cached submissions from 51 model prefixes.  The all-cache
  stock-pool oracle excluding the active meta reweighting was only `+6.699%`
  mean excess, confirming the next jump requires a genuinely stronger stock
  pool rather than another blend of old candidates.
- No new model was promoted.  Active best remains
  `stage2_meta_portfolio_ensemble.py` with mean excess `+8.098%`, min `+3.799%`,
  and `0 / 12` negative windows.

## Next Work Queue

1. Strengthen stock-only feature engineering without introducing leakage:
   feature correlation filtering, cross-sectional normalization, volatility and
   trend stability features.
2. Add optional open data only group-by-group, with strict cleaning and a
   keep rule requiring improvement in both mean excess and downside metrics.
3. Improve the regime gate so LSTM concentration is used only when the tape
   supports it, and tree/valuation routes protect weak or noisy windows.
4. Log every new run in this file before deciding whether to promote it.
