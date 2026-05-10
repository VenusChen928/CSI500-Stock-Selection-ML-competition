# Stage2 5-Day Replay And Plan

## Goal

Stage2 should optimize the original 5-trading-day excess-return objective.
The stage1 3-day guarded ensemble is useful as a design pattern, but its guard
thresholds and portfolio concentration should not be copied blindly.

## What Worked In Prior 5-Day Runs

The strongest historical 5-day route was `gated_layered_lstm`.

Recent 5-window result:

| model | mean excess | min excess | max excess |
| --- | ---: | ---: | ---: |
| gated_layered_lstm | +2.192% | +0.254% | +4.156% |
| layered_lstm | +0.887% | -5.692% | +4.156% |
| lstm_rank_weight | +0.734% | -5.619% | +3.722% |
| tuned_xgb | +0.411% | -0.853% | +1.782% |
| baseline | -0.655% | -2.202% | +0.909% |

Key interpretation:

- LSTM has the best upside, but pure LSTM can suffer large underperformance.
- The best 5-day design was not pure LSTM; it was a gated route that chooses LSTM only when validation evidence is strong.
- Tree models are weaker on average but useful as a defensive floor.

9-window summary:

| model | mean excess | min excess | max excess |
| --- | ---: | ---: | ---: |
| gated_layered_lstm | +1.668% | -0.853% | +5.933% |
| layered_lstm | +0.766% | -5.692% | +5.933% |
| LightGBM | +0.732% | -1.558% | +4.757% |
| tuned_xgb | +0.724% | -0.853% | +3.820% |
| baseline | -0.151% | -2.202% | +3.765% |

## What Did Not Work Cleanly

Open-data experiments were noisy when used directly:

| route | latest 5-day excess |
| --- | ---: |
| lstm_rank_weight_current_best | +3.722% |
| tuned_xgb_current | +1.439% |
| open_data_ranker | +1.278% |
| open_data_regressor | +0.372% |
| lstm_open_rank_weight | -1.529% |

Conclusion:

- Open data should not be injected directly into the main scoring model without cleaning and ablation.
- Open data is more promising as a risk/gating layer, a low-weight auxiliary feature group, or a post-model filter.

Feature ablation also showed broad feature stuffing is harmful:

| feature set | validation mean | validation min | latest excess |
| --- | ---: | ---: | ---: |
| core | +1.099% | +0.601% | +2.428% |
| core+risk_liquidity | +1.176% | +0.369% | +1.207% |
| core+price_action | +1.259% | +0.512% | +0.760% |
| all | -1.335% | -2.098% | -0.389% |

Conclusion:

- More features can improve validation but hurt latest score if they are correlated/noisy.
- Stage2 feature expansion must use correlation filtering, missingness checks, and out-of-sample ablation.

## Open Data Quality Notes

Available archived open data:

| file | usable role | quality note |
| --- | --- | --- |
| `stock_value_em.parquet` | valuation/risk/size features | Good date coverage from 2023-01-03 to 2026-04-30; low missingness. |
| `market_pb.parquet` | market valuation regime | Excellent long history; market-level only. |
| `stock_fund_flow.parquet` | recent flow/risk/gating | Only 140 trading dates from 2025-09-26; do not use for long-history model training without coverage gating. |
| `qvix_500etf.parquet` | volatility regime | About 68% missing in archived file; must be forward-filled and coverage-tested before use. |

## Stage2 Technical Route

Recommended route:

1. Build a clean 5-day feature panel using only original OHLCV/index plus strict open-data joins.
2. Keep feature groups separate:
   - core technical trend/reversal
   - volatility/risk/liquidity
   - valuation/size
   - market regime
   - fund-flow recent-only
3. Within each group, apply daily cross-sectional robust z-score and correlation pruning.
4. Train multiple base learners for 5-day excess target:
   - LSTM rank/weight sequence model for upside
   - LightGBM Huber/Quantile as defensive tabular learner
   - XGBoost Huber/hist as low-correlation tree learner
   - optional CatBoost Quantile only if runtime is controlled
5. Use a gated/stacked portfolio layer:
   - normal regime: LSTM-led rank-weight portfolio
   - uncertain/high-benchmark regime: blend or fallback to tree consensus
   - high volatility/high drawdown regime: reduce concentration and increase top_k
6. Select policy on rolling 5-day windows using:
   - mean excess
   - min excess
   - std excess
   - number of negative windows

## Immediate Next Experiments

1. Rebuild a clean `stage2_open_data_features.py` module.
2. Re-run 5-day multi-window evaluation on:
   - baseline/tuned XGB
   - LightGBM
   - current LSTM
   - guarded LSTM/tree ensemble
3. Add open-data groups one at a time:
   - valuation only
   - market PB regime only
   - fund-flow only as a gate/filter
   - all cleaned groups with correlation pruning
4. Keep a route only if it improves both:
   - mean excess
   - min excess or negative-window count

