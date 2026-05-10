# Final Top-K Sanity Check

Question: should the final `2026-05-08` portfolio use more than 30 stocks?

Conclusion: keep 30 names.  The final as-of regime is
`baseline_guard_overheated_high_breadth`, so the relevant comparison is not all
routes, but the historical windows where the guard also falls back to
`baseline_xgb`.

## Similar Historical Baseline-Guard Windows

| as_of | window | reason | top30 | top35 | top40 | top50 | top60 |
|---|---|---|---:|---:|---:|---:|---:|
| 2026-01-23 | 2026-01-26 to 2026-01-30 | overheated high breadth | +1.141% | +0.616% | +0.153% | -0.380% | -0.551% |
| 2026-03-20 | 2026-03-23 to 2026-03-27 | severe broad selloff | +2.845% | +3.045% | +3.098% | +2.916% | +2.695% |

Across these two baseline-routed windows:

| top_k | mean excess | min excess |
|---:|---:|---:|
| 30 | +1.993% | +1.141% |
| 35 | +1.831% | +0.616% |
| 40 | +1.626% | +0.153% |
| 50 | +1.268% | -0.380% |
| 60 | +1.072% | -0.551% |

For the final `2026-05-08` portfolio shape:

| top_k | max weight | effective names | top 5 weight | top 10 weight |
|---:|---:|---:|---:|---:|
| 30 | 6.452% | 22.87 | 30.11% | 54.84% |
| 35 | 5.556% | 26.62 | 26.19% | 48.41% |
| 40 | 4.878% | 30.37 | 23.17% | 43.29% |
| 50 | 3.922% | 37.87 | 18.82% | 35.69% |
| 60 | 3.279% | 45.37 | 15.85% | 30.33% |

Top30 is not equal-weighted and does not hit the 10% cap.  It keeps enough
concentration to preserve the XGBoost rank signal while still satisfying all
competition constraints.
