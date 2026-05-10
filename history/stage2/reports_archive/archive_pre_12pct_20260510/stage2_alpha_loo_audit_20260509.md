# Stage2 Alpha LOO Audit

## Variant Summary

|              |     mean |      min |      max |   negative_windows |
|:-------------|---------:|---------:|---------:|-------------------:|
| final        | 0.052066 | 0.022350 | 0.093260 |           0.000000 |
| no_secondary | 0.049946 | 0.022350 | 0.093260 |           0.000000 |
| no_route     | 0.049778 | 0.022350 | 0.093260 |           0.000000 |
| no_liquidity | 0.047088 | 0.022220 | 0.080050 |           0.000000 |
| no_regime    | 0.038848 | 0.010830 | 0.069380 |           0.000000 |
| no_alpha     | 0.027481 | 0.007100 | 0.045690 |           0.000000 |

## Final vs Ablations

| comparison               |   mean_delta |   min_delta |   negative_delta_windows |
|:-------------------------|-------------:|------------:|-------------------------:|
| final_minus_no_alpha     |     0.024585 |    0.002630 |                        0 |
| final_minus_no_regime    |     0.013218 |    0.000000 |                        0 |
| final_minus_no_liquidity |     0.004978 |    0.000000 |                        0 |
| final_minus_no_route     |     0.002288 |    0.000000 |                        0 |
| final_minus_no_secondary |     0.002120 |    0.000000 |                        0 |

## Leave-One-Window-Out Selection

| heldout_as_of   | selected_by_other_windows   |   final_rank_on_train |   heldout_selected_excess |   heldout_final_excess | final_would_be_selected   |
|:----------------|:----------------------------|----------------------:|--------------------------:|-----------------------:|:--------------------------|
| 2026-01-30      | final                       |                     1 |                  0.022350 |               0.022350 | True                      |
| 2026-02-06      | final                       |                     1 |                  0.033020 |               0.033020 | True                      |
| 2026-02-13      | final                       |                     1 |                  0.093260 |               0.093260 | True                      |
| 2026-03-02      | final                       |                     1 |                  0.029990 |               0.029990 | True                      |
| 2026-03-09      | final                       |                     1 |                  0.027910 |               0.027910 | True                      |
| 2026-03-16      | final                       |                     1 |                  0.073030 |               0.073030 | True                      |
| 2026-03-23      | final                       |                     1 |                  0.065170 |               0.065170 | True                      |
| 2026-03-30      | final                       |                     1 |                  0.035600 |               0.035600 | True                      |
| 2026-04-07      | final                       |                     1 |                  0.067360 |               0.067360 | True                      |
| 2026-04-14      | final                       |                     1 |                  0.048390 |               0.048390 | True                      |
| 2026-04-21      | final                       |                     1 |                  0.048660 |               0.048660 | True                      |
| 2026-04-28      | final                       |                     1 |                  0.080050 |               0.080050 | True                      |
