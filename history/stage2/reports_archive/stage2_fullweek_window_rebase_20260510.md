# Stage2 Full-Workweek Window Rebase

This note separates the old five-trading-day validation setup from the new
strict full-workweek diagnostic requested on 2026-05-10.

## Window Definition

- Old rolling setup: take the next 5 trading dates after each `as_of`, regardless
  of weekends or exchange holidays inside the calendar span.
- New full-workweek setup: keep only windows where the 5 evaluation dates are
  Monday through Friday with no calendar gap, i.e. `start.weekday()==Monday`,
  `end.weekday()==Friday`, and `end-start == 4 calendar days`.

## Old 12-Window / 9-Window Status

- The old 12-window report had only 2 complete Mon-Fri windows.
- The remaining 10 old 12-window windows crossed a weekend and/or a holiday.
- The old 9-window report had 0 complete Mon-Fri windows.
- Therefore old 9w/12w excess numbers should not be mixed with the new
  full-workweek validation numbers.

## Latest Complete Full-Week Windows

The latest 12 complete windows available in local data are:

| as_of | start | end |
|---|---|---|
| 2026-01-09 | 2026-01-12 | 2026-01-16 |
| 2026-01-16 | 2026-01-19 | 2026-01-23 |
| 2026-01-23 | 2026-01-26 | 2026-01-30 |
| 2026-01-30 | 2026-02-02 | 2026-02-06 |
| 2026-02-06 | 2026-02-09 | 2026-02-13 |
| 2026-02-27 | 2026-03-02 | 2026-03-06 |
| 2026-03-06 | 2026-03-09 | 2026-03-13 |
| 2026-03-13 | 2026-03-16 | 2026-03-20 |
| 2026-03-20 | 2026-03-23 | 2026-03-27 |
| 2026-03-27 | 2026-03-30 | 2026-04-03 |
| 2026-04-10 | 2026-04-13 | 2026-04-17 |
| 2026-04-17 | 2026-04-20 | 2026-04-24 |

The local data currently has no later complete Mon-Fri window because the dates
around 2026-05-01 to 2026-05-05 are exchange holidays/weekend gaps.

## New Experiments

- `stage2_backtest_5day.py --full-week-only` now selects strict complete
  Monday-Friday windows.
- `stage2_fullweek_tree_portfolio.py` trains only on historical as-of rows whose
  next 5 trading dates are a complete Monday-Friday workweek.
- The full-week tree route uses cross-sectional rank normalization, Spearman
  correlation filtering, robust LightGBM/XGBoost blend, and no external data.

## Current Full-Week Results

Recent 6 complete Mon-Fri windows:

| model | mean excess | min excess | negative windows |
|---|---:|---:|---:|
| meta_portfolio_ensemble | +1.542% | +0.019% | 0 |
| hybrid_gate | +1.297% | -0.784% | 1 |
| fullweek_tree_aggressive | +0.800% | -1.375% | 3 |
| fullweek_tree | +0.677% | -1.358% | 3 |

Blend diagnostic:

- Best mean remains pure `meta_portfolio_ensemble`: mean `+1.542%`.
- Best min blend was roughly `60% meta + 40% fullweek_tree_aggressive`, but mean
  fell to `+1.245%`; this improves floor only marginally and does not approach
  the `+10%` target.

Single-factor diagnostic:

- Best mean factor was `ret_1d high`: mean `+2.082%`, min `-2.524%`.
- Best floor among high-mean factors was `ret_3d high`: mean `+1.524%`, min
  `-0.554%`.
- No single internal feature approached `+10%` mean or `+5%` floor.

Canonical 12-window rerun for the current old-route main model:

| model | mean excess | min excess | max excess | negative windows |
|---|---:|---:|---:|---:|
| meta_portfolio_ensemble | +1.764% | -3.009% | +6.872% | 2 |

Rolling weekly adaptive factor sweep:

| route | mean excess | min excess | max excess | negative windows |
|---|---:|---:|---:|---:|
| best adaptive weekly factor blend | +1.038% | -2.863% | +4.096% | 3 |

The adaptive factor blend used only historical complete-week windows before each
test `as_of` to select factors.  Its best setting selected a small group of
short-horizon momentum/reversal and gap features, but it still underperformed
the old-route meta ensemble under the strict full-workweek metric.

## Route Implication

The old high excess route should not be considered validated for uninterrupted
workweeks.  Under the new full-workweek distribution, current internal
price/volume features show weak alpha.  The next valid optimization direction is
not more concentration or small weight tuning; it is a different weekly alpha
design, likely focused on Friday close to next-week continuation/reversal,
weekly market regime, and feature interactions learned only from complete-week
samples.
