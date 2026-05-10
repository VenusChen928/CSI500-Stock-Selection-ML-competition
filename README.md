# CSI500 Stage2 Portfolio Submission

This repository is organized for the Stage2 five-trading-day submission.

Upload exactly this file:

```text
submissions/portfolio.csv
```

All duplicate portfolio CSVs were moved to `history/` so the final upload file
is unambiguous.

## Environment

```bash
conda activate mlcomp-sp26
pip install -r requirements.txt
```

The project was last checked with data through `2026-05-08` in
`data/prices.parquet`.

## Submission Rules

The competition validator expects a UTF-8 CSV with exactly two columns:

```text
stock_code,weight
000001,0.0333
600000,0.0250
```

The active constraints are: current CSI500 universe only, at least 30 positive
weight stocks, non-negative weights summing to `1.0 +/- 1e-4`, and max single
stock weight `0.10`.

## Reproduce Final Portfolio

```bash
python stage2_baseline_guard_ensemble.py \
  --as-of 20260508 \
  --baseline-top-k 0 \
  --out submissions/portfolio.csv \
  --meta-out submissions/stage2/final_report_materials/01_final_portfolio_metadata.csv
python validate_submission.py submissions/portfolio.csv
```

## Final Model Route

`stage2_baseline_guard_ensemble.py` is the final compact submission route.  It
uses only data up to the requested `as_of` date:

- build OHLCV features from `features.py`;
- train the official-style XGBoost baseline from `baseline_xgboost.py`;
- compute as-of-observable market regime statistics;
- choose the baseline top-k and rank-weighted portfolio.

For `as_of=20260508`, the guard selects 30 rank-weighted names.  A final top-k
sanity check kept top30 because the closest historical baseline-routed windows
showed better mean and minimum excess than top35/top40/top50/top60.

## Validation Commands

Run official format validation:

```bash
python validate_submission.py submissions/portfolio.csv
```

Leakage-audit and full-week validation evidence is stored under
`submissions/stage2/final_report_materials/`.

Historical reports are archived under `history/stage2/reports_archive/`.
