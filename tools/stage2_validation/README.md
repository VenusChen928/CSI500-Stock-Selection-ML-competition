# Stage2 Validation Tools

These scripts are not required to build the final portfolio directly.  They are
kept here for reproducible validation and report evidence.

- `stage2_backtest_5day.py`: subprocess-based multi-window evaluator.
  Use `--summary-out` and `--detail-out` together when refreshing final report
  evidence, so aggregate and per-window files keep their documented names.
- `stage2_allocation_ic.py`: recomputes allocation IC from the saved backtest
  portfolio CSVs and the per-window detail file.
- `stage2_fast_backtest_numpy.py`: faster parallel evaluator with numpy scoring.
- `stage2_leakage_audit.py`: compares full-data generation with as-of-truncated
  generation to check leakage.
- `stage2_validation_audit.py`: older route-level audit helper retained for
  report traceability.

Run commands from the repository root, for example:

```bash
python tools/stage2_validation/stage2_leakage_audit.py \
  --as-of 20260109 20260227 20260313 20260327 20260410 20260508 \
  --models baseline_guard_adaptive
```
