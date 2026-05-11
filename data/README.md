# Data Layout

This directory holds the active official competition data cache used by the
Stage2 portfolio generator.

## Expected Files

- `constituents.csv`: current CSI500 universe.
- `prices.parquet`: stock daily bars.
- `index.parquet`: CSI500 benchmark index daily bars.

## Refresh

Before a deadline, refresh through the official updater:

```bash
python download_data.py --update --end YYYYMMDD
```

For the final Stage2 candidate here, the data cache was checked through
`2026-05-08`.

Do not place noisy exploratory open-data files here.  The final Stage2 route
uses only the official competition data cache in this directory.
