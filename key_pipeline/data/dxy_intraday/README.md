# DXY intraday data (manual staging)

## What this folder is for

**Local staging** for **manually downloaded** DXY 1-minute bars (e.g. Barchart exports). The pipeline does **not** pull live DXY from an API; you place CSVs here (or under the legacy `data/dxy_intraday/` layout), then run [`intraday/merge_dxy_intraday.py`](../../intraday/merge_dxy_intraday.py) to produce a single **`data/dxy_intraday_min.csv`** at the **repository root** for `map_intraday.py` and eval notebooks.

## What lives here

- **`dxy_data_*.csv`** — typical **combined** export filename pattern consumed by `merge_dxy_intraday.py` (see that script’s `PRIMARY_GLOB`).  
- Other per-period CSVs may appear here depending on your download workflow.

## Role in the pipeline

```text
Manual CSVs (this folder) → merge_dxy_intraday.py → data/dxy_intraday_min.csv → map_intraday.py
```

## Current I/O expectations

See [`intraday/merge_dxy_intraday.py`](../../intraday/merge_dxy_intraday.py) for glob order (primary vs legacy paths). Run merges from **repository root**.

## Placeholder note

This directory is a **working drop zone**, not a long-term archival layout. A future **market-data API** would replace manual drops while preserving the same downstream contract (`data/dxy_intraday_min.csv` or a configured equivalent).

## Target data-state model (reference)

Minute bars are **raw-ish market inputs**; they may be conceptually grouped under **`/data/raw/`** or a dedicated vendor cache in a future layout.

**Storage:** flat files today; **database** or **object store** is preferable for production.

## See also

- [`../../intraday/README.md`](../../intraday/README.md)  
- [`../../README.md`](../../README.md)  
