# Intraday (DXY)

## What this folder does

**Merge** manually sourced DXY 1-minute bars into a canonical series, **map** article timestamps to bars and forward returns, and **label** “true” criticality/direction from short-horizon moves. This is the **DXY side of enrichment** for evaluation (distinct from **macro** merge in [`article_extract/`](../article_extract/README.md)).

## What lives here

| File | Role |
|------|------|
| **`merge_dxy_intraday.py`** | Normalizes local DXY CSVs → repo-root `data/dxy_intraday_min.csv` |
| **`map_intraday.py`** | Joins classified articles + `dxy_intraday_min.csv` → mapped CSV with horizons |
| **`map_gt_intraday.py`** | Maps DXY onto ground-truth example CSV |
| **`label_true_criticality.py`** | Labels `true_criticality` / `true_direction` from 15m moves vs rolling SD |

## Role in the pipeline

| Stage | Role |
|--------|------|
| Orchestrator | [`run_key_pipeline.py`](../run_key_pipeline.py) step **`intraday`** (runs `map_intraday`) |
| DXY inputs | **Manual** downloads: place files under [`../data/dxy_intraday/`](../data/dxy_intraday/README.md), then run `merge_dxy_intraday.py` |

## Current I/O expectations

- **Input:** `data/dxy_intraday_min.csv` (produced by `merge_dxy_intraday.py` from files under `key_pipeline/data/dxy_intraday/` or legacy `data/dxy_intraday/*.csv`).  
- **Article side:** paths like `data/apr_jan_res.csv` / `data/apr_jan_mapped.csv` as in `map_intraday.py` defaults.  
- Run from **repository root**.

## Placeholder note

`merge_dxy_intraday.py` is intentionally **limited**: it does not fetch markets; it only reshapes what you already have. A future **API feed** would replace the manual drop + merge step while keeping the same **contract** for downstream scripts (`Time`, `Open`, sorted, timezone rules aligned with `map_intraday.py`).

## Target data-state model (reference)

Intraday joins sit with **enriched** evaluation-ready tables (macro + prices). Conceptually:

- **`/data/raw/`** — untouched scraped data  
- **`/data/processed/`** — cleaned articles  
- **`/data/enriched/`** — macro + (for this repo) DXY intraday alignment for metrics  
- **`/results/`** — evaluation outputs  
- **`/archive/`** — deprecated experiments  

**Storage:** files today; **database** long-term.

## See also

- [`../data/dxy_intraday/README.md`](../data/dxy_intraday/README.md) — where to put manual DXY CSVs  
- [`../result analysis/README.md`](../result%20analysis/README.md) — evaluation notebooks  
