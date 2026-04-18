# Ingestion

## What this folder does

Google News–oriented pulls (e.g. CNBC/DXY-related queries) using the script in this directory. This is the **ingest → raw** stage: outputs are **raw** scraped rows before the article pipeline cleans and enriches them.

## What lives here

- **`gnews_dxy_extract.py`** — main entry; run as `python -m key_pipeline.ingestion.gnews_dxy_extract` (see orchestrator [`run_key_pipeline.py`](../run_key_pipeline.py), step `ingest`).

## Role in the pipeline

| Stage | Role |
|--------|------|
| Orchestrator | [`run_key_pipeline.py`](../run_key_pipeline.py) step **`ingest`** |
| Next | [`article_extract/`](../article_extract/) for processed articles |

## Current I/O expectations

Outputs are written to paths configured **inside** `gnews_dxy_extract.py` (edit dates/query windows there before running). Run from **repository root** so relative `data/...` paths match the rest of the pipeline.

## Placeholder note

This folder structure is a **working convention**. A future layout may align with a clearer split under `data/raw/`, `data/processed/`, etc., without changing the logical order: **ingestion → raw**.

## Target data-state model (reference)

- **`/data/raw/`** — untouched scraped data  
- **`/data/processed/`** — cleaned data with full text and standardized publication times  
- **`/data/enriched/`** — data merged with macro variables (and intraday joins where applicable)  
- **`/results/`** — final evaluation output  
- **`/archive/`** — deprecated experiments and retired datasets  

Today the repo uses **file-based** CSVs under various `data/...` paths; a **database** would be the natural long-term storage layer instead of flat files.

## See also

- [`../README.md`](../README.md) — full pipeline table  
- [`../article_extract/README.md`](../article_extract/README.md) — next step  
