# Article extract

## What this folder does

**URL resolution, full-text extraction, publish-time handling, and cleaning** — plus **macro feature merge** onto the article table used for modeling. This is the **processed** stage (and macro merge contributes to **enriched** data).

## What lives here

| File | Role |
|------|------|
| **`run_article_pipeline.py`** | Orchestrates extract steps; use `--through clean` for extract-only |
| **`merge_macro_features.py`** | Joins macro columns from `articles_with_macro.csv` into the base article CSV |
| **`scrape_publish_time.py`**, **`selenium_try.py`**, **`recover_publish_times.py`**, **`_run_selenium_missing.py`** | Helpers for publish times and text |

## Role in the pipeline

| Stage | Role |
|--------|------|
| Orchestrator | [`run_key_pipeline.py`](../run_key_pipeline.py) steps **`extract`** and **`macro`** |
| Before | [`ingestion/`](../ingestion/) (optional raw pulls) |
| Next | [`predict/`](../predict/) for classification |

**Enriched (macro):** Macro merge lives **here** (`merge_macro_features.py`), not under `intraday/`. DXY minute bars and forward returns are attached later in [`intraday/`](../intraday/README.md) — that is separate from macro enrichment.

## Current I/O expectations

Paths are **hardcoded** in the scripts (e.g. `data/apr_jan_cln.csv`, `data/articles_with_macro.csv`, and paths under `data/dxy_training/claude/` in `run_article_pipeline.py`). Align **extract** output with **macro** input paths (copy/symlink or edit constants) as described in [`../README.md`](../README.md).

Run from **repository root**.

## Placeholder note

This layout is **placeholder**; the desired end state is still **processed → enriched** via macro merge, then predict/intraday, regardless of whether files live under `data/processed/` or `data/enriched/` on disk.

## Target data-state model (reference)

- **`/data/raw/`** — untouched scraped data  
- **`/data/processed/`** — cleaned data with full text and standardized publication times  
- **`/data/enriched/`** — merged with macro variables  
- **`/results/`** — final evaluation output  
- **`/archive/`** — deprecated experiments and retired datasets  

Storage is **file-based** today; a **database** is the long-term ideal for lineage and concurrency.

## See also

- [`../README.md`](../README.md)  
- [`../intraday/README.md`](../intraday/README.md) — DXY mapping after predict  
