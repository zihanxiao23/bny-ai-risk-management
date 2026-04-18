# BNY-AI-Risk-Management

Duke MIDS capstone work with BNY Mellon: analysis, modeling, and pipeline code for AI-driven risk management.

## Key pipeline (DXY / CNBC)

**Overview:** End-to-end, file-based workflow for Google News–sourced articles, full-text and publish-time extraction, optional macro enrichment, Claude-based DXY classification, attachment of manually sourced DXY minute bars, and offline evaluation notebooks. Run stages from the **repository root** so `data/...` paths resolve. Details, step table, and data-path notes: **[key_pipeline/README.md](key_pipeline/README.md)**.

**Orchestrator (from repo root):** `python key_pipeline/run_key_pipeline.py`  
(default steps: extract → macro → predict → intraday; add `ingest` or `eval` via `--steps`).

### `key_pipeline/` layout (links)

| Path | Role | README |
|------|------|--------|
| [run_key_pipeline.py](key_pipeline/run_key_pipeline.py) | Sequences pipeline steps as subprocesses | — |
| [ingestion/](key_pipeline/ingestion/) | Google News daily pulls | [ingestion/README.md](key_pipeline/ingestion/README.md) |
| [article_extract/](key_pipeline/article_extract/) | URL/text, publish times, clean CSV; macro merge | [article_extract/README.md](key_pipeline/article_extract/README.md) |
| [predict/](key_pipeline/predict/) | Claude classifier + HF consensus helpers | [predict/README.md](key_pipeline/predict/README.md) |
| [intraday/](key_pipeline/intraday/) | Merge DXY CSVs, map bars to articles, labels | [intraday/README.md](key_pipeline/intraday/README.md) |
| [data/dxy_intraday/](key_pipeline/data/dxy_intraday/) | Manual DXY 1m CSV staging | [data/dxy_intraday/README.md](key_pipeline/data/dxy_intraday/README.md) |
| [result analysis/](key_pipeline/result%20analysis/) | Eval CLI + notebooks (folder name has a space) | [result analysis/README.md](key_pipeline/result%20analysis/README.md) |
| [archived/](key_pipeline/archived/) | Older predict variants / experiments | — |
| [data/](key_pipeline/data/) | Pipeline-adjacent experiment CSVs and staging | see `data/dxy_intraday/` above |

**Also useful**

- **Google News script:** [key_pipeline/ingestion/gnews_dxy_extract.py](key_pipeline/ingestion/gnews_dxy_extract.py)
- **CNBC / taxonomy data index:** [data/cnbc_writers_daily/README.md](data/cnbc_writers_daily/README.md)

## Legacy ETL (archived)

Yahoo Finance, GDELT, and the old `scripts/` GNews combiner were moved to **`archive/legacy_etl/`**. `make run-etl` still runs those three steps for Docker or historical workflows. GitHub Actions paths were updated to match.

## GDELT / Yahoo (reference)

Incremental backfill and resiliency behavior for GDELT are documented in **`archive/legacy_etl/news_feeds/gdelt/README.md`** (config keys such as `summary.backfill_limit`, `summary.max_hf_new_per_run`, and `--fail_on_feed_error`). The former root paths `news_feeds/gdelt` now live under `archive/legacy_etl/news_feeds/gdelt`.
