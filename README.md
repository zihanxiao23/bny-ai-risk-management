# BNY-AI-Risk-Management

Duke MIDS capstone work with BNY Mellon: analysis, modeling, and pipeline code for AI-driven risk management.

## Key pipeline (DXY / CNBC)

The **active** ingestion, article processing, macro merge, intraday mapping, prediction, and evaluation code lives in **`key_pipeline/`**. Start here:

- [key_pipeline/README.md](key_pipeline/README.md)
- One-shot orchestrator (from repo root): `python key_pipeline/run_key_pipeline.py`
- **Google News → daily CNBC-style CSVs:** `key_pipeline/ingestion/gnews_dxy_extract.py`
- **CNBC / taxonomy data index:** [data/cnbc_writers_daily/README.md](data/cnbc_writers_daily/README.md)

## Legacy ETL (archived)

Yahoo Finance, GDELT, and the old `scripts/` GNews combiner were moved to **`archive/legacy_etl/`**. `make run-etl` still runs those three steps for Docker or historical workflows. GitHub Actions paths were updated to match.

## GDELT / Yahoo (reference)

Incremental backfill and resiliency behavior for GDELT are documented in **`archive/legacy_etl/news_feeds/gdelt/README.md`** (config keys such as `summary.backfill_limit`, `summary.max_hf_new_per_run`, and `--fail_on_feed_error`). The former root paths `news_feeds/gdelt` now live under `archive/legacy_etl/news_feeds/gdelt`.
