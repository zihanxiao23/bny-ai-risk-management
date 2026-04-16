# Archive

Items here are **not part of the active `key_pipeline/` workflow** but are kept for history, coursework, or optional CI.

- **`legacy_etl/`** — Former `scripts/` (GNews/Yahoo/combine ETL) and `news_feeds/` (GDELT, Yahoo Finance ingestion). GitHub Actions under `.github/workflows/` that still need these paths were updated to `archive/legacy_etl/news_feeds/...`.
- **`legacy_root_scripts/`** — Older prediction variants (`claude_dxy_predict_v*`, `gemini_*`), one-off merges, and legacy DXY helpers (`process_dxy_articles.py`, `recover_missing_text.py`, etc.).
- **`notebooks/`** — Exploratory or superseded notebooks from the repo root (e.g. `IntradayMapping.ipynb`; intraday logic is in `key_pipeline/intraday/map_intraday.py`).

To restore the old hourly Docker/Make ETL behavior, use `make run-etl` (targets point into `archive/legacy_etl/scripts`).
