# CNBC writers daily & related structured data

This folder is the **logical** home for “CNBC writers daily” style corpora and taxonomy-side notes. Large CSV trees stay in their existing paths to avoid massive git moves; use this README as the index.

## Daily CNBC article dumps (2020–2025 and ongoing)

- **Canonical location:** `data/dxy_training/cnbc/`  
  One file per calendar day (e.g. `2025-04-10.csv`), suitable for backfills and writer-daily style analysis.
- **Ingestion script:** `key_pipeline/ingestion/gnews_dxy_extract.py` — set `OUTPUT_DIR` to `data/dxy_training/cnbc` (default in repo) so new pulls land next to historical files.

## Subfolders here

- **`daily/`** — Optional staging or symlinks; empty by default. Prefer writing new pulls directly to `data/dxy_training/cnbc/` unless you deliberately isolate an experiment.
- **`taxonomy/`** — Place any exported taxonomy tables, event lists, or calibration CSVs you maintain outside the classifier. The live taxonomy used in production prompts is still embedded in `key_pipeline/predict/claude_dxy_predict.py`.

## Claude / merged training tables

Article-level tables used for resolution, publish times, cleaning, and LLM output live under **`data/dxy_training/claude/`** (for example `apr_may_oct_jan.csv`, `apr_jan_ft.csv`, `apr_jan_res.csv`). Macro-merged and mapped evaluation CSVs often live under **`data/`** or **`key_pipeline/data/`** depending on the experiment; see `key_pipeline/README.md`.
