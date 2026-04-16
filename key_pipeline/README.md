# Key pipeline (DXY / CNBC)

Run from the **repository root** so `data/...` paths resolve.

## End-to-end flow (canonical order)

| Step | What it does | Folder / entry |
|------|----------------|------------------|
| 1. **Ingest (Google News)** | Daily CNBC-oriented pulls via pygooglenews | `ingestion/gnews_dxy_extract.py` — `python -m key_pipeline.ingestion.gnews_dxy_extract` |
| 2. **Full text + publish times** | Resolve Google News URLs, trafilatura text, Selenium publish scrape, drop null text | `article_extract/` — orchestrator uses `run_article_pipeline.py --through clean` |
| 3. **Macro merge** | Join macro columns onto the article table used for modeling | `macro/merge_macro_features.py` |
| 4. **Predict** | Claude classifier (+ consensus context from HF extract) | `predict/` — `claude_dxy_predict.py`, `consensus_extract_hf.py`; batch wrapper `run_apr_jan_classify.py` |
| 5. **Map intraday** | Attach DXY minute bars and forward returns | `intraday/map_intraday.py` (+ helpers `label_true_criticality.py`, `merge_dxy_intraday.py`, `map_gt_intraday.py`) |
| 6. **Eval** | Hit-rate / SD-gated metrics on a **mapped** CSV | `evaluation/eval_signal.py` and `evaluation/eval_*.ipynb` |

One command (same order, skips **ingest** and **eval** by default):

```bash
python key_pipeline/run_key_pipeline.py
# same as:
python key_pipeline/run_key_pipeline.py --steps extract,macro,predict,intraday
```

- **ingest** — add to `--steps` when you want Google News (edit dates in `ingestion/gnews_dxy_extract.py` first).
- **eval** — add when you want CLI metrics; pass `--input` on `eval_signal` if needed:  
  `python -m key_pipeline.evaluation.eval_signal --input data/your_mapped.csv`
- **Shorthand `article`** — runs **extract** then **predict** (same as the old bundled article runner with classify).

### Data-path note (macro ↔ predict)

- **Extract** writes cleaned rows to `data/dxy_training/claude/apr_jan_cln.csv`.
- **Macro** (as shipped) reads/writes **`data/apr_jan_cln.csv`**. For macro to sit on the same rows as extract, either copy/symlink that file to the path `merge_macro_features.py` expects, or edit `INPUT_BASE` / `INPUT_MACRO` in `macro/merge_macro_features.py` to point at your real cleaned CSV.
- **Predict** (`run_apr_jan_classify.py`) reads **`data/dxy_training/claude/apr_jan_cln.csv`**. After you align macro paths, point predict’s input at the macro-enriched CSV if that is no longer the claude-dir file.

## Folder layout

```
key_pipeline/
  run_key_pipeline.py          # orchestrator
  ingestion/                   # step 1 (single main script)
  article_extract/             # step 2 (Selenium, scrape, clean, optional classify in one runner)
  macro/                       # step 3
  predict/                     # step 4 (+ RAG variant)
  intraday/                    # step 5
  evaluation/                  # step 6 (CLI + notebooks)
  archived/                    # old predict versions / experiments (not in default flow)
  data/                        # experiment CSVs (not the main repo `data/` tree)
```

## Article extract utilities

| File | Role |
|------|------|
| `run_article_pipeline.py` | Steps 1–4; use `--through clean` for 1–3 only |
| `scrape_publish_time.py` | Publish-time scraping |
| `selenium_try.py` | URL / text helpers |
| `recover_publish_times.py` | Re-scrape publish times when URLs were fixed later |
| `_run_selenium_missing.py` | Small helper for missing rows |

## Merged vs separate

Modules stay **separate**; `run_key_pipeline.py` only sequences subprocesses so you can re-run one stage.

## Legacy / orphans

Older root scripts, GDELT/Yahoo ETL, and exploratory notebooks live under **`archive/`** (see `archive/README.md`). They are not imported by the active flow above.
