# Predict

## What this folder does

**LLM-based classification** for DXY-relevant articles (e.g. Claude), with **consensus / macro context** from Hugging Face–style extraction in `consensus_extract_hf.py`.

## What lives here

| File | Role |
|------|------|
| **`claude_dxy_predict.py`** | Main classifier module; can be run as a script (`__main__`) |
| **`consensus_extract_hf.py`** | Shared helpers for consensus/context features |

## Role in the pipeline

| Stage | Role |
|--------|------|
| Orchestrator | [`run_key_pipeline.py`](../run_key_pipeline.py) step **`predict`** |
| Before | [`article_extract/`](../article_extract/) (cleaned + optionally macro-merged CSV) |
| Next | [`intraday/`](../intraday/README.md) to attach DXY minute bars |

## Current I/O expectations

Classifier inputs/outputs are configured **inside** `claude_dxy_predict.py` and related paths (often under `data/...`). Run from **repository root**.

## Orchestrator

`run_key_pipeline.py` step **`predict`** runs **`python -m key_pipeline.predict.claude_dxy_predict`**, which executes this module’s **`__main__`** block. Input/output CSV paths are the **`process_csv_to_csv(...)` defaults** at the bottom of `claude_dxy_predict.py` — change them there if your run should target different files (e.g. `apr_jan` vs other windows).

## Placeholder note

Naming and paths here are **legacy/working**; behavior is unchanged from the current scripts.

## Target data-state model (reference)

Downstream of **`/data/processed/`** and **`/data/enriched/`** (macro); outputs feed intraday mapping and then **`/results/`** or evaluation notebooks.

**Storage:** files today; **database** preferred long-term.

## See also

- [`../README.md`](../README.md)  
- [`../intraday/README.md`](../intraday/README.md)  
