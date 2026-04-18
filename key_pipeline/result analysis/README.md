# Result analysis

## What this folder does

**Evaluation and reporting**: notebooks and a CLI script for **signal quality** (directional accuracy, SD-gated metrics, etc.) on **mapped** CSVs (outputs of [`intraday/map_intraday.py`](../intraday/map_intraday.py) or similar).

## What lives here

| File | Role |
|------|------|
| **`eval_signal.py`** | CLI evaluation (metrics on a mapped CSV) |
| **`eval_signal.ipynb`**, **`eval_signal_v2.ipynb`**, **`eval_signal_clean.ipynb`**, **`eval_signal_2b.ipynb`**, **`eval_rolling_sd.ipynb`**, **`eval_gt_ext.ipynb`** | Notebook analyses |

## Role in the pipeline

| Stage | Role |
|--------|------|
| Logical position | **After** intraday mapping — **results** / evaluation |
| Inputs | Typically `data/...` mapped CSVs and `data/dxy_intraday_min.csv` (see each notebook) |

## Current I/O expectations

Paths are **hardcoded or parameterized** inside each notebook and in `eval_signal.py` defaults (e.g. `data/dxy_training/claude/...`). Run from **repository root** unless a notebook sets otherwise.

## Orchestrator caveat (eval step)

[`run_key_pipeline.py`](../run_key_pipeline.py) step **`eval`** runs:

`python -m key_pipeline.evaluation.eval_signal`

There is **no** `key_pipeline/evaluation/` package in this repo; **`eval_signal.py` lives in this folder** (`result analysis/`). The documented `python -m key_pipeline.evaluation.eval_signal` path may **not** resolve until the package layout is fixed or a shim is added. Prefer running the script by path (quote the folder because of the space), for example:

`python "key_pipeline/result analysis/eval_signal.py"`

or invoke the module in a way your environment supports once the package layout matches the `-m` string.

## Placeholder note

This folder name **includes a space** (`result analysis/`). The structure is **legacy/working**; a future rename to `result_analysis/` would require updating notebooks and docs.

## Target data-state model (reference)

- **`/results/`** — final evaluation output (this folder’s role)  
- **`/archive/`** — deprecated experiments  

**Storage:** notebooks + CSV outputs today; **database** for metrics and run lineage long-term.

## See also

- [`../README.md`](../README.md)  
- [`../intraday/README.md`](../intraday/README.md)  
