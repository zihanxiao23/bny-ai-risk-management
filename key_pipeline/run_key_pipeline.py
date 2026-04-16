#!/usr/bin/env python3
"""
Single entry point for the DXY key pipeline. Changes cwd to the repository root
so paths like data/dxy_training/claude/ resolve correctly.

Usage (from repo root):
  python key_pipeline/run_key_pipeline.py
  python key_pipeline/run_key_pipeline.py --steps ingest,extract,macro,predict,intraday,eval

Steps (comma-separated; order when listed together is always canonical below):
  ingest   — Google News daily pulls (-m key_pipeline.ingestion.gnews_dxy_extract).
  extract  — URL resolution, full text, publish times, clean CSV (-m ...article_extract.run_article_pipeline --through clean).
  macro    — Merge macro columns (-m key_pipeline.macro.merge_macro_features).
  predict  — Claude batch classify apr/jan (-m key_pipeline.predict.run_apr_jan_classify).
  intraday — Map to DXY minute bars (-m key_pipeline.intraday.map_intraday).
  eval     — CLI metrics on a mapped CSV (-m key_pipeline.evaluation.eval_signal).

Shorthand:
  article  — same as extract,then,predict (full article path including LLM).

Default: extract,macro,predict,intraday (no ingest; eval is often ad hoc).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_step(name: str, argv: list[str]) -> None:
    print(f"\n>>> [{name}] {' '.join(argv)}\n", flush=True)
    subprocess.run(argv, cwd=repo_root(), check=True)


def expand_requested(raw: list[str]) -> list[str]:
    out: list[str] = []
    for s in raw:
        if s == "article":
            out.extend(["extract", "predict"])
        else:
            out.append(s)
    return out


def main() -> None:
    order = ["ingest", "extract", "macro", "predict", "intraday", "eval"]
    parser = argparse.ArgumentParser(description="Run key_pipeline steps in canonical order.")
    parser.add_argument(
        "--steps",
        default="extract,macro,predict,intraday",
        help=f"Comma-separated subset of: {', '.join(order)}, plus shorthand 'article' (=extract+predict).",
    )
    args = parser.parse_args()
    root = repo_root()
    os.chdir(root)

    py = sys.executable
    requested = expand_requested(
        [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    )
    for s in requested:
        if s not in order:
            parser.error(f"Unknown step {s!r}; allowed: {', '.join(order)} plus 'article'")

    dedup: list[str] = []
    for s in requested:
        if s not in dedup:
            dedup.append(s)
    requested = dedup

    for step in order:
        if step not in requested:
            continue
        if step == "ingest":
            run_step("ingest", [py, "-m", "key_pipeline.ingestion.gnews_dxy_extract"])
        elif step == "extract":
            run_step(
                "extract",
                [
                    py,
                    "-m",
                    "key_pipeline.article_extract.run_article_pipeline",
                    "--through",
                    "clean",
                ],
            )
        elif step == "macro":
            run_step("macro", [py, "-m", "key_pipeline.macro.merge_macro_features"])
        elif step == "predict":
            run_step("predict", [py, "-m", "key_pipeline.predict.run_apr_jan_classify"])
        elif step == "intraday":
            run_step("intraday", [py, "-m", "key_pipeline.intraday.map_intraday"])
        elif step == "eval":
            run_step("eval", [py, "-m", "key_pipeline.evaluation.eval_signal"])


if __name__ == "__main__":
    main()
