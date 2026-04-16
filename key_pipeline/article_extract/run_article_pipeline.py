"""
Article pipeline runner (lives under key_pipeline/article_extract/).

  Step 1 – URL resolution + text extraction  → apr_jan_ft.csv
  Step 2 – Scrape actual publish times        → apr_jan_pub_ac.csv
  Step 3 – Clean: keep rows where full_text is not null → apr_jan_cln.csv
  Step 4 – Claude DXY classification          → apr_jan_res.csv  (only if --through classify)

For the default orchestrated order (macro before predict), use --through clean here and run
`python -m key_pipeline.predict.run_apr_jan_classify` after macro merge.
"""

import argparse
import os
import sys
import time
import traceback
import pandas as pd

DIR = "data/dxy_training/claude/"

INPUT_CSV      = DIR + "apr_may_oct_jan.csv"
FT_CSV         = DIR + "apr_jan_ft.csv"
PUB_CSV        = DIR + "apr_jan_pub_ac.csv"
CLN_CSV        = DIR + "apr_jan_cln.csv"

LOG = "pipeline.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ==============================================================================
# STEP 1 – URL resolution + text extraction
# Reuses selenium_try.py helpers. Skips rows that already have both
# actual_link and full_text to avoid clobbering previously-resolved data.
# ==============================================================================

def step1_fetch_and_extract():
    log("=== STEP 1: URL resolution + text extraction ===")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    import trafilatura
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from tqdm import tqdm

    df = pd.read_csv(INPUT_CSV)
    log(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Identify rows that still need URL resolution
    needs_url = df["actual_link"].isna()
    log(f"  Rows needing URL resolution: {needs_url.sum()}")
    log(f"  Rows with existing actual_link (skipping Selenium): {(~needs_url).sum()}")

    if needs_url.sum() > 0:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(options=options)

        resolved = []
        try:
            for url in tqdm(df.loc[needs_url, "link"], desc="Fetching URLs"):
                try:
                    driver.get(url)
                    WebDriverWait(driver, 10).until(
                        lambda d: "news.google.com" not in d.current_url
                    )
                    resolved.append(driver.current_url)
                except Exception:
                    resolved.append(url)   # fallback: keep original
                time.sleep(0.5)
        finally:
            driver.quit()

        df.loc[needs_url, "actual_link"] = resolved

    # Extract text for all rows that are missing full_text
    needs_text = df["full_text"].isna()
    log(f"  Rows needing text extraction: {needs_text.sum()}")

    def extract_text(url):
        if pd.isna(url) or url == "" or "news.google.com" in str(url):
            return None
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            return text if text else None
        except Exception:
            return None

    if needs_text.sum() > 0:
        from tqdm import tqdm
        texts = []
        for url in tqdm(df.loc[needs_text, "actual_link"], desc="Extracting text"):
            texts.append(extract_text(url))
        df.loc[needs_text, "full_text"] = texts

    df.to_csv(FT_CSV, index=False)
    filled = df["full_text"].notna().sum()
    log(f"Saved {len(df)} rows to {FT_CSV}  (full_text filled: {filled}/{len(df)})")


# ==============================================================================
# STEP 2 – Scrape actual publish times via scrape_publish_time.py
# ==============================================================================

def step2_scrape_times():
    log("=== STEP 2: Scraping publish times ===")

    # scrape_published_times appends .csv to out_file_name, so pass without extension
    from key_pipeline.article_extract.scrape_publish_time import scrape_published_times
    scrape_published_times(
        in_file_name=FT_CSV,
        out_file_name=PUB_CSV,   # function will save to PUB_CSV + ".csv"
    )

    actual_path = PUB_CSV + ".csv"
    if os.path.exists(actual_path):
        # Rename to the expected clean path (drop the double .csv)
        os.replace(actual_path, PUB_CSV)
        log(f"Renamed {actual_path} → {PUB_CSV}")
    else:
        log(f"Output already at expected path: {PUB_CSV}")


# ==============================================================================
# STEP 3 – Clean: rows with full_text not null
# ==============================================================================

def step3_clean():
    log("=== STEP 3: Cleaning CSV ===")

    df = pd.read_csv(PUB_CSV)
    log(f"Loaded {len(df)} rows from {PUB_CSV}")

    cln = df[df["full_text"].notna()].copy().reset_index(drop=True)
    cln.to_csv(CLN_CSV, index=False)
    log(f"Saved {len(cln)} rows to {CLN_CSV}  ({len(df) - len(cln)} dropped — null full_text)")


# ==============================================================================
# STEP 4 – Claude DXY classification
# ==============================================================================

def step4_classify():
    log("=== STEP 4: Running Claude DXY predict ===")

    from key_pipeline.predict import claude_dxy_predict as cdp
    cdp.process_csv_to_csv(
        input_csv_path=CLN_CSV,
        output_csv_path=DIR + "apr_jan_res.csv",
        ground_truth_csv_path=DIR + "gt_example_ft.csv",
        max_few_shot_examples=39,
        test_row_index=None,
        verbose=False,   # suppress per-article output to keep log readable
    )
    log("Step 4 complete.")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="URL resolution, full text, publish times, optional clean-only or full classify."
    )
    ap.add_argument(
        "--through",
        choices=("clean", "classify"),
        default="classify",
        help="clean = steps 1–3 only (no LLM). classify = include step 4 Claude classification.",
    )
    args = ap.parse_args()

    log("Pipeline starting.")
    start = time.time()

    steps = [
        ("step1_fetch_and_extract", step1_fetch_and_extract),
        ("step2_scrape_times",      step2_scrape_times),
        ("step3_clean",             step3_clean),
        ("step4_classify",          step4_classify),
    ]
    if args.through == "clean":
        steps = steps[:3]

    for name, fn in steps:
        try:
            fn()
        except Exception:
            log(f"ERROR in {name}:\n{traceback.format_exc()}")
            log("Pipeline aborted.")
            sys.exit(1)

    elapsed = (time.time() - start) / 3600
    log(f"Pipeline complete. Total time: {elapsed:.2f}h")
