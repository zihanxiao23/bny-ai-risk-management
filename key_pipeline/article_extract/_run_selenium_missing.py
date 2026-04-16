"""
Ad-hoc runner: fills missing full_text in data/gt_340.csv by calling
get_all_urls + extract_text from selenium_try.py only on rows where
full_text is NaN. Writes the merged 340-row result to data/gt_340_ft.csv.
"""
import sys, time
sys.path.insert(0, ".")
from key_pipeline.article_extract.selenium_try import get_all_urls, extract_text
import pandas as pd

INPUT  = "data/gt_340.csv"
OUTPUT = "data/gt_340_ft.csv"

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows from {INPUT}", flush=True)

mask = df["full_text"].isna()
n = int(mask.sum())
print(f"{n} rows missing full_text — processing those", flush=True)

todo = df[mask].copy()

print("\n[1/2] Resolving URLs via Selenium...", flush=True)
t0 = time.time()
todo["actual_link"] = get_all_urls(todo)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

print("\n[2/2] Extracting text via trafilatura...", flush=True)
t0 = time.time()
todo["full_text"] = todo["actual_link"].apply(extract_text)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

df.loc[mask, "actual_link"] = todo["actual_link"].values
df.loc[mask, "full_text"]   = todo["full_text"].values

recovered = int(todo["full_text"].notna().sum())
print(f"\nRecovered full_text for {recovered}/{n} rows", flush=True)

df.to_csv(OUTPUT, index=False)
print(f"Saved → {OUTPUT}", flush=True)
