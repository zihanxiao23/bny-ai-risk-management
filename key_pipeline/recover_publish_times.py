"""
Re-scrape publish times for rows that now have real article URLs
but are still missing article_published_utc.

This happens because scrape_publish_time.py ran when those rows still
had news.google.com URLs (before recover_missing_text.py resolved them).

Reads:  data/apr_jan_pub_ac.csv
Writes: data/apr_jan_pub_ac.csv  (in-place)
        data/apr_jan_cln.csv     (re-filtered)
"""

import pandas as pd
from tqdm import tqdm
from scrape_publish_time import build_driver, scrape_article_time, to_utc

PUB_CSV = "data/aug_mar_ft_pub.csv"
CLN_CSV = "data/aug_mar_ft_pub_cln.csv"


def main():
    df = pd.read_csv(PUB_CSV)
    print(f"Loaded {len(df)} rows from {PUB_CSV}")

    needs = (
        df["actual_link"].notna()
        & ~df["actual_link"].str.contains("news.google.com", na=False)
        & df["article_published_utc"].isna()
    )
    idx_list = df[needs].index.tolist()
    print(f"Rows needing publish time scrape: {len(idx_list)}")

    if not idx_list:
        print("Nothing to do.")
        return

    driver = build_driver()
    success = 0

    try:
        for idx in tqdm(idx_list, desc="Scraping publish times"):
            url = df.at[idx, "actual_link"]
            raw, via = scrape_article_time(driver, url)
            utc_val = to_utc(raw)

            df.at[idx, "article_published_raw"] = raw
            df.at[idx, "article_published_utc"] = utc_val
            df.at[idx, "found_via"]             = via

            if utc_val:
                success += 1
    finally:
        driver.quit()

    df.to_csv(PUB_CSV, index=False)
    filled = df["article_published_utc"].notna().sum()
    print(f"\nSaved {len(df)} rows → {PUB_CSV}")
    print(f"Newly recovered timestamps : {success} / {len(idx_list)}")
    print(f"Total article_published_utc: {filled} / {len(df)}")

    cln = df[df["full_text"].notna()].copy().reset_index(drop=True)
    cln.to_csv(CLN_CSV, index=False)
    print(f"Saved {len(cln)} rows → {CLN_CSV}")


if __name__ == "__main__":
    main()
