"""
Recovery script: re-resolve Google News CBM URLs that Selenium failed on,
then extract full_text for those rows.

Reads:  apr_jan_pub_ac.csv  (728 rows still have news.google.com actual_link)
Writes: apr_jan_pub_ac.csv  (in-place update with resolved links + new full_text)
        apr_jan_cln.csv     (re-filtered to full_text not null)
"""

import time
import trafilatura
import pandas as pd
from tqdm import tqdm
from googlenewsdecoder import new_decoderv1

DIR     = "data/dxy_training/claude/"
PUB_CSV = DIR + "apr_jan_pub_ac.csv"
CLN_CSV = DIR + "apr_jan_cln.csv"


def extract_text(url):
    if not url or "news.google.com" in str(url):
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text if text else None
    except Exception:
        return None


def main():
    df = pd.read_csv(PUB_CSV)
    print(f"Loaded {len(df)} rows from {PUB_CSV}")

    google_mask = (
        df["actual_link"].str.contains("news.google.com", na=False)
        & df["full_text"].isna()
    )
    print(f"Rows with unresolved Google News URLs: {google_mask.sum()}")

    failed_idx = df[google_mask].index.tolist()
    resolved_links = []
    resolved_texts = []

    for idx in tqdm(failed_idx, desc="Decoding + extracting"):
        url = df.at[idx, "actual_link"]
        # Step 1: decode Google News URL
        try:
            result = new_decoderv1(url, interval=1)
            if result.get("status"):
                actual = result["decoded_url"]
            else:
                actual = url
        except Exception:
            actual = url

        resolved_links.append(actual)

        # Step 2: extract text from resolved URL
        text = extract_text(actual)
        resolved_texts.append(text)

        time.sleep(0.3)  # light rate-limit

    df.loc[failed_idx, "actual_link"] = resolved_links
    df.loc[failed_idx, "full_text"]   = resolved_texts

    df.to_csv(PUB_CSV, index=False)
    filled = df["full_text"].notna().sum()
    print(f"\nSaved {len(df)} rows to {PUB_CSV}  (full_text: {filled}/{len(df)})")

    cln = df[df["full_text"].notna()].copy().reset_index(drop=True)
    cln.to_csv(CLN_CSV, index=False)
    print(f"Saved {len(cln)} rows to {CLN_CSV}")


if __name__ == "__main__":
    main()
