import json
import re
from collections import Counter

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from tqdm import tqdm

NAIVE_TIMEZONE = "UTC"

URL_COL = "actual_link"

SELECTORS = [
    ("meta_article", By.CSS_SELECTOR, "meta[property='article:published_time']"),
    ("meta_pubdate", By.CSS_SELECTOR, "meta[name='pubdate']"),
    ("meta_date", By.CSS_SELECTOR, "meta[name='date']"),
    ("meta_itemprop", By.CSS_SELECTOR, "meta[itemprop='datePublished']"),
    ("time_tag", By.TAG_NAME, "time"),
    ("entry_date", By.CSS_SELECTOR, ".entry-date"),
]


def build_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)
    return driver


def to_utc(value, naive_tz=NAIVE_TIMEZONE):
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        ts = pd.to_datetime(value, errors="raise")
        if ts.tzinfo is None:
            ts = ts.tz_localize(naive_tz)
        return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def extract_json_ld(text):
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r'"(?:datePublished|dateCreated|uploadDate)"\s*:\s*"([^"]+)"', text)
        return m.group(1) if m else None

    stack = data if isinstance(data, list) else [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key in ("datePublished", "dateCreated", "uploadDate"):
                if item.get(key):
                    return item[key]
            for v in item.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(item, list):
            stack.extend(item)
    return None


def scrape_article_time(driver, url):
    if pd.isna(url) or not str(url).startswith(("http://", "https://")):
        return None, "Invalid URL"

    try:
        driver.get(url)

        for label, by, selector in SELECTORS:
            try:
                el = driver.find_element(by, selector)
                val = (
                    el.get_attribute("content")
                    or el.get_attribute("datetime")
                    or el.text
                    or ""
                ).strip()
                if len(val) > 5:
                    return val, label
            except Exception:
                pass

        for script in driver.find_elements(By.XPATH, "//script[@type='application/ld+json']"):
            val = extract_json_ld(script.get_attribute("innerHTML"))
            if val:
                return val, "json_ld"

        return None, "Not Found"

    except TimeoutException:
        return None, "Timeout"
    except WebDriverException:
        return None, "WebDriverError"
    except Exception as e:
        return None, type(e).__name__


def scrape_published_times(
    in_file_name,
    out_file_name=None,
    url_col=URL_COL,
    naive_timezone=NAIVE_TIMEZONE,
    checkpoint_every=25,
):
    df = pd.read_csv(in_file_name)

    driver = build_driver()
    rows = []
    via_counts = Counter()

    try:
        for i, url in enumerate(
            tqdm(df[url_col], total=len(df), desc="Scraping published times"),
            start=1,
        ):
            raw, via = scrape_article_time(driver, url)
            utc_val = to_utc(raw, naive_timezone)

            rows.append((raw, utc_val, via))
            via_counts[via] += 1

            if checkpoint_every and i % checkpoint_every == 0:
                success_count = sum(
                    v
                    for k, v in via_counts.items()
                    if k not in {"Not Found", "Timeout", "WebDriverError", "Invalid URL"}
                )
                tqdm.write(
                    f"[{i}/{len(df)}] success={success_count} "
                    f"not_found={via_counts['Not Found']} "
                    f"timeout={via_counts['Timeout']}"
                )
    finally:
        driver.quit()

    tmp = pd.DataFrame(
        rows,
        columns=["article_published_raw", "article_published_utc", "found_via"],
        index=df.index,
    )

    df = pd.concat([df, tmp], axis=1)

    if out_file_name:
        df.to_csv(f"{out_file_name}.csv", index=False)
        print(f"\nSaved: {out_file_name}.csv")

    print("\nFound-via summary:")
    for k, v in sorted(via_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    success_mask = df["article_published_utc"].notna()
    print(f"\nTotal rows: {len(df)}")
    print(f"Successful UTC timestamps: {int(success_mask.sum())}")
    print(f"Failed / missing: {int((~success_mask).sum())}")

    return df, via_counts


if __name__ == "__main__":
    df_out, summary = scrape_published_times(
        in_file_name="data/gt_200_fix_ft.csv",
        out_file_name="data/gt_200_fix_ft_pub"
    )