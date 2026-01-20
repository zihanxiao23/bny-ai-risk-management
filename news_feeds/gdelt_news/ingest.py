#!/usr/bin/env python3
import argparse
import csv
import hashlib
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional
from urllib.parse import urlparse, urlunparse

import requests
import yaml
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
SCHEMA = [
    "id",
    "title",
    "link",
    "published",
    "source",
    "summary",
    "query",
    "fetched_at",
]


@dataclass
class FeedConfig:
    name: str
    query: str


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_config(path: str) -> List[FeedConfig]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    feeds = []
    for entry in data.get("feeds", []):
        if not entry.get("query"):
            raise ValueError("Each feed must include query")
        feeds.append(
            FeedConfig(
                name=entry.get("name", "feed"),
                query=entry["query"],
            )
        )
    return feeds


def should_retry(exception: Exception) -> bool:
    if isinstance(exception, requests.HTTPError) and exception.response is not None:
        return exception.response.status_code in {429} or exception.response.status_code >= 500
    if isinstance(exception, requests.RequestException):
        return exception.response is None
    return False


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(should_retry),
)
def fetch_json(params: dict, session: requests.Session) -> Optional[dict]:
    logging.debug("Fetching GDELT API with params: %s", params)
    response = session.get(
        API_URL,
        params=params,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()
    if 400 <= response.status_code < 500:
        logging.error("Received %s from GDELT API", response.status_code)
        return None
    return response.json()


def normalize_link(link: str) -> str:
    parsed = urlparse(link)
    normalized = parsed._replace(query="", fragment="")
    return urlunparse(normalized)


def compute_id(normalized_link: str) -> str:
    return hashlib.sha256(normalized_link.encode("utf-8")).hexdigest()


def ensure_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_feed_ids ("
            "id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)"
        )
        conn.commit()


def insert_seen_id(conn: sqlite3.Connection, feed_id: str, first_seen_at: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_feed_ids (id, first_seen_at) VALUES (?, ?)",
        (feed_id, first_seen_at),
    )


def ensure_csv(out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SCHEMA)


def append_rows(out_csv: str, rows: Iterable[List[str]]) -> None:
    with open(out_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def build_params(query: str, max_records: int, days_back: int) -> dict:
    start_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    return {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
    }


def parse_articles(payload: dict) -> List[dict]:
    if not payload:
        return []
    return payload.get("articles", []) or []


def ingest_feed(
    feed: FeedConfig,
    out_csv: str,
    db_path: str,
    max_records: int,
    days_back: int,
    session: requests.Session,
    fetched_at: str,
) -> int:
    params = build_params(feed.query, max_records, days_back)
    payload = fetch_json(params, session)
    if payload is None:
        logging.error("Failed to fetch GDELT results for %s", feed.name)
        return 0

    articles = parse_articles(payload)
    if not articles:
        logging.warning("No articles returned for %s", feed.name)

    ensure_db(db_path)
    ensure_csv(out_csv)

    new_rows = []
    new_count = 0
    skipped = 0

    with sqlite3.connect(db_path) as conn:
        for article in articles:
            link = article.get("url")
            title = article.get("title", "").strip()
            if not link or not title:
                continue
            normalized_link = normalize_link(link)
            feed_id = compute_id(normalized_link)
            existing = conn.execute(
                "SELECT 1 FROM seen_feed_ids WHERE id = ?", (feed_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue
            source = (
                article.get("domain")
                or article.get("sourcecountry")
                or article.get("sourcecollection")
                or "GDELT"
            )
            summary = article.get("snippet", "")
            published = article.get("seendate", "")
            insert_seen_id(conn, feed_id, fetched_at)
            new_rows.append(
                [
                    feed_id,
                    title,
                    normalized_link,
                    published,
                    source,
                    summary,
                    feed.query,
                    fetched_at,
                ]
            )
            new_count += 1
        conn.commit()

    if new_rows:
        append_rows(out_csv, new_rows)

    logging.info("Feed %s complete: %d new, %d skipped", feed.name, new_count, skipped)
    return new_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GDELT news ingestion")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--query", help="Run a single query")
    parser.add_argument("--name", default="feed", help="Name for single query")
    parser.add_argument(
        "--out_csv",
        default="data/gdelt_news.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--db_path",
        default="state/seen_ids.sqlite",
        help="SQLite DB path",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=250,
        help="Max records per feed",
    )
    parser.add_argument(
        "--days_back",
        type=int,
        default=7,
        help="How many days back to query",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.query:
        feeds = [FeedConfig(name=args.name, query=args.query)]
    else:
        feeds = load_config(args.config)
        if not feeds:
            logging.error("No feeds defined in %s", args.config)
            return 2

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()

    total_new = 0
    for feed in feeds:
        try:
            total_new += ingest_feed(
                feed,
                out_csv=args.out_csv,
                db_path=args.db_path,
                max_records=args.max_records,
                days_back=args.days_back,
                session=session,
                fetched_at=fetched_at,
            )
        except requests.RequestException as exc:
            logging.error("Network error for %s: %s", feed.name, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logging.exception("Unexpected error for %s: %s", feed.name, exc)
            continue

    logging.info("Ingestion complete. Total new items: %d", total_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
