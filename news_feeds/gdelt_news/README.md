# GDELT News Feed Ingestion

This pipeline ingests weekly news from the GDELT 2.1 DOC API and appends new items to a CSV that matches the existing GNews schema. We switched from Yahoo Finance HTML scraping because it is unreliable in GitHub Actions due to anti-bot protections and intermittent 404s.

## What it does
- Fetches recent articles from the GDELT DOC API for each configured query.
- Normalizes article URLs and computes `id = sha256(normalized_link)`.
- Persists deduped IDs in a SQLite store shared across all feeds.
- Appends new rows to the CSV in a fixed schema.

## Output schema
The CSV header is fixed and must match exactly:
```
id,title,link,published,source,summary,query,fetched_at
```

## Dedupe mechanism
The SQLite database at `state/seen_ids.sqlite` stores IDs in the table:
```
seen_feed_ids(id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL)
```
Before appending to the CSV, the script checks this table and skips previously seen IDs. The `state/seen_ids.sqlite` file is created at runtime and intentionally not committed to git.

## Configuration (add queries here)
Update `config.yaml` to add new feeds without changing code:
```yaml
feeds:
  - name: bny
    query: '("BNY Mellon" OR "Bank of New York Mellon" OR BK) -sports -gossip'
```

## Run locally
From `news_feeds/gdelt_news/`:
```bash
python ingest.py \
  --config config.yaml \
  --out_csv data/gdelt_news.csv \
  --db_path state/seen_ids.sqlite \
  --max_records 250 \
  --days_back 7
```

Single-query run without config:
```bash
python ingest.py \
  --name bny \
  --query '("BNY Mellon" OR "Bank of New York Mellon" OR BK) -sports -gossip' \
  --out_csv data/gdelt_news.csv \
  --db_path state/seen_ids.sqlite \
  --max_records 250 \
  --days_back 7
```

Validate output:
```bash
python validate.py --csv data/gdelt_news.csv
```

## GitHub Actions
The workflow `.github/workflows/gdelt_weekly.yml` runs weekly (UTC) and via manual dispatch. It:
1. Installs requirements from `news_feeds/gdelt_news/requirements.txt`
2. Runs ingestion and validation
3. Commits and pushes if the CSV changed

## Logging
Logs are structured at INFO level by default. Use `--log_level DEBUG` for verbose output.
