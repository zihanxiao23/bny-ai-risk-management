"""
Google News DXY/USD Prediction Scraper
Captures high-quality articles from reputable sources for predicting USD index changes.
"""
import os
import hashlib
import re
import time
import random
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import pandas as pd
from pygooglenews import GoogleNews

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Reputable sources WITHOUT paywalls (reusable prefix)
# SOURCES = '(source:reuters OR source:cnbc OR source:marketwatch OR source:bloomberg OR source:apnews OR source:forbes OR source:axios OR source:politico OR source:bbc OR source:cnn OR source:investing.com OR source:thehill)'
SOURCES = '(source:cnbc)'
# TIER 1: Direct USD/DXY drivers (highest signal)
TIER1_USD_CATALYSTS = [
    # Fed policy shocks
    f'{SOURCES} "Federal Reserve" AND (emergency OR surprise OR pivot)',
    f'{SOURCES} "FOMC" AND (surprise OR unexpected OR emergency)',
    f'{SOURCES} "Fed rate" AND (cut OR hike OR hold) AND surprise',
    # Merged Powell + Fed chair pressure into one — catches both dovish/hawkish signals and political interference
    f'{SOURCES} "Federal Reserve" AND (dovish OR hawkish OR independence OR "political pressure")',

    # Inflation shocks — consolidated 4 → 2
    f'{SOURCES} ("CPI" OR "PCE" OR "core inflation") AND "United States" AND (surprise OR miss OR beat OR unexpected)',
    f'{SOURCES} "inflation data" AND (shock OR surprise OR unexpected)',

    # Jobs data shocks — unchanged
    f'{SOURCES} "nonfarm payrolls" AND (miss OR beat OR surprise)',
    f'{SOURCES} "unemployment rate" AND (spike OR jump OR unexpected)',
    f'{SOURCES} "jobless claims" AND surge',
    f'{SOURCES} "jobs report" AND (disappointing OR strong OR weak)',

    # NEW: Other high-impact macro data releases (Consumer Confidence, Durable Goods, ISM, GDP, Retail Sales)
    f'{SOURCES} ("consumer confidence" OR "durable goods" OR "retail sales" OR "ISM" OR "GDP") AND "United States" AND (miss OR beat OR surprise OR unexpected)',
]

TIER2_TARIFF_TRADE_SHOCKS = [
    # Trump tariff events — unchanged
    f'{SOURCES} "Trump" AND "tariff" AND (announce OR pause)',
    f'{SOURCES} "reciprocal tariff" AND (China OR global OR pause)',
    f'{SOURCES} "trade war" AND "United States" AND escalate',
    f'{SOURCES} "tariff" AND (retaliation OR countermeasures)',

    # Dollar policy — merged explicit FX rhetoric into existing dollar queries
    # NEW: catches executive comments on dollar strength/weakness (Trump Iowa-style, Bessent statements)
    f'{SOURCES} ("Trump" OR "Bessent" OR "Treasury Secretary") AND dollar AND (weak OR strong OR policy OR intervention OR statement)',
    f'{SOURCES} "dollar index" AND (plunge OR surge OR crash OR rally)',
    f'{SOURCES} "DXY" AND (drop OR rally OR fall OR rise)',
    f'{SOURCES} "currency manipulation" AND dollar',

    # International responses — consolidated 2 → 1
    f'{SOURCES} "tariff" AND (China OR Europe OR Canada OR Mexico) AND (retaliate OR countermeasure OR response)',
    f'{SOURCES} "trade tensions" AND dollar',
]

TIER3_MACRO_DRIVERS = [
    # Treasury market stress — tightened bond market query to reduce noise
    f'{SOURCES} "Treasury yield" AND (spike OR plunge OR surge)',
    f'{SOURCES} "bond market" AND (selloff OR rally OR volatility) AND ("United States" OR Treasury OR dollar)',
    f'{SOURCES} "10-year yield" AND (jump OR fall OR soar)',
    f'{SOURCES} "bond vigilante"',

    # Growth/recession fears — unchanged
    f'{SOURCES} "recession" AND "United States" AND (warning OR risk OR probability)',
    f'{SOURCES} "GDP growth" AND "United States" AND (downgrade OR upgrade OR forecast)',
    f'{SOURCES} "economic outlook" AND "United States" AND (deteriorate OR improve)',

    # Fed independence — merged 3 → 2
    f'{SOURCES} "Fed independence" AND (threat OR concern OR Trump)',
    f'{SOURCES} "Powell" AND (fired OR dismissed OR pressure OR attacks)',

    # Dollar reserve status — unchanged
    f'{SOURCES} "dollar" AND "reserve currency" AND (threat OR decline OR shift)',
    f'{SOURCES} "de-dollarization"',
    f'{SOURCES} "dollar dominance" AND (waning OR declining)',

    # Safe haven flows — tightened to reduce generic gold/commodity articles
    f'{SOURCES} "safe haven" AND (dollar OR yen) AND (flows OR demand OR rally OR pressure)',
    f'{SOURCES} "risk off" AND dollar',
    f'{SOURCES} "flight to safety" AND currency',

    # Rate differentials — unchanged
    f'{SOURCES} "rate differential" AND dollar',
    f'{SOURCES} "yield advantage" AND dollar',

    # NEW: Other major central banks — ECB/BoJ/BoC decisions directly move DXY components
    f'{SOURCES} ("ECB" OR "Bank of Japan" OR "Bank of Canada") AND (rate OR decision OR hike OR cut) AND (dollar OR outlook)',
]

# Combine queries with priority weights
ALL_QUERIES = (
    [(q, 10) for q in TIER1_USD_CATALYSTS] +      # 13 queries, weight 10
    [(q, 8) for q in TIER2_TARIFF_TRADE_SHOCKS] + # 10 queries, weight 8
    [(q, 6) for q in TIER3_MACRO_DRIVERS]         # 18 queries, weight 6
)

# API configuration
LANG = "en"
COUNTRY = "US"
MAX_ARTICLES_PER_DAY = 40  # Target 40 articles (range will be 25-50)
MAX_ARTICLES_PER_QUERY = 10  # Don't pull more than 10 from any single query

# Rate limiting
MIN_DELAY = 1.0  # Minimum seconds between queries
MAX_DELAY = 2.0  # Maximum seconds between queries
ERROR_DELAY = 5  # Seconds to wait after error
DAY_DELAY_MIN = 5  # Minimum seconds between days
DAY_DELAY_MAX = 10  # Maximum seconds between days

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_url(url: str) -> str:
    """Remove common tracking params to get canonical URL."""
    try:
        p = urlparse(url)
        if not p.scheme:
            return url
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not re.match(r'^(utm_|gclid|fbclid|mc_|ref)', k, re.I)]
        return urlunparse((p.scheme, p.netloc, p.path, "", urlencode(q), ""))
    except Exception:
        return url


def make_id(title: str, canonical_link: str) -> str:
    """Generate unique ID for article based on title and link."""
    base = f"{(title or '').strip()}|{(canonical_link or '').strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def write_daily_file(rows: list, date_str: str) -> None:
    """Write articles for a specific day to CSV."""
    if not rows:
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{date_str}.csv')
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(rows)} articles to {output_path}")


# ============================================================================
# MAIN SCRAPING FUNCTION
# ============================================================================

def run_pull_for_date(target_date: datetime, max_articles: int = 40) -> pd.DataFrame:
    """
    Pull news for a specific date with priority-based query system.
    
    Args:
        target_date: The date to pull news for (articles from this day predict DXY on next day)
        max_articles: Maximum number of articles to collect
    
    Returns:
        DataFrame of collected articles
    """
    gn = GoogleNews(lang=LANG, country=COUNTRY)
    run_iso = datetime.now().isoformat()
    
    from_date = target_date.strftime('%Y-%m-%d')
    to_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    prediction_date = to_date
    
    print(f"\n{'='*70}")
    print(f"Fetching articles for {from_date}")
    print(f"(To predict DXY change on {prediction_date})")
    print(f"Target: {max_articles} articles | Max queries: {len(ALL_QUERIES)}")
    print(f"{'='*70}")
    
    collected_articles = []
    seen_ids = set()  # Track duplicates within this day only
    queries_executed = 0
    
    # Process queries in priority order
    for query_idx, (query, priority_weight) in enumerate(ALL_QUERIES, 1):
        # Stop if we've collected enough articles
        if len(collected_articles) >= max_articles:
            print(f"\n✓ Reached target of {max_articles} articles. Stopping.")
            break
        
        # Show abbreviated query (remove source prefix for readability)
        query_display = query.replace(SOURCES, '[SOURCES]')
        print(f"[{query_idx}/{len(ALL_QUERIES)}] P={priority_weight} | {query_display[:65]}...")
        
        try:
            # Fetch articles for this query
            res = gn.search(query, from_=from_date, to_=to_date)
            entries = res.get("entries", [])
            queries_executed += 1
            
            articles_added = 0
            
            for e in entries[:MAX_ARTICLES_PER_QUERY]:
                # Stop if we've hit the daily limit
                if len(collected_articles) >= max_articles:
                    break
                
                title = e.get("title", "")
                link = e.get("link", "")
                src = (e.get("source", {}) or {}).get("title", "")
                published = e.get("published", "")
                
                canonical = clean_url(link)
                _id = make_id(title, canonical)
                
                # Skip duplicates within this day
                if _id in seen_ids:
                    continue
                
                # Create article record
                row = {
                    "id": _id,
                    "title": title,
                    "link": canonical,
                    "published": published,
                    "source": src,
                    "query": query_display,  # Store abbreviated version
                    "priority_weight": priority_weight,
                    "article_date": from_date,
                    "prediction_date": prediction_date,
                    "fetched_at": run_iso
                }
                
                collected_articles.append(row)
                seen_ids.add(_id)
                articles_added += 1
            
            print(f"  → +{articles_added} | Total: {len(collected_articles)}/{max_articles}")
            
            # Rate limiting
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            time.sleep(ERROR_DELAY)
            continue
    
    # Summary
    print(f"\n{'─'*70}")
    print(f"SUMMARY: {len(collected_articles)} articles from {queries_executed} queries")
    if collected_articles:
        source_counts = pd.DataFrame(collected_articles)['source'].value_counts()
        print(f"Top sources: {dict(source_counts.head(5))}")
    print(f"{'─'*70}")
    
    # Save results
    if collected_articles:
        write_daily_file(collected_articles, from_date)
    
    return pd.DataFrame(collected_articles)


def run_date_range(start_date: datetime, end_date: datetime, max_articles_per_day: int = 40):
    """
    Collect articles for a range of dates.
    
    Args:
        start_date: First date to collect
        end_date: Last date to collect
        max_articles_per_day: Target articles per day
    """
    current_date = start_date
    total_articles = 0
    total_queries = 0
    days_processed = 0
    start_time = time.time()
    
    total_days = (end_date - start_date).days + 1
    
    print("\n" + "="*70)
    print("DXY/USD INDEX PREDICTION DATASET COLLECTION")
    print("="*70)
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total days: {total_days}")
    print(f"Target: {max_articles_per_day} articles per day")
    print(f"Query pool: {len(ALL_QUERIES)} queries (Tier 1-3)")
    # print(f"Sources: Reuters, CNBC, MarketWatch, Bloomberg, AP, Forbes, Axios, Politico, etc.")
    print("Sources: AP NEWS")
    print("="*70 + "\n")
    
    while current_date <= end_date:
        try:
            # Collect articles for this date
            df = run_pull_for_date(current_date, max_articles=max_articles_per_day)
            
            # Track stats
            total_articles += len(df)
            queries_used = df['query'].nunique() if len(df) > 0 else 0
            total_queries += queries_used
            days_processed += 1
            
            # Progress update every 10 days
            if days_processed % 10 == 0:
                elapsed_hours = (time.time() - start_time) / 3600
                days_remaining = (end_date - current_date).days
                queries_per_hour = total_queries / elapsed_hours if elapsed_hours > 0 else 0
                
                print(f"\n{'═'*70}")
                print(f"📊 PROGRESS UPDATE")
                print(f"{'═'*70}")
                print(f"Days: {days_processed}/{total_days} | Remaining: {days_remaining}")
                print(f"Articles: {total_articles} total ({total_articles/days_processed:.1f} avg/day)")
                print(f"Queries: {total_queries} total ({total_queries/days_processed:.1f} avg/day)")
                print(f"Rate: {queries_per_hour:.1f} queries/hour")
                print(f"Elapsed: {elapsed_hours:.2f}h | Est. remaining: {elapsed_hours/days_processed*days_remaining:.2f}h")
                print(f"{'═'*70}\n")
            
            # Short pause between days
            time.sleep(random.uniform(DAY_DELAY_MIN, DAY_DELAY_MAX))
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            print(f"Completed: {days_processed}/{total_days} days")
            print(f"To resume, set START_DATE = datetime({current_date.year}, {current_date.month}, {current_date.day})")
            break
        except Exception as e:
            print(f"\n✗ Error processing {current_date.strftime('%Y-%m-%d')}: {e}")
            print(f"Continuing to next date...")
        
        current_date += timedelta(days=1)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("🎉 COLLECTION COMPLETE")
    print("="*70)
    print(f"Days processed: {days_processed}")
    print(f"Total articles: {total_articles}")
    print(f"Total queries: {total_queries}")
    print(f"Average: {total_articles/days_processed:.1f} articles/day, {total_queries/days_processed:.1f} queries/day")
    print(f"Runtime: {total_time/3600:.2f} hours")
    print(f"Query rate: {total_queries/(total_time/3600):.1f} queries/hour")
    print(f"\n📁 Output: {days_processed} CSV files in {OUTPUT_DIR}/")
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set your date range here
    # Example: Capture the 2023-2025 period with major DXY events
    START_DATE = datetime(2026, 2, 16)
    END_DATE = datetime(2026, 2, 28)
    ARTICLES_PER_DAY = 40

    # Output directory
    OUTPUT_DIR = 'data/cnbc/'
    
    print("\n🚀 Starting DXY/USD dataset collection...")
    print(f"📅 Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"🎯 Target: {ARTICLES_PER_DAY} articles per day")
    print(f"📰 Sources: Reputable outlets (no paywalls)")
    print(f"⚡ Rate limit: 1-2 second delays\n")
    
    # Run collection
    run_date_range(START_DATE, END_DATE, max_articles_per_day=ARTICLES_PER_DAY)
    