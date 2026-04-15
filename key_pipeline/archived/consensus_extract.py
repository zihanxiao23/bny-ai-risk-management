# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import time
# import os
# from dotenv import load_dotenv

# load_dotenv()
# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# # ==============================================================================
# # EVENT MAPPING
# # Maps Finnhub event names to your DXY taxonomy event numbers.
# # Add/adjust strings based on what Finnhub actually returns for your date range.
# # ==============================================================================

# FINNHUB_EVENT_MAP = {
#     # CPI / Inflation
#     "CPI m/m":                          7,
#     "Core CPI m/m":                     7,
#     "CPI y/y":                          7,
#     "Core CPI y/y":                     7,
#     "PCE Price Index m/m":              7,
#     "Core PCE Price Index m/m":         7,

#     # NFP / Employment
#     "Non-Farm Employment Change":       10,   # beat/miss determined post-fetch
#     "Nonfarm Payrolls":                 10,
#     "Unemployment Rate":                12,

#     # GDP
#     "GDP q/q":                          13,   # beat/miss determined post-fetch
#     "Prelim GDP q/q":                   13,

#     # Retail Sales
#     "Retail Sales m/m":                 15,   # beat/miss determined post-fetch
#     "Core Retail Sales m/m":            15,

#     # PMI / ISM
#     "ISM Manufacturing PMI":            17,   # beat/miss determined post-fetch
#     "ISM Services PMI":                 17,
#     "S&P Global Manufacturing PMI":     17,
#     "S&P Global Services PMI":          17,

#     # Trade Balance
#     "Trade Balance":                    20,
# }

# # Events where beat = bullish DXY, miss = bearish DXY
# BEAT_IS_BULLISH = {7, 10, 12, 13, 15, 17, 20}


# def fetch_economic_calendar(from_date: str, to_date: str) -> list[dict]:
#     """
#     Fetch economic calendar events from Finnhub for a date range.
    
#     Args:
#         from_date: "YYYY-MM-DD"
#         to_date:   "YYYY-MM-DD"
    
#     Returns:
#         List of raw event dicts from Finnhub.
#     """
#     url = "https://finnhub.io/api/v1/calendar/economic"
#     params = {
#         "from":  from_date,
#         "to":    to_date,
#         "token": FINNHUB_API_KEY,
#     }
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     data = response.json()
#     return data.get("economicCalendar", [])


# def compute_surprise(actual, estimate, event_number: int) -> dict:
#     """
#     Compute surprise direction and magnitude given actual vs consensus.
    
#     Returns dict with:
#         surprise_direction: "beat" | "miss" | "inline" | "unknown"
#         surprise_magnitude: float (actual - estimate), None if not computable
#         beat_is_bullish_dxy: bool
#         dxy_direction_implied: "up" | "down" | "neutral" | "unknown"
#     """
#     if actual is None or estimate is None:
#         return {
#             "surprise_direction":    "unknown",
#             "surprise_magnitude":    None,
#             "beat_is_bullish_dxy":   event_number in BEAT_IS_BULLISH,
#             "dxy_direction_implied": "unknown",
#         }

#     try:
#         actual_f   = float(actual)
#         estimate_f = float(estimate)
#     except (ValueError, TypeError):
#         return {
#             "surprise_direction":    "unknown",
#             "surprise_magnitude":    None,
#             "beat_is_bullish_dxy":   event_number in BEAT_IS_BULLISH,
#             "dxy_direction_implied": "unknown",
#         }

#     diff = actual_f - estimate_f
#     threshold = 0.01   # treat < 0.01 deviation as inline

#     if abs(diff) < threshold:
#         direction = "inline"
#     elif diff > 0:
#         direction = "beat"
#     else:
#         direction = "miss"

#     bullish = event_number in BEAT_IS_BULLISH
#     if direction == "inline":
#         dxy_implied = "neutral"
#     elif direction == "beat":
#         dxy_implied = "up" if bullish else "down"
#     else:
#         dxy_implied = "down" if bullish else "up"

#     return {
#         "surprise_direction":    direction,
#         "surprise_magnitude":    round(diff, 4),
#         "beat_is_bullish_dxy":   bullish,
#         "dxy_direction_implied": dxy_implied,
#     }


# def build_consensus_table(from_date: str, to_date: str) -> pd.DataFrame:
#     """
#     Pull economic calendar from Finnhub, filter to DXY-relevant USD events,
#     compute surprise metrics, and return as a clean DataFrame.

#     This is your lookup table — join it to your articles by event_number + date.
#     """
#     print(f"Fetching Finnhub economic calendar: {from_date} → {to_date}")
#     raw_events = fetch_economic_calendar(from_date, to_date)
#     print(f"  {len(raw_events)} total events returned")

#     rows = []
#     for ev in raw_events:
#         # Filter to USD only
#         if ev.get("country", "").upper() != "US":
#             continue

#         event_name = ev.get("event", "")
#         event_number = FINNHUB_EVENT_MAP.get(event_name)

#         # Skip events not in your taxonomy
#         if event_number is None:
#             continue

#         actual   = ev.get("actual")
#         estimate = ev.get("estimate")
#         prev     = ev.get("prev")
#         date_str = ev.get("time", "")[:10]   # trim to YYYY-MM-DD

#         surprise = compute_surprise(actual, estimate, event_number)

#         # Resolve beat/miss variant of event number
#         # e.g. event 10 (NFP beat) vs 11 (NFP miss)
#         resolved_event_number = resolve_beat_miss(event_number, surprise["surprise_direction"])

#         rows.append({
#             "release_date":          date_str,
#             "event_name_finnhub":    event_name,
#             "event_number":          resolved_event_number,
#             "actual":                actual,
#             "estimate":              estimate,
#             "previous":              prev,
#             "surprise_direction":    surprise["surprise_direction"],
#             "surprise_magnitude":    surprise["surprise_magnitude"],
#             "dxy_direction_implied": surprise["dxy_direction_implied"],
#         })

#     df = pd.DataFrame(rows)
#     print(f"  {len(df)} DXY-relevant USD events after filtering")
#     return df


# def resolve_beat_miss(event_number: int, surprise_direction: str) -> int:
#     """
#     Some event pairs in your taxonomy split beat/miss into separate numbers.
#     Resolve the correct number based on surprise direction.
#     """
#     BEAT_MISS_MAP = {
#         # base_number: (beat_number, miss_number)
#         7:  (7, 8),    # CPI above / below consensus
#         10: (10, 11),  # NFP beat / miss
#         13: (13, 14),  # GDP beat / miss
#         15: (15, 16),  # Retail Sales beat / miss
#         17: (17, 18),  # PMI/ISM beat / miss
#     }

#     if event_number not in BEAT_MISS_MAP:
#         return event_number

#     beat_num, miss_num = BEAT_MISS_MAP[event_number]
#     if surprise_direction == "beat":
#         return beat_num
#     elif surprise_direction == "miss":
#         return miss_num
#     else:
#         return event_number   # inline or unknown — keep base


# def save_consensus_table(from_date: str, to_date: str, output_path: str):
#     """Pull, compute, and save the consensus table to CSV."""
#     df = build_consensus_table(from_date, to_date)
#     df.to_csv(output_path, index=False)
#     print(f"\nSaved {len(df)} rows → {output_path}")
#     print(df.to_string(index=False))
#     return df


# if __name__ == "__main__":
#     # Pull the last 12 months
#     to_date   = datetime.today().strftime("%Y-%m-%d")
#     from_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

#     save_consensus_table(
#         from_date=from_date,
#         to_date=to_date,
#         output_path="data/consensus_table.csv",
#     )
# '''

# ---

# **What this produces and how it connects to your existing pipeline:**

# The output CSV has one row per release event with these key fields:

# release_date | event_name_finnhub | event_number | actual | estimate | surprise_direction | surprise_magnitude | dxy_direction_implied
# 2025-01-10   | Non-Farm Employment Change | 10 | 256K | 165K | beat | 91.0 | up
# 2025-01-15   | CPI m/m            | 7            | 0.4   | 0.3      | beat               | 0.1                | up
# '''

import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://finnhub.io/api/v1/calendar/economic"
params = {
    "from":  "2025-03-01",
    "to":    "2025-03-07",
    "token": os.getenv("FINNHUB_API_KEY"),
}

response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
print(response.json())