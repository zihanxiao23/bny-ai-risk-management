import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RAW_CSV_PATH    = "data/forex_factory_cache.csv"
CLEAN_CSV_PATH  = "data/consensus_table.csv"
DXY_INTRADAY    = "data/dxy_intraday_min.csv"

DXY_HORIZONS = {
    "dxy_1h": pd.Timedelta(hours=1),
    "dxy_2h": pd.Timedelta(hours=2),
    "dxy_4h": pd.Timedelta(hours=4),
    "dxy_1d": pd.Timedelta(days=1),
}

# Download URL — run once, cache locally
HF_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar"
    "/resolve/main/forex_factory_cache.csv"
)

# ==============================================================================
# EVENT NAME MAP
# Maps Forex Factory event names to your DXY taxonomy event numbers.
# Actual vs forecast beat/miss resolved later — base numbers used here.
# ==============================================================================

FF_EVENT_MAP = {
    # CPI / Inflation  →  7 (beat) / 8 (miss) resolved post-parse
    "CPI m/m":                      7,
    "Core CPI m/m":                 7,
    "CPI y/y":                      7,
    "Core CPI y/y":                 7,
    "PCE Price Index m/m":          7,
    "Core PCE Price Index m/m":     7,
    "Core PCE Price Index y/y":     7,

    # NFP / Employment  →  10 (beat) / 11 (miss) resolved post-parse
    "Non-Farm Employment Change":   10,
    "Unemployment Rate":            12,

    # GDP  →  13 (beat) / 14 (miss) resolved post-parse
    "Prelim GDP q/q":               13,
    "GDP q/q":                      13,
    "Final GDP q/q":                13,

    # Retail Sales  →  15 (beat) / 16 (miss) resolved post-parse
    "Retail Sales m/m":             15,
    "Core Retail Sales m/m":        15,

    # PMI / ISM  →  17 (beat) / 18 (miss) resolved post-parse
    "ISM Manufacturing PMI":        17,
    "ISM Services PMI":             17,
    "Flash Manufacturing PMI":      17,
    "Flash Services PMI":           17,

    # Trade Balance  →  20
    "Trade Balance":                20,
}

# Beat/miss resolution pairs — (beat_number, miss_number)
BEAT_MISS_PAIRS = {
    7:  (7,  8),
    10: (10, 11),
    13: (13, 14),
    15: (15, 16),
    17: (17, 18),
}

# For these events, a higher actual = bullish DXY
HIGHER_IS_BULLISH = {7, 10, 12, 13, 15, 17}
# For trade balance, a less-negative number = bullish
HIGHER_IS_BULLISH.add(20)

# ==============================================================================
# STEP 1 — DOWNLOAD (once, then cached)
# ==============================================================================

def download_if_missing(raw_path: str, url: str):
    path = Path(raw_path)
    if path.exists():
        print(f"✅ Using cached file: {raw_path}")
        return
    print(f"⬇️  Downloading from Hugging Face (~68MB)...")
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Saved to {raw_path}")


# ==============================================================================
# STEP 2 — PARSE NUMERIC STRINGS
# Forex Factory stores values like "256K", "3.4%", "-$12.3B", "1.235T"
# This strips units and converts to float for arithmetic.
# ==============================================================================

def parse_ff_number(raw: str) -> float | None:
    """
    Convert Forex Factory value strings to floats.

    Examples:
        "256K"    →  256000.0
        "3.4%"    →  3.4
        "-12.3B"  →  -12300000000.0
        "1.235T"  →  1235000000000.0
        "54.5"    →  54.5
        null/""   →  None
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Strip currency symbols and spaces
    s = re.sub(r"[$€£¥\s]", "", s)

    # Extract sign
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")

    # Strip % — treat as raw number (3.4% → 3.4, not 0.034)
    s = s.replace("%", "")

    # Multiplier suffixes
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    multiplier = 1.0
    if s and s[-1].upper() in multipliers:
        multiplier = multipliers[s[-1].upper()]
        s = s[:-1]

    try:
        return sign * float(s) * multiplier
    except ValueError:
        return None


# ==============================================================================
# STEP 3 — COMPUTE SURPRISE
# ==============================================================================

def compute_surprise(actual_f: float | None,
                     forecast_f: float | None,
                     base_event_number: int) -> dict:
    """
    Returns surprise direction, magnitude, and implied DXY direction.
    Threshold is intentionally small — caller can filter by magnitude.
    """
    if actual_f is None or forecast_f is None:
        return {
            "surprise_direction":     "unknown",
            "surprise_magnitude":     None,
            "dxy_direction_expected": "unknown",
        }

    diff      = actual_f - forecast_f
    threshold = abs(forecast_f) * 0.005   # 0.5% of forecast = inline

    if abs(diff) <= threshold:
        direction = "inline"
    elif diff > 0:
        direction = "beat"
    else:
        direction = "miss"

    bullish = base_event_number in HIGHER_IS_BULLISH
    if direction == "inline":
        dxy = "neutral"
    elif direction == "beat":
        dxy = "up" if bullish else "down"
    else:
        dxy = "down" if bullish else "up"

    return {
        "surprise_direction":     direction,
        "surprise_magnitude":     round(diff, 6),
        "dxy_direction_expected": dxy,
    }


def resolve_event_number(base: int, surprise_direction: str) -> int:
    """Resolve beat/miss into the correct taxonomy event number."""
    if base not in BEAT_MISS_PAIRS:
        return base
    beat_n, miss_n = BEAT_MISS_PAIRS[base]
    if surprise_direction == "beat":
        return beat_n
    elif surprise_direction == "miss":
        return miss_n
    return base   # inline or unknown — keep base


# ==============================================================================
# STEP 4 — BUILD CONSENSUS TABLE
# ==============================================================================

def build_consensus_table(raw_path: str,
                           backtest_start: str = "2024-03-01",
                           backtest_end:   str = "2025-03-22") -> pd.DataFrame:
    """
    Load raw Forex Factory CSV, filter to USD high-impact events in your
    backtest window, compute surprise metrics, return clean DataFrame.
    """
    print(f"📂 Loading {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"   {len(df):,} total rows loaded")

    # --- Parse and convert DateTime to UTC ---
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df["release_date_utc"] = df["DateTime"].dt.tz_convert("UTC")
    df["release_date"] = df["release_date_utc"].dt.strftime("%Y-%m-%d")

    # --- Filter to backtest window ---
    start = pd.Timestamp(backtest_start, tz="UTC")
    end   = pd.Timestamp(backtest_end,   tz="UTC")
    df = df[(df["release_date_utc"] >= start) & (df["release_date_utc"] <= end)]
    print(f"   {len(df):,} rows in backtest window ({backtest_start} → {backtest_end})")

    # --- Filter to USD, high-impact only ---
    df = df[
        (df["Currency"] == "USD") &
        (df["Impact"].str.contains("High", na=False))
    ]
    print(f"   {len(df):,} USD high-impact rows")

    # --- Filter to events in taxonomy ---
    df = df[df["Event"].isin(FF_EVENT_MAP.keys())].copy()
    print(f"   {len(df):,} rows matching DXY taxonomy events")

    # --- Parse numeric values ---
    df["actual_f"]   = df["Actual"].apply(parse_ff_number)
    df["forecast_f"] = df["Forecast"].apply(parse_ff_number)
    df["previous_f"] = df["Previous"].apply(parse_ff_number)

    # --- Map base event number ---
    df["base_event_number"] = df["Event"].map(FF_EVENT_MAP)

    # --- Compute surprise ---
    surprise_rows = df.apply(
        lambda r: compute_surprise(
            r["actual_f"], r["forecast_f"], r["base_event_number"]
        ),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df, surprise_rows], axis=1)

    # --- Resolve final event number (beat vs miss) ---
    df["event_number"] = df.apply(
        lambda r: resolve_event_number(
            r["base_event_number"], r["surprise_direction"]
        ),
        axis=1,
    )

    # --- Select and rename output columns ---
    out = df[[
        "release_date",
        "release_date_utc",
        "Event",
        "event_number",
        "Actual",
        "Forecast",
        "Previous",
        "actual_f",
        "forecast_f",
        "previous_f",
        "surprise_direction",
        "surprise_magnitude",
        "dxy_direction_expected",
    ]].rename(columns={
        "Event":    "event_name_ff",
        "Actual":   "actual_raw",
        "Forecast": "forecast_raw",
        "Previous": "previous_raw",
    }).sort_values("release_date_utc").reset_index(drop=True)

    return out


# ==============================================================================
# STEP 5 — JOIN TO YOUR ARTICLES
# ==============================================================================

def load_consensus_table(clean_path: str) -> pd.DataFrame:
    """Load the pre-built consensus table for lookups.
    Derives base_event_number if not present in the CSV (beat/miss → base)."""
    df = pd.read_csv(clean_path, parse_dates=["release_date_utc"])
    if "base_event_number" not in df.columns:
        _base = {8: 7, 11: 10, 14: 13, 16: 15, 18: 17}
        df["base_event_number"] = df["event_number"].apply(lambda x: _base.get(x, x))
    return df


def lookup_consensus(article_date: str,
                     event_number: int,
                     consensus_df: pd.DataFrame,
                     window_days: int = 1) -> dict | None:
    """
    Find the consensus record for a given article date and event number.

    Matches within a window (default ±1 day) to handle articles published
    slightly after the release timestamp.

    Returns the closest match by time, or None if no match found.
    """
    article_dt = pd.Timestamp(article_date, tz="UTC")
    window     = pd.Timedelta(days=window_days)

    # Base event numbers to match against (beat and miss map to same base)
    base_numbers = {
        8: 7, 11: 10, 14: 13, 16: 15, 18: 17
    }
    lookup_base = base_numbers.get(event_number, event_number)

    mask = (
        (consensus_df["base_event_number"] == lookup_base) &
        (consensus_df["release_date_utc"] >= article_dt - window) &
        (consensus_df["release_date_utc"] <= article_dt + window)
    )
    matches = consensus_df[mask]

    if matches.empty:
        return None

    # Return the closest match
    closest = matches.iloc[
        (matches["release_date_utc"] - article_dt).abs().argsort()[:1]
    ]
    return closest.iloc[0].to_dict()


def build_consensus_context(consensus_record: dict | None) -> str:
    """
    Format a consensus record into a string for injection into Agent 2's
    user prompt. Returns empty string if no record found.
    """
    if consensus_record is None:
        return ""

    lines = [
        "\n── CONSENSUS DATA (structured, do not re-derive) ──────────────",
        f"  Actual   : {consensus_record.get('actual_raw', 'N/A')}",
        f"  Forecast : {consensus_record.get('forecast_raw', 'N/A')}",
        f"  Previous : {consensus_record.get('previous_raw', 'N/A')}",
        f"  Surprise : {consensus_record.get('surprise_direction', 'unknown')} "
        f"(magnitude: {consensus_record.get('surprise_magnitude', 'N/A')})",
        f"  DXY expected direction: {consensus_record.get('dxy_direction_expected', 'unknown')}",
        "────────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)

def lookup_consensus_history(event_number: int,
                              consensus_df: pd.DataFrame,
                              article_date: str | None = None,
                              n_prior: int = 3) -> dict | None:
    """
    Look up consensus history for a given event_number.

    Two modes:
    - article_date=None  : return ALL historical releases for the event,
                           ordered oldest → newest, as a flat list.
    - article_date=str   : return the release closest to article_date
                           (within ±1 day) plus n_prior releases before it.

    Returns None if no matching events are found.
    """
    # Resolve base event number for lookups
    base_numbers = {8: 7, 11: 10, 14: 13, 16: 15, 18: 17}
    lookup_base  = base_numbers.get(event_number, event_number)

    same_event = consensus_df[
        consensus_df["base_event_number"] == lookup_base
    ].sort_values("release_date_utc")

    if same_event.empty:
        return None

    # ── All-history mode ─────────────────────────────────────────────────────
    if article_date is None:
        return {
            "current": None,
            "history": [row.to_dict() for _, row in same_event.iterrows()],
        }

    # ── Date-anchored mode ───────────────────────────────────────────────────
    article_dt = pd.Timestamp(article_date, tz="UTC")
    window     = pd.Timedelta(days=1)

    current_mask = (
        (same_event["release_date_utc"] >= article_dt - window) &
        (same_event["release_date_utc"] <= article_dt + window)
    )
    current_matches = same_event[current_mask]
    if current_matches.empty:
        return None

    current = current_matches.iloc[
        (current_matches["release_date_utc"] - article_dt)
        .abs().argsort()[:1]
    ].iloc[0]

    prior = same_event[
        same_event["release_date_utc"] < current["release_date_utc"]
    ].tail(n_prior)

    return {
        "current": current.to_dict(),
        "history": [row.to_dict() for _, row in prior.iterrows()],
    }


def build_consensus_context_with_history(record: dict | None,
                                          history_table: str = "") -> str:
    """
    Format current release + history into a prompt block for Agent 2.

    Optionally appends a pre-formatted DXY reaction table (from
    build_historical_response_table) after the consensus block.
    """
    lines = []

    if record is not None and record.get("current") is not None:
        current = record["current"]
        history = record["history"]   # oldest → newest

        lines.append("\n── CONSENSUS DATA (structured, do not re-derive) ──────────────")

        if history:
            lines.append("  Recent trend (oldest → newest):")
            for h in history:
                lines.append(
                    f"    {h['release_date']}  "
                    f"actual={h['actual_raw']:>8}  "
                    f"forecast={h['forecast_raw']:>8}  "
                    f"→ {h['surprise_direction']:>7}  "
                    f"dxy={h['dxy_direction_expected']}"
                )
            lines.append("")

        lines += [
            f"  Current release ({current['release_date']}):",
            f"    Actual   : {current['actual_raw']}",
            f"    Forecast : {current['forecast_raw']}",
            f"    Previous : {current['previous_raw']}",
            f"    Surprise : {current['surprise_direction']} "
            f"(magnitude: {current['surprise_magnitude']})",
            f"    DXY expected direction: {current['dxy_direction_expected']}",
            "────────────────────────────────────────────────────────────────",
        ]

    if history_table:
        lines.append("\n" + history_table)

    return "\n".join(lines) if lines else ""

# ==============================================================================
# STEP 6 — MERGE CONSENSUS EVENTS WITH DXY PRICES
# ==============================================================================

def _build_forward_returns(dxy: pd.DataFrame) -> pd.DataFrame:
    """Add dxy_Nh and pct_Nh forward-return columns to a DXY price frame."""
    time_idx = dxy.set_index("Time")["Open"]
    for col, delta in DXY_HORIZONS.items():
        dxy[col] = time_idx.reindex(time_idx.index + delta).values
    for fwd_col, pct_col in [
        ("dxy_1h", "pct_1h"), ("dxy_2h", "pct_2h"),
        ("dxy_4h", "pct_4h"), ("dxy_1d", "pct_1d"),
    ]:
        dxy[pct_col] = (dxy[fwd_col] - dxy["Open"]) / dxy["Open"] * 100
    return dxy


def merge_with_dxy_csv(consensus_df: pd.DataFrame,
                        dxy_path: str = DXY_INTRADAY) -> pd.DataFrame:
    """
    Merge consensus events with the minute-level DXY CSV (ET-localised).
    Falls back to minute-level floor for the join key.
    """
    print(f"📈 Loading DXY intraday CSV: {dxy_path}...")
    dxy = pd.read_csv(dxy_path)
    dxy["Time"] = (
        pd.to_datetime(dxy["Time"])
          .dt.tz_localize("America/New_York")
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )
    dxy = dxy.sort_values("Time").reset_index(drop=True)
    print(f"   {len(dxy):,} minute rows  ({dxy['Time'].min()} → {dxy['Time'].max()})")

    dxy = _build_forward_returns(dxy)

    join_key = (
        consensus_df["release_date_utc"]
        .dt.tz_localize(None)
        .dt.floor("min")
    )
    out = consensus_df.copy()
    out["_join_key"] = join_key

    dxy_cols = (["Time", "Open"] + list(DXY_HORIZONS.keys())
                + ["pct_1h", "pct_2h", "pct_4h", "pct_1d"])
    merged = out.merge(
        dxy[dxy_cols].rename(columns={"Open": "dxy_open"}),
        how="left",
        left_on="_join_key",
        right_on="Time",
    ).drop(columns=["_join_key", "Time"])

    matched = merged["pct_1h"].notna().sum()
    print(f"   Matched {matched}/{len(merged)} events to a DXY minute")
    return merged


# ==============================================================================
# MAIN — Build and save the consensus table
# ==============================================================================

if __name__ == "__main__":

    BACKTEST_START = "2020-01-01"
    BACKTEST_END   = "2026-03-24"

    # Step 1 — download Forex Factory cache once
    download_if_missing(RAW_CSV_PATH, HF_DOWNLOAD_URL)

    # Step 2 — build consensus table
    consensus_df = build_consensus_table(
        raw_path       = RAW_CSV_PATH,
        backtest_start = BACKTEST_START,
        backtest_end   = BACKTEST_END,
    )

    # Step 3 — merge with minute-level DXY from the local CSV
    if Path(DXY_INTRADAY).exists():
        consensus_df = merge_with_dxy_csv(consensus_df, DXY_INTRADAY)
    else:
        print(f"   ⚠️  {DXY_INTRADAY} not found — skipping DXY merge")

    # Step 4 — save
    Path("data").mkdir(exist_ok=True)
    consensus_df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"\n💾 Saved {len(consensus_df)} rows → {CLEAN_CSV_PATH}")

    preview_cols = [
        "release_date", "event_name_ff", "event_number",
        "actual_raw", "forecast_raw", "surprise_direction",
        "dxy_direction_expected",
    ]
    if "pct_1h" in consensus_df.columns:
        preview_cols += ["pct_1h", "pct_2h", "pct_4h", "pct_1d"]
    print(f"\n📊 Preview:")
    print(consensus_df[preview_cols].to_string(index=False))