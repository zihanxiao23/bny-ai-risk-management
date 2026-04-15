import os
import time
import anthropic
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv
from consensus_extract_hf import (
    load_consensus_table,
    lookup_consensus_history,
    build_consensus_context_with_history,
)

load_dotenv()

# ==============================================================================
# CLIENT SETUP
# Uses ANTHROPIC_API_KEY from .env file in the same directory as this script.
# Create a .env file with:  ANTHROPIC_API_KEY=your_key_here
# Get your key from: https://console.anthropic.com/
#
# Data privacy: Anthropic does NOT use API traffic to train models by default.
# Your article content and prompts are not used for training.
# ==============================================================================

def get_client():
    """Initialize Anthropic client using .env API key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found. Add it to your .env file.")
    return anthropic.Anthropic(api_key=api_key)


MODEL = "claude-haiku-4-5-20251001"

# Delay between articles (seconds). At 2s we use ~65% of the 450k/min input
# token rate limit, leaving headroom for large Pool B blocks or long articles.
# Increase to 3s if you see rate-limit errors mid-run.
INTER_ARTICLE_DELAY = 2


# ==============================================================================
# TIER LOOKUP TABLE
# Based on DXY driver hierarchy research:
#   Fed Board IFDP (2017, 2021, 2024), BIS WP 695, J.P. Morgan AM (2025),
#   ICE USDX White Paper, NBER WP 32329, CEPR/Eichengreen (2023)
#
# Special categories:
#   0  = Irrelevant — not a DXY-relevant event (no tier assigned)
#   28 = Other / Mixed — relevant to DXY but doesn't fit named categories
#
# Tier 1 — Primary Structural: Fed policy, rate differentials, core macro data
#           Largest magnitude, longest-lasting DXY impact
# Tier 2 — Secondary Structural: risk appetite, capital flows, fiscal policy
#           Medium-term, recurring influence
# Tier 3 — Geopolitical & Sentiment: trade/tariff, CB divergence, institutional risk
#           Event-driven, acute but shorter-lived volatility
# Tier 4 — Ambiguous: "Other / Mixed" catchall for relevant but uncategorised events
# ==============================================================================

IRRELEVANT_EVENT_NUMBER = 0
OTHER_EVENT_NUMBER      = 28

EVENT_TIERS = {
    # Tier 1: Fed monetary policy & core macro fundamentals
    1:  1,   # Fed Rate Hike
    2:  1,   # Fed Rate Cut
    3:  1,   # Hawkish Pivot / Surprise
    4:  1,   # Dovish Pivot / Surprise
    5:  1,   # Fed QE / Balance Sheet Expansion
    6:  1,   # Fed QT / Tapering Announcement
    7:  1,   # CPI/PCE Above Consensus
    8:  1,   # CPI/PCE Below Consensus
    9:  2,   # Inflation In-Line (priced in → Tier 2)
    10: 1,   # NFP/Jobs Beat
    11: 1,   # NFP/Jobs Miss
    12: 1,   # Unemployment Rate Surprise
    13: 1,   # GDP Beat
    14: 1,   # GDP Miss / Recession Signal
    31: 1,   # Fed Rate Path Repricing (market odds shifting)

    # Tier 2: Secondary structural drivers
    15: 2,   # Retail Sales Beat
    16: 2,   # Retail Sales Miss
    17: 2,   # PMI/ISM Beat
    18: 2,   # PMI/ISM Miss
    20: 2,   # Trade Balance Surprise
    21: 2,   # Debt Ceiling / Fiscal Crisis
    22: 2,   # Fiscal Stimulus / Spending Package
    23: 2,   # US Banking / Financial System Stress
    24: 2,   # Risk-Off Shock (safe-haven flows)
    25: 2,   # Treasury Yield Spike / Bond Market Stress
    32: 2,   # Fiscal Policy Probability Shift
    34: 2,   # Legislative Progress/Blockage

    # Tier 3: Geopolitical & sentiment
    19: 3,   # Trade Tariff / Policy Shock
    26: 3,   # US Political Shock
    27: 3,   # Foreign Central Bank Shock
    29: 3,   # Trade Policy Reversal Signal
    30: 3,   # Trade Policy Escalation Signal
    33: 3,   # Legal/Judicial Event Affecting Economic Policy
    35: 3,   # Executive Policy Signal
    36: 3,   # ECB Rate Change
    37: 3,   # BOJ Rate Change
    38: 3,   # BOE Rate Change
    39: 3,   # Other Major CB Rate Change

    # Tier 4: Relevant but uncategorised
    28: 4,   # Other / Mixed
}

TIER_LABELS = {
    1: "Tier 1 — Primary Structural (Fed/Core Macro)",
    2: "Tier 2 — Secondary Structural (Risk/Flows/Fiscal)",
    3: "Tier 3 — Geopolitical & Sentiment",
    4: "Tier 4 — Other / Uncategorised",
}

EVENT_NAMES = {
    0:  "Irrelevant",
    1:  "Fed Rate Hike",
    2:  "Fed Rate Cut",
    3:  "Hawkish Pivot / Surprise",
    4:  "Dovish Pivot / Surprise",
    5:  "Fed QE / Balance Sheet Expansion",
    6:  "Fed QT / Tapering Announcement",
    7:  "CPI/PCE Above Consensus",
    8:  "CPI/PCE Below Consensus",
    9:  "Inflation In-Line",
    10: "NFP/Jobs Beat",
    11: "NFP/Jobs Miss",
    12: "Unemployment Rate Surprise",
    13: "GDP Beat",
    14: "GDP Miss / Recession Signal",
    15: "Retail Sales Beat",
    16: "Retail Sales Miss",
    17: "PMI/ISM Beat",
    18: "PMI/ISM Miss",
    19: "Trade Tariff / Policy Shock",
    20: "Trade Balance Surprise",
    21: "Debt Ceiling / Fiscal Crisis",
    22: "Fiscal Stimulus / Spending Package",
    23: "US Banking / Financial System Stress",
    24: "Risk-Off Shock",
    25: "Treasury Yield Spike / Bond Market Stress",
    26: "US Political Shock",
    27: "Foreign Central Bank Shock",
    28: "Other / Mixed",
    29: "Trade Policy Reversal Signal",
    30: "Trade Policy Escalation Signal",
    31: "Fed Rate Path Repricing",
    32: "Fiscal Policy Probability Shift",
    33: "Legal/Judicial Event Affecting Economic Policy",
    34: "Legislative Progress or Blockage",
    35: "Executive Policy Signal",
    36: "ECB Rate Change",
    37: "BOJ Rate Change",
    38: "BOE Rate Change",
    39: "Other Major CB Rate Change",
}

# Content types that can never produce high criticality (hard floors)
NON_ACTIONABLE_CONTENT_TYPES = {"opinion", "analysis", "preview", "recap"}


# ==============================================================================
# GROUND TRUTH LOADING — two-pool few-shot architecture
#
# Pool A (Agent 1 — Classifier):
#   A balanced sample across event types, built ONCE at startup and injected
#   into the classifier system prompt. Stable string -> benefits from caching.
#   Purpose: calibrate event taxonomy + content_type classification.
#   Only rows with non-null true_criticality are eligible for sampling.
#
# Pool B (Agent 2 — Criticality/Direction):
#   All labeled GT rows whose event maps to the SAME TIER as the article.
#   Built PER-ARTICLE at Agent 2 call time from the full GT DataFrame.
#   Sorted by date descending so the most recent (regime-relevant) examples
#   appear first. Each example shows: headline, content preview, event,
#   content_type, true_criticality, true_direction (when set), pct_15m outcome,
#   AND the point-in-time macro regime snapshot at publication -- so the model
#   can learn "this macro context + this event type => high/not high".
#   Only rows with non-null true_criticality are included.
#
# Required GT CSV columns:
#   title, full_text, gt_content_type, gt_event_number
#
# Label columns (used when present, silently skipped when absent):
#   true_criticality  -- 'high' | 'not high'  (15-min DXY move threshold)
#   true_direction    -- 'up' | 'down' | 'neutral'  (15-min DXY direction)
#   pct_15m           -- realized 15-min DXY % move (float)
#   Open_15m          -- DXY index level at 15-min mark (float)
#
# Macro columns (shown in Pool B examples when non-null):
#   us_2yr_yield, fed_funds_rate, vix, dxy_zscore_52w_broad, etc.
#   All the same macro columns used by Agent 2's MACRO REGIME CONTEXT block.
#
# TIER CAP: Tier 1 may have many examples -- cap prevents prompt bloat.
# GT_TIER_CAPS controls max per tier; only labeled rows count toward the cap.
# ==============================================================================

# Maximum GT examples injected into Agent 2 per tier.
# Tier 1 default capped at 16; Tiers 2/3/4 use all available.
GT_TIER_CAPS = {1: 16, 2: 999, 3: 999, 4: 999}


def load_gt_dataframe(ground_truth_csv_path: str) -> "pd.DataFrame":
    """
    Load the ground truth CSV and enrich with derived columns used by both pools.

    Validates required columns, derives gt_event_name and gt_tier, and adds a
    gt_labeled boolean column marking rows eligible for few-shot use (i.e. those
    with non-null true_criticality). Unlabeled rows are kept in the DataFrame for
    reference but are excluded from Pool A sampling and Pool B injection.

    Returns the enriched DataFrame. Caller holds the reference for the
    full session -- do not reload per article.
    """
    print(f"Loading ground truth CSV: {ground_truth_csv_path}")
    gt_df = pd.read_csv(ground_truth_csv_path)

    required = {"title", "gt_content_type", "gt_event_number"}
    missing = required - set(gt_df.columns)
    if missing:
        raise ValueError(
            f"Ground truth CSV is missing required columns: {missing}\n"
            f"Found columns: {list(gt_df.columns)}"
        )

    gt_df["gt_event_name"] = gt_df["gt_event_number"].map(EVENT_NAMES)
    unknown = gt_df["gt_event_name"].isna()
    if unknown.any():
        bad = gt_df.loc[unknown, "gt_event_number"].unique().tolist()
        raise ValueError(f"Ground truth contains unrecognised gt_event_number values: {bad}")

    gt_df["gt_tier"] = gt_df["gt_event_number"].map(EVENT_TIERS)

    has_criticality = "true_criticality" in gt_df.columns
    has_direction   = "true_direction" in gt_df.columns
    has_pct_15m     = "pct_15m" in gt_df.columns

    # gt_labeled: True only for rows usable as few-shot examples.
    # Requires at minimum a non-null true_criticality label.
    if has_criticality:
        gt_df["gt_labeled"] = gt_df["true_criticality"].notna()
    else:
        gt_df["gt_labeled"] = False

    label_status = []
    if has_criticality:
        labeled = gt_df[gt_df["gt_labeled"]]
        n_high     = (labeled["true_criticality"] == "high").sum()
        n_not_high = (labeled["true_criticality"] == "not high").sum()
        n_skip     = (~gt_df["gt_labeled"]).sum()
        label_status.append(
            f"true_criticality: {n_high} high / {n_not_high} not high "
            f"({n_skip} unlabeled rows skipped)"
        )
    else:
        label_status.append("true_criticality MISSING")

    if has_direction:
        n_dir = gt_df.loc[gt_df["gt_labeled"], "true_direction"].notna().sum()
        label_status.append(f"true_direction: {n_dir} labeled")
    else:
        label_status.append("true_direction MISSING")

    if has_pct_15m:
        n_pct = gt_df.loc[gt_df["gt_labeled"], "pct_15m"].notna().sum()
        label_status.append(f"pct_15m: {n_pct} rows")

    print(f"GT loaded: {len(gt_df)} rows total | " + " | ".join(label_status))
    return gt_df


def _fmt_pct(val) -> str:
    """Format a float percentage for display, or return N/A."""
    if val is None:
        return "N/A"
    try:
        import math
        f = float(val)
        return "N/A" if math.isnan(f) else f"{f:+.3f}%"
    except (TypeError, ValueError):
        return "N/A"


# ---- Pool A: Classifier few-shot (built once at startup) --------------------

def build_classifier_few_shot_block(gt_df: "pd.DataFrame", max_examples: int = 12) -> str:
    """
    Pool A -- build a stable few-shot string for the Agent 1 system prompt.

    Only uses rows with non-null true_criticality (gt_labeled == True).
    Samples up to 2 examples per event type for taxonomy coverage, capped at
    max_examples total. Shuffled so no event type dominates the top of the prompt.

    CACHING: the returned string is injected once into the system prompt and
    cached server-side. Never regenerate it per article.
    """
    if gt_df is None or gt_df.empty:
        return ""

    # Only sample from fully-labeled rows
    pool = gt_df[gt_df.get("gt_labeled", pd.Series(True, index=gt_df.index))].copy()
    if pool.empty:
        return ""

    sampled = (
        pool.groupby("gt_event_number", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 2), random_state=42))
        .sample(frac=1, random_state=42)
        .head(max_examples)
        .reset_index(drop=True)
    )

    content_col = (
        "full_text" if "full_text" in gt_df.columns
        else "content" if "content" in gt_df.columns
        else None
    )
    has_criticality = "true_criticality" in gt_df.columns
    has_direction   = "true_direction" in gt_df.columns
    _MOVE_COLS      = ["pct_15m", "pct_1h", "pct_4h", "pct_1d"]

    lines = [
        "================================================================",
        "FEW-SHOT EXAMPLES -- real articles labeled by a human analyst",
        "Use these to calibrate your classification judgment and output format.",
        "true_criticality = 15-min DXY reaction: 'high' = meaningful market",
        "move; 'not high' = limited or no reaction within 15 minutes.",
        "true_direction = DXY movement direction in 15-min window.",
        "================================================================\n",
    ]

    for i, (_, row) in enumerate(sampled.iterrows(), 1):
        content_preview = (
            str(row.get(content_col, ""))[:500].strip() if content_col else ""
        )
        lines.append(f"-- Example {i} ------------------------------------------")
        lines.append(f"Title   : {row['title']}")
        if content_preview:
            lines.append(f"Content : {content_preview}...")
        lines.append("LABELS:")
        lines.append(f"  content_type -> {row['gt_content_type']}")
        lines.append(f"  event_name   -> {row['gt_event_name']}")
        if has_criticality and pd.notna(row.get("true_criticality")):
            lines.append(f"  criticality  -> {row['true_criticality']}")
        if has_direction and pd.notna(row.get("true_direction")):
            lines.append(f"  direction    -> {row['true_direction']}")
        moves = {col: _fmt_pct(row.get(col)) for col in _MOVE_COLS if col in gt_df.columns}
        if any(v != "N/A" for v in moves.values()):
            lines.append("ACTUAL DXY MOVES (15m and longer horizons, where measured):")
            for col in _MOVE_COLS:
                if col in moves and moves[col] != "N/A":
                    lines.append(f"  {col} -> {moves[col]}")
        lines.append("")

    lines.append("================================================================")
    lines.append("Now classify the new article below using the same judgment.\n")
    return "\n".join(lines)


# ── Macro columns shown per Pool B example ───────────────────────────────────
# cftc_net_usd_zscore intentionally excluded — zero-variance in current data.
_POOL_B_MACRO_FIELDS = [
    ("fed_funds_rate",        "Fed Funds Rate",      "%"),
    ("us_2yr_yield",          "US 2yr yield",        "%"),
    ("us_10yr_yield",         "US 10yr yield",       "%"),
    ("us_de_2yr_spread",      "US-DE 2yr spread",    "%"),
    ("yield_curve_2s10s",     "Yield curve 2s10s",   "%"),
    ("vix",                   "VIX",                 ""),
    ("vix_regime",            "VIX regime",          ""),
    ("dxy_zscore_52w_broad",  "DXY 52w z-score",    ""),
    ("dxy_20d_return_broad",  "DXY 20d return",      "%"),
    ("cpi_yoy",               "CPI YoY",             "%"),
    ("core_cpi_yoy",          "Core CPI YoY",        "%"),
    ("unemployment_rate",     "Unemployment",        "%"),
    ("sahm_indicator",        "Sahm indicator",      ""),
]


def _fmt_macro_val(val, suffix: str) -> str:
    """Format a macro value for Pool B display, or return None if missing."""
    import math
    if val is None:
        return None
    if isinstance(val, str):
        return val.strip() or None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        return f"{f:.2f}{suffix}"
    except (TypeError, ValueError):
        return None


# ---- Pool B: Criticality/Direction few-shot (built per-article) -------------

def build_tier_few_shot_block(gt_df: "pd.DataFrame", tier: int) -> str:
    """
    Pool B -- build a few-shot block for Agent 2 from all labeled GT rows in
    the same tier as the article being assessed.

    Rows are sorted by date descending (most recent first) so the model sees
    the most regime-relevant examples first. This reduces the impact of macro
    regime mismatch between old GT examples and the current evaluation period.

    Each example shows:
      - Article title + content preview
      - Event type and content_type
      - true_criticality outcome (high / not high)
      - true_direction and pct_15m when available
      - Point-in-time macro snapshot at publication date

    Only rows where true_criticality is non-null are included.
    Cap applied via GT_TIER_CAPS to prevent prompt bloat for large tiers.
    Returns empty string if gt_df is None/empty, no labeled rows for tier,
    or true_criticality column is absent.
    """
    if gt_df is None or gt_df.empty:
        return ""

    has_criticality = "true_criticality" in gt_df.columns
    has_direction   = "true_direction" in gt_df.columns
    has_pct_15m     = "pct_15m" in gt_df.columns
    if not has_criticality:
        return ""   # need at least criticality labels to be useful

    # Filter to labeled rows for this tier only
    labeled_mask = gt_df.get("gt_labeled", pd.Series(True, index=gt_df.index))
    tier_rows = gt_df[(gt_df["gt_tier"] == tier) & labeled_mask].copy()
    if tier_rows.empty:
        return ""

    # FIX: Sort by date descending so most recent (regime-relevant) examples
    # appear first. Falls back to criticality/event sort if no date column found.
    date_col = next(
        (c for c in ["article_date", "article_published_utc", "published"]
         if c in tier_rows.columns),
        None
    )
    if date_col:
        tier_rows = tier_rows.sort_values(date_col, ascending=False).reset_index(drop=True)
    else:
        tier_rows = tier_rows.sort_values(
            ["true_criticality", "gt_event_number"]
        ).reset_index(drop=True)

    cap = GT_TIER_CAPS.get(tier, 999)
    if len(tier_rows) > cap:
        # Balanced downsample: preserve event-type and criticality diversity
        tier_rows = (
            tier_rows.groupby("gt_event_number", group_keys=False)
            .apply(lambda x: x.sample(
                max(1, round(cap * len(x) / len(tier_rows))),
                random_state=42
            ))
            .head(cap)
        )
        tier_rows = tier_rows.reset_index(drop=True)

    content_col = (
        "full_text" if "full_text" in gt_df.columns
        else "content" if "content" in gt_df.columns
        else None
    )
    # Identify which macro fields are actually present in the GT DataFrame
    available_macro = [
        (col, label, suffix)
        for col, label, suffix in _POOL_B_MACRO_FIELDS
        if col in gt_df.columns
    ]

    tier_label = TIER_LABELS.get(tier, f"Tier {tier}")
    n_high     = (tier_rows["true_criticality"] == "high").sum()
    n_not_high = (tier_rows["true_criticality"] == "not high").sum()
    n_total    = len(tier_rows)
    lines = [
        "================================================================",
        f"TIER REFERENCE EXAMPLES -- {tier_label}",
        f"({n_total} labeled GT articles: {n_high} high / {n_not_high} not high criticality)",
        "Each example shows the article, its OUTCOME (true_criticality,",
        "true_direction, pct_15m), and the MACRO REGIME at publication.",
        "Learn: which macro conditions + event types produced high vs not high.",
        "================================================================\n",
    ]

    # Convert to plain dicts to avoid pandas MultiIndex ambiguity after groupby/apply
    rows_dicts = tier_rows.to_dict(orient="records")
    _name_to_num = {v: k for k, v in EVENT_NAMES.items()}

    for i, row in enumerate(rows_dicts, 1):
        content_preview = (
            str(row.get(content_col, ""))[:350].strip() if content_col else ""
        )
        event_name = row.get("gt_event_name") or EVENT_NAMES.get(row.get("gt_event_number", 0), "Unknown")
        evt_num = row.get("gt_event_number") or _name_to_num.get(event_name, "")
        crit    = row.get("true_criticality", "")
        dir_val = row.get("true_direction") if has_direction else None
        pct_15  = _fmt_pct(row.get("pct_15m")) if has_pct_15m else None

        lines.append(f"-- Example {i} [{crit}] ------------------------------------")
        lines.append(f"Title        : {row.get('title', '')}")
        if content_preview:
            lines.append(f"Content      : {content_preview}...")
        lines.append(f"Event        : {event_name} (#{evt_num})")
        lines.append(f"Content type : {row.get('gt_content_type', '')}")

        # Outcome block
        lines.append("OUTCOME:")
        lines.append(f"  criticality -> {crit}")
        if dir_val and str(dir_val).strip().lower() not in ("", "nan", "none"):
            lines.append(f"  direction   -> {dir_val}")
        if pct_15 and pct_15 != "N/A":
            lines.append(f"  pct_15m     -> {pct_15}")

        # Macro snapshot block -- only if at least one value is non-null
        macro_lines = []
        for col, label, suffix in available_macro:
            formatted = _fmt_macro_val(row.get(col), suffix)
            if formatted is not None:
                macro_lines.append(f"  {label:<22}: {formatted}")
        if macro_lines:
            lines.append("MACRO REGIME AT PUBLICATION:")
            lines.extend(macro_lines)

        lines.append("")

    lines.append("================================================================")
    lines.append("Now assess criticality and direction for the NEW article below.")
    lines.append("Use the outcome patterns and macro regimes above as reference.\n")
    return "\n".join(lines)


# ==============================================================================
# SHARED HELPER: call Claude with optional cached system prompt
# ==============================================================================

def call_claude(client, user_prompt: str, system_prompt: str = "",
                max_tokens: int = 128) -> dict:
    """
    Send a prompt to Claude and return parsed JSON.

    If system_prompt is provided it is sent with cache_control so Anthropic
    caches it server-side across calls. The user_prompt contains the
    per-article content that varies each call.

    max_tokens defaults to 128 (Agent 1). Agent 2 passes 512 to accommodate
    the richer JSON response schema.
    """
    kwargs = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
    }

    if system_prompt:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    message = client.messages.create(**kwargs)
    text = message.content[0].text.strip()

    # Strip markdown code fences if model adds them
    if "```" in text:
        text = text.split("```json")[-1].split("```")[0].strip()

    return json.loads(text)


# ==============================================================================
# SYSTEM PROMPT BUILDER
# ==============================================================================

TAXONOMY_SYSTEM_PROMPT = """You are an expert FX analyst classifying financial news articles for DXY (US Dollar Index) impact analysis.

DXY basket for reference: EUR 57.6% | JPY 13.6% | GBP 11.9% | CAD 9.1% | SEK 4.2% | CHF 3.6%

════════════════════════════════════════════════════════════════
CONTENT TYPE DEFINITIONS — classify this first
════════════════════════════════════════════════════════════════

hard_news : Breaking report of a CURRENT event as it happens — data release just published,
            decision just announced, statement just made. Primary source preferred.
preview   : Written BEFORE a scheduled event to discuss what analysts expect.
            Key signal: "what to expect", "preview", "ahead of", "tomorrow's", "upcoming".
recap     : Written AFTER an event to summarise what happened.
            Key signal: "here's what happened", "recap", "wrap-up", "look back".
analysis  : Commentary, interpretation, or deep-dive that doesn't break new facts.
            Often bylined opinion with sourced data but no new reporting.
opinion   : Editorial, personal viewpoint, or speculative outlook with no news anchor.

────────────────────────────────────────────────────────────────
RULE: preview and recap articles CANNOT drive high criticality
regardless of event tier. Label them correctly.
────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════
SPECIAL CATEGORIES — read these first before the named events
════════════════════════════════════════════════════════════════

0.  Irrelevant
    → Article has NO bearing on DXY or FX rates.
    Use this for:
      • Pure opinion or commentary on already-known, fully-priced-in policy
        (no new information, no sourced signal, no concrete event)
      • Corporate earnings / company results with no macro or sector-wide FX signal
      • Cryptocurrency, commodities, or equities stories with no dollar link
      • Technical chart analysis or price target updates with no event anchor
      • Lifestyle, politics, or general news unrelated to macro/trade/monetary policy
      • Forecasts and speculation with no sourced basis
    DO NOT use for articles that touch trade policy, central banks, macro data,
    geopolitics with USD implications, or fiscal/legislative developments —
    even if the article's framing is analytical rather than breaking news.

28. Other / Mixed
    → Article IS relevant to DXY/FX but doesn't fit any named category below.
    Use this for: dollar sentiment pieces with a concrete event anchor, sovereign
    credit developments, multi-topic articles, FX market structure stories,
    de-dollarisation trends, or any DXY-relevant event not covered below.

    ⚠️  FED CHAIR DIRECT QUOTES: Any article where the Federal Reserve Chair
        (Powell or successor) is directly quoted making a new statement —
        regardless of the topic framing (asset prices, AI, inflation, financial
        stability) — is relevant to DXY. Route to event 3, 31, or 28.
        Never classify as Irrelevant (event 0).

    ⚠️  SURROGATE EMPLOYMENT INDICATORS: Use event 28 (NOT events 10/11/12) for:
      • ADP private payrolls report (not official BLS data)
      • Challenger job cuts (not official BLS data)
      • JOLTS job openings (leading indicator, not payrolls)
    Events 10, 11, 12 are RESERVED for official BLS/BEA releases only.

════════════════════════════════════════════════════════════════
NAMED EVENT CATEGORIES
════════════════════════════════════════════════════════════════

## US MONETARY POLICY
1.  Fed Rate Hike (actual FOMC decision announced — not expected, not previewed)
2.  Fed Rate Cut (actual FOMC decision announced — not expected, not previewed)
3.  Hawkish Pivot / Surprise (guidance or statement tighter than market expected;
    includes FOMC hold when market priced in a cut)
4.  Dovish Pivot / Surprise (guidance or statement looser than market expected)
5.  Fed QE / Balance Sheet Expansion
6.  Fed QT / Tapering Announcement

## US INFLATION DATA
7.  CPI/PCE Above Consensus  (official BLS/BEA release, print > consensus)
8.  CPI/PCE Below Consensus  (official BLS/BEA release, print < consensus)
9.  Inflation In-Line (only use if surrounding context makes it notably significant)

## US EMPLOYMENT DATA  — official BLS/BEA releases ONLY
10. NFP/Jobs Beat (BLS nonfarm payrolls above consensus)
11. NFP/Jobs Miss (BLS nonfarm payrolls below consensus OR unemployment rate surprise)
12. Unemployment Rate Surprise (BLS unemployment materially above/below consensus)

## US GROWTH DATA
13. GDP Beat
14. GDP Miss / Recession Signal
15. Retail Sales Beat
16. Retail Sales Miss
17. PMI/ISM Beat (manufacturing or services)
18. PMI/ISM Miss

## US TRADE & FISCAL
19. Trade Tariff / Policy Shock (new tariffs announced or implemented)
20. Trade Balance Surprise
21. Debt Ceiling / Fiscal Crisis
22. Fiscal Stimulus / Spending Package

## MARKET STRUCTURE EVENTS
23. US Banking / Financial System Stress
24. Risk-Off Shock (flight to USD safety triggered by a specific event)
25. Treasury Yield Spike / Bond Market Stress
26. US Political Shock (election result, impeachment, major unexpected policy reversal)
27. Foreign Central Bank Shock (surprise action by ECB/BOJ/BOE affecting rate differential)

## POLICY EXPECTATION SHIFTS (no decision yet, but odds are moving)
29. Trade Policy Reversal Signal (tariff removal or reduction becoming likely)
30. Trade Policy Escalation Signal (new tariffs or expansion becoming more likely)
31. Fed Rate Path Repricing (market probability of cuts/hikes shifting without a decision)
32. Fiscal Policy Probability Shift (stimulus or austerity becoming more or less likely)
33. Legal/Judicial Event Affecting Economic Policy
34. Legislative Progress or Blockage (bill with fiscal or trade consequence)
35. Executive Policy Signal (White House trial balloon, leaked plan, executive order signal)

## FOREIGN CENTRAL BANK RATE CHANGES
36. ECB rate hike or cut (actual decision)
37. BOJ rate hike or cut (actual decision)
38. BOE rate hike or cut (actual decision)
39. Other major central bank rate change (RBA, SNB, BoC, etc.)"""


def build_system_prompt(few_shot_block: str = "") -> str:
    """
    Combine the static taxonomy with the (optional) Pool A few-shot block.
    Called once per run; the result is passed to every Agent 1 call_claude() call.
    few_shot_block is the output of build_classifier_few_shot_block().
    """
    if few_shot_block:
        return TAXONOMY_SYSTEM_PROMPT + "\n\n" + few_shot_block
    return TAXONOMY_SYSTEM_PROMPT


# ==============================================================================
# AGENT 1: EVENT CLASSIFIER
# ==============================================================================

def agent_1_classify(client, title, date, content, system_prompt: str = ""):
    """
    Classify article into the DXY event taxonomy.

    Returns:
      event_number : 0 = Irrelevant, 28 = Other/Mixed, 1–39 = named event
      content_type : hard_news | opinion | analysis | recap | preview | other
      confidence   : high | medium | low
    """

    user_prompt = f"""Classify the article below using the taxonomy and examples in your instructions.

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "event_number": <integer>,
  "content_type": "<hard_news|preview|recap|analysis|opinion|other>",
  "confidence": "<high|medium|low>",
}}"""

    try:
        return call_claude(client, user_prompt, system_prompt=system_prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 1 (Classifier) Error: {e}")
        return {
            "event_number": IRRELEVANT_EVENT_NUMBER,
            "content_type": "other",
            "confidence":   "low",
        }


# ==============================================================================
# AGENT 2: CRITICALITY ASSESSOR
# ==============================================================================

CRITICALITY_SYSTEM_PROMPT = """You are an FX desk analyst assessing whether a
news article is actionable for DXY trading.

════════════════════════════════════════════════════════════════
YOUR INPUTS — read carefully before forming any judgment
════════════════════════════════════════════════════════════════

You will always receive:
  • A pre-classification: event type, tier, content type
  • The article title, date, and content

You will sometimes receive:
  • A HISTORICAL RESPONSE TABLE showing how DXY actually moved after
    previous releases of the same event type. This table covers a prior
    reference period and does NOT include the current article's release.
    It shows you how this event type has historically behaved — not what
    the current release's surprise magnitude is.

If no historical table is present, the event is either unscheduled
(tariff shock, Fed speech, political development) or outside the
reference period. In that case use qualitative judgment only — the
reasoning rules below still apply but the table-specific instructions
do not.

════════════════════════════════════════════════════════════════
WHEN A HISTORICAL TABLE IS PROVIDED
════════════════════════════════════════════════════════════════

The table shows realized DXY % moves (pct_1h, pct_4h, pct_1d) after
prior releases of this event type. Use it to answer three questions:

1. DIRECTIONAL CONSISTENCY
   For prior beats: did DXY typically move up? For prior misses: did it
   move down? If direction followed the surprise in most cases, confidence
   is high. If results were mixed, set direction_confidence to low
   regardless of what the article implies.

2. TYPICAL MAGNITUDE
   What is the typical DXY move range for this event type? A pct_1h range
   of +0.1% to +0.2% tells you this event produces modest reactions.
   A range of +0.3% to +0.6% tells you it is a high-impact event type.
   Use this to calibrate criticality — not just whether a surprise occurred.

3. FOLLOW-THROUGH VS REVERSAL
   If pct_1d moves are consistently larger than pct_1h, the event produces
   durable impact — weight toward higher criticality. If pct_1d is smaller
   than pct_1h, the market fades the initial reaction — weight toward lower
   criticality for sustained trading relevance.

IMPORTANT: The table does not tell you the current release's surprise
magnitude. You must infer whether the current event is a surprise from
the article text. The table tells you how the market has historically
reacted to surprises of this type — use it to calibrate your response
once you have assessed whether a surprise occurred.

════════════════════════════════════════════════════════════════
EVIDENCE HIERARCHY
════════════════════════════════════════════════════════════════

For CRITICALITY (apply in this order):
  1. Hard floors below — these override everything
  2. If table present: typical DXY magnitude range for this event type
     combined with whether article indicates a surprise occurred
  3. If table present: directional consistency and follow-through pattern
  4. Tier default — apply only when table is absent or has fewer than
     3 rows of the relevant surprise direction
  5. Article text — always use for context, never as primary evidence
     for scheduled quantitative releases

For DIRECTION:
  1. Hard floors below — these override everything
  2. For data releases (NFP, CPI, GDP, retail sales): read the ACTUAL
     FIGURE vs. CONSENSUS from the article text directly — this always
     takes priority over event_number classification
  3. If table present: directional consistency from historical rows
     of the same surprise direction as the current article
  4. If table absent: theoretical implied direction from event taxonomy
  5. Article text — use for additional context

For DIRECTION CHAIN:
  When table is present, write the chain using table observations:
    [article surprise assessment] → [historical reaction pattern] →
    [current DXY implication]
  Example: "article indicates CPI beat → prior beats averaged +0.31%
            pct_1h with consistent follow-through → upward DXY pressure
            expected with high confidence"

  When table is absent, write the chain using macro reasoning:
    [event type] → [rate/policy implication] → [DXY effect]
  Example: "tariff escalation on EU goods → reduces EUR demand →
            EUR/USD falls → DXY rises"

════════════════════════════════════════════════════════════════
HARD FLOORS — these override all other reasoning
════════════════════════════════════════════════════════════════

── FOMC DECISIONS (ALWAYS HIGH) ────────────────────────────────
All actual FOMC decisions (rate cut, hike, or hold) are HIGH criticality.
Forward guidance — dot plot, pace language, vote count, press conference
framing — always carries incremental information not priced in prior to
release. Never downgrade an FOMC announcement to low on the grounds that
the decision itself was expected.

── FOMC CARVE-OUT FOR "ALREADY PRICED IN" ──────────────────────
For FOMC events (event types 1, 2, 3, 4, 31), the "already priced in"
and "widely expected" downgrade rules do NOT apply to:
  • The decision announcement itself (rate cut, hike, or hold)
  • The press conference or Powell speech content
  • Dot plot or rate path guidance
  • Vote composition and dissents
These always contain incremental information not fully priced in before
release. Only downgrade an FOMC article when ALL THREE conditions hold:
  (a) Published 12+ hours after the event
  (b) Contains no new quotes, dissent detail, or forward guidance
  (c) Is purely descriptive recap of what was already widely reported

── CONTENT TYPE FLOORS (non-negotiable) ────────────────────────
- opinion, analysis, preview, recap → ALWAYS not high, no exceptions
- Article written BEFORE a scheduled event → ALWAYS not high
- Post-event recap of something already fully priced in → not high
- If your reasoning would include the phrases "already priced in",
  "widely expected", "no new information", or "no surprise" → must
  return not high
  EXCEPTION: FOMC carve-out above takes precedence for Fed events.

── SURROGATE INDICATORS (ALWAYS LOW) ───────────────────────────
ADP private payrolls, Challenger job cuts, JOLTS job openings,
University of Michigan sentiment, Conference Board consumer confidence.
Never assign high to a surrogate regardless of surprise size.
If the event is classified as event 28 (Other/Mixed) and content is a
surrogate indicator, return not high.

── NFP/DATA RELEASE DIRECTION OVERRIDE ─────────────────────────
For NFP and all scheduled data releases (events 7–18), direction must
be derived from the ACTUAL FIGURE vs. CONSENSUS stated in the article
text. The event_number classification (beat vs. miss) may be wrong.
  • If the article says payrolls "rose more than expected", "beat
    forecasts", or reports a positive surprise → direction = up
  • If the article says payrolls "fell short", "missed", or reports
    a negative surprise → direction = down
  • Article text always overrides the event_number for direction on
    quantitative data releases.

── GEOGRAPHY CHECK ──────────────────────────────────────────────
DXY basket weights: EUR 57.6% | JPY 13.6% | GBP 11.9% | CAD 9.1% |
                    SEK 4.2%  | CHF 3.6%
Tariffs or trade events affecting non-basket economies → not high unless
the article explicitly links them to USD reserve status or broad dollar
demand. ECB, BOJ, BOE, BOC policy actions are always DXY-relevant.
PBOC, RBA, RBNZ actions → not high unless causing broad safe-haven USD demand.

── SOURCE QUALITY ───────────────────────────────────────────────
Authoritative primary sources: AP, Reuters, Bloomberg, WSJ, CNBC,
Fed.gov, BLS.gov, BEA.gov, Treasury.gov, BIS, IMF official releases.
Secondary sources reporting a primary decision → not high and note
in reasoning. If source credibility is uncertain → not high.

── DOWNGRADE TRIGGERS ───────────────────────────────────────────
- Article is vague or lacks concrete details → downgrade
- Event described as expected with no new information → downgrade
- "The Fed is expected to...", "Markets anticipate...", "Analysts
  predict..." → downgrade
- Data release described as delayed, revised, or partially priced in
  → not high
  EXCEPTION: FOMC carve-out above takes precedence for Fed events.

── ESCALATION TRIGGERS (Tier 3/4 unscheduled events only) ──────
- "emergency meeting", "unscheduled rate decision", "surprise cut/hike"
- Explicitly described as historic or unprecedented in scope
- Tariffs raised to 50%+ on DXY-basket countries
- Fiscal package $500B+ | Bank failure $500B+ assets
- "dollar plunged/surged X%", "yields spiked X bps", "circuit breakers"
- USD safe-haven demand described as materialising, not merely predicted
- Senior Fed/Treasury official directly contradicting current policy
- Sanctions or asset freezes impairing dollar liquidity or reserve status

════════════════════════════════════════════════════════════════
MACRO REGIME CONTEXT — how to use it
════════════════════════════════════════════════════════════════

When a MACRO REGIME CONTEXT block is present, use it to calibrate
criticality and direction within the tier default range. It does not
override hard floors but adjusts your judgment for the current environment.

── POSITIONING IS A CRITICALITY SIGNAL ONLY, NOT A DIRECTION SIGNAL ──
DXY z-score and CFTC positioning tell you whether a move could be larger
than usual — they do NOT tell you which direction the move will go.
An oversold DXY (z-score < -1.5) means bullish events may have more
follow-through AND bearish events face less resistance — it amplifies
BOTH directions equally. Never use "DXY is oversold" to infer direction
= "up". Direction must come from the event's macro mechanism or the
historical table, never from positioning alone.

Key signals to check before forming your judgment:

DXY POSITIONING: If dxy_zscore_52w_broad > 1.5, DXY is extended and
  crowded — downgrade CRITICALITY confidence for bullish events (less room
  to run). If < -1.5, DXY is oversold — high-criticality events of either
  direction may produce larger-than-usual moves.

FED REGIME: If regime_flag=active cycle and months_since_last_change is
  low, Fed-sensitive events (CPI, NFP, FOMC) carry higher criticality
  because each data point can shift the next decision. If on hold for
  many months, the same data has less immediate policy impact.

VIX REGIME: If vix_regime=high and vix_20d_change > 0 (rising), risk-off
  safe-haven demand amplifies criticality for geopolitical and financial
  stress events (events 19, 24, 25, 26). If VIX is low and falling,
  these events have lower follow-through.

US-DE SPREAD: Widening spread (USD rates rising vs EUR rates) is
  structurally bullish for DXY. Narrowing is bearish. Use to assess
  whether the macro backdrop supports or resists the event's implied
  DXY direction.

SAHM INDICATOR: If >= 0.5, recession signal is active — labor market
  misses carry escalated criticality and GDP misses become high regardless
  of tier default."""


# ==============================================================================
# MACRO CONTEXT BUILDER
# ==============================================================================

# Columns from the input CSV that are not macro variables.
# FIX: added published_utc, url, article_published_raw, found_via, actual_link
# to prevent junk columns from leaking into the macro_row dict.
_ARTICLE_COLS = {
    "id", "title", "link", "published", "source", "query", "priority_weight",
    "article_date", "prediction_date", "fetched_at", "title_len", "actual_link",
    "full_text", "article_published_raw", "article_published_utc", "found_via",
    "published_utc", "url",
}

# Macro columns that are always null or zero-variance — skip entirely.
# FIX: added cftc_net_usd_zscore which is all-zero in current data and
# causes the model to incorrectly infer "neutral positioning" throughout.
_NULL_MACRO_COLS = {"gold_20d_return", "cftc_net_usd_zscore"}


def build_macro_context(macro_row: dict, event_number: int) -> str:
    """
    Format point-in-time macro variables into a thematic prompt block
    for injection into Agent 2's user prompt.

    Only injects groups relevant to the event type. Silently skips nulls.
    Returns empty string if macro_row is None or empty.
    """
    if not macro_row:
        return ""

    import math

    def _v(key, precision=2, suffix=""):
        """Return formatted value or None if missing/null."""
        val = macro_row.get(key)
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return str(val) if val else None
        if math.isnan(f):
            return None
        return f"{f:.{precision}f}{suffix}"

    def _s(key):
        """Return string value or None."""
        val = macro_row.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return str(val).strip() or None

    lines = ["\n── MACRO REGIME CONTEXT (point-in-time at article date) ────────"]

    # ── Fed & Rates — always inject ──────────────────────────────────────────
    fed_lines = []
    if _v("fed_funds_rate"):
        fed_lines.append(f"  Fed Funds Rate          : {_v('fed_funds_rate')}%")
    if _v("months_since_last_change", 0):
        fed_lines.append(f"  Months since last change: {_v('months_since_last_change', 0)}")
    regime_raw = macro_row.get("regime_flag")
    if regime_raw is not None:
        try:
            flag = int(float(regime_raw))
            label = {0: "on hold", 1: "active cycle", 2: "emergency"}.get(flag, "unknown")
            fed_lines.append(f"  Fed regime              : {label}")
        except (TypeError, ValueError):
            pass
    if _v("yield_curve_2s10s"):
        fed_lines.append(f"  Yield curve 2s10s       : {_v('yield_curve_2s10s')}%")
    if _v("us_de_2yr_spread"):
        fed_lines.append(f"  US-DE 2yr spread        : {_v('us_de_2yr_spread')}%")
    if _v("us_2yr_yield"):
        fed_lines.append(f"  US 2yr yield            : {_v('us_2yr_yield')}%")
    if _v("us_10yr_yield"):
        fed_lines.append(f"  US 10yr yield           : {_v('us_10yr_yield')}%")
    if _v("real_yield_10yr"):
        fed_lines.append(f"  Real yield 10yr         : {_v('real_yield_10yr')}%")
    if fed_lines:
        lines.append("  [Fed & Rates]")
        lines.extend(fed_lines)

    # ── DXY Positioning — always inject ──────────────────────────────────────
    # Note: cftc_net_usd_zscore excluded via _NULL_MACRO_COLS (all-zero in data)
    dxy_lines = []
    if _v("dxy_zscore_52w_broad"):
        dxy_lines.append(f"  DXY 52w z-score         : {_v('dxy_zscore_52w_broad')}")
    if _v("dxy_20d_return_broad"):
        dxy_lines.append(f"  DXY 20d return          : {_v('dxy_20d_return_broad')}%")
    if _v("dxy_pct_above_200d_broad"):
        dxy_lines.append(f"  DXY % above 200d MA     : {_v('dxy_pct_above_200d_broad')}%")
    if dxy_lines:
        lines.append("  [DXY Positioning]")
        lines.extend(dxy_lines)

    # ── Risk Environment — always inject ─────────────────────────────────────
    risk_lines = []
    vix_val  = _v("vix")
    vix_reg  = _s("vix_regime")
    if vix_val and vix_reg:
        risk_lines.append(f"  VIX                     : {vix_val} ({vix_reg} regime)")
    elif vix_val:
        risk_lines.append(f"  VIX                     : {vix_val}")
    if _v("vix_20d_change"):
        risk_lines.append(f"  VIX 20d change          : {_v('vix_20d_change')}")
    if _v("sp500_20d_return"):
        risk_lines.append(f"  S&P500 20d return       : {_v('sp500_20d_return')}%")
    if _v("wti_20d_return"):
        risk_lines.append(f"  WTI oil 20d return      : {_v('wti_20d_return')}%")
    if risk_lines:
        lines.append("  [Risk Environment]")
        lines.extend(risk_lines)

    # ── Inflation State — inject for inflation events (7,8,9) ────────────────
    if event_number in {7, 8, 9}:
        inf_lines = []
        if _v("cpi_yoy"):
            inf_lines.append(f"  CPI YoY                 : {_v('cpi_yoy')}%")
        if _v("core_cpi_yoy"):
            inf_lines.append(f"  Core CPI YoY            : {_v('core_cpi_yoy')}%")
        if _v("core_pce_yoy"):
            inf_lines.append(f"  Core PCE YoY            : {_v('core_pce_yoy')}%")
        if _v("pce_gap"):
            inf_lines.append(f"  PCE gap vs 2% target    : {_v('pce_gap')}%")
        if _v("breakeven_10yr"):
            inf_lines.append(f"  10yr breakeven          : {_v('breakeven_10yr')}%")
        if inf_lines:
            lines.append("  [Inflation State]")
            lines.extend(inf_lines)

    # ── Labor Market — inject for employment events (10,11,12) ───────────────
    if event_number in {10, 11, 12}:
        lab_lines = []
        if _v("unemployment_rate"):
            lab_lines.append(f"  Unemployment rate       : {_v('unemployment_rate')}%")
        if _v("nfp_3m_avg", 0):
            lab_lines.append(f"  NFP 3m avg              : {_v('nfp_3m_avg', 0)}K")
        if _v("sahm_indicator"):
            lab_lines.append(f"  Sahm indicator          : {_v('sahm_indicator')} (recession signal >=0.5)")
        if _v("initial_claims_4wma", 0):
            lab_lines.append(f"  Initial claims 4wma     : {_v('initial_claims_4wma', 0)}")
        if lab_lines:
            lines.append("  [Labor Market State]")
            lines.extend(lab_lines)

    # ── Growth — inject for GDP/retail events (13,14,15,16) ──────────────────
    if event_number in {13, 14, 15, 16}:
        gdp_lines = []
        if _v("gdp_qoq"):
            gdp_lines.append(f"  GDP q/q (latest)        : {_v('gdp_qoq')}%")
        if _v("retail_sales_mom"):
            gdp_lines.append(f"  Retail sales m/m        : {_v('retail_sales_mom')}%")
        if _v("sahm_indicator"):
            gdp_lines.append(f"  Sahm indicator          : {_v('sahm_indicator')} (recession signal >=0.5)")
        if gdp_lines:
            lines.append("  [Growth State]")
            lines.extend(gdp_lines)

    # ── Credit Conditions — inject for stress events (21,23,24,25) ───────────
    if event_number in {21, 23, 24, 25}:
        cred_lines = []
        if _v("ig_oas"):
            cred_lines.append(f"  IG OAS                  : {_v('ig_oas')}%")
        if _v("hy_oas"):
            cred_lines.append(f"  HY OAS                  : {_v('hy_oas')}%")
        if _v("hy_ig_spread"):
            cred_lines.append(f"  HY-IG spread            : {_v('hy_ig_spread')}%")
        if cred_lines:
            lines.append("  [Credit Conditions]")
            lines.extend(cred_lines)

    lines.append("────────────────────────────────────────────────────────────────")

    # If nothing was populated beyond the header and footer, return empty
    if len(lines) <= 2:
        return ""

    return "\n".join(lines)


def extract_macro_row(row: pd.Series) -> dict:
    """
    Extract macro variable columns from an article row.
    Excludes article-specific columns (_ARTICLE_COLS) and known-null/zero-variance
    macro columns (_NULL_MACRO_COLS). The only timestamp used for article dating
    is article_published_utc; published_utc is excluded as a duplicate/junk field.
    Returns dict of {col: value} for non-null macro fields only.
    """
    macro = {}
    for col, val in row.items():
        if col in _ARTICLE_COLS:
            continue
        if col in _NULL_MACRO_COLS:
            continue
        macro[col] = val
    return macro


def agent_2_criticality(client, title, date, content, event_number, content_type,
                         system_prompt: str = "", consensus_context: str = "",
                         macro_context: str = "", tier_few_shot_block: str = ""):
    """
    Assess criticality of a relevant article.

    Default by tier (factual hard_news only):
      Tier 1 -> high  |  Tier 2/3/4 -> not high

    Hard floors (enforced both in prompt and in Python post-call):
      preview / recap / analysis / opinion -> always not high

    consensus_context:    pre-formatted historical DXY reaction table string.
    macro_context:        pre-formatted point-in-time macro regime string.
    tier_few_shot_block:  Pool B -- GT examples from the same tier.
                          Injected FIRST so the model sees real criticality/
                          direction patterns before the quantitative context.
    """

    tier       = EVENT_TIERS.get(event_number, 4)
    tier_label = TIER_LABELS.get(tier)
    event_name = EVENT_NAMES.get(event_number, "Other / Mixed")
    is_non_actionable = content_type in NON_ACTIONABLE_CONTENT_TYPES

    if is_non_actionable:
        default_level = "not high"
    elif tier == 1:
        default_level = "high"
    else:
        default_level = "not high"

    if consensus_context:
        table_reminder = ("A historical response table is provided. "
                          "Read it before forming any judgment on criticality or direction.")
    else:
        table_reminder = "No historical data available. Use qualitative reasoning only."

    # Injection order: tier few-shot -> macro -> consensus -> article
    # Tier examples first so the model internalises tier-level patterns before
    # seeing the quantitative context blocks.
    tier_block      = f"\n{tier_few_shot_block}\n" if tier_few_shot_block else ""
    consensus_block = f"\n{consensus_context}\n"   if consensus_context   else ""
    macro_block     = f"\n{macro_context}\n"       if macro_context       else ""

    user_prompt = f"""Assess the criticality of this article for DXY/FX trading.

{table_reminder}

Pre-classification:
  Event type          : {event_name} (Category #{event_number})
  Research-based tier : {tier_label}
  Content type        : {content_type}
  Default criticality : {default_level}  <- starting point; confirm or override
{tier_block}{macro_block}{consensus_block}
Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "criticality_level":    "<high|not high>",
  "reasoning":            "<1-2 sentences -- cite tier examples (15-min outcomes) or table patterns if used, article text if not>",
  "direction":            "<up|down|neutral|unclear>",
  "direction_chain":      "<chain of reasoning as described in your instructions>",
  "direction_confidence": "<high|medium|low>",
  "table_used":           <true|false>
}}"""

    try:
        result = call_claude(client, user_prompt, system_prompt=system_prompt,
                             max_tokens=512)
        # Enforce hard floor in Python -- model cannot override this
        if is_non_actionable:
            result["criticality_level"] = "not high"
        return result
    except Exception as e:
        print(f"  WARNING  Agent 2 (Criticality) Error: {e}")
        return {"criticality_level": "not high"}


# ==============================================================================
# HISTORICAL RESPONSE TABLE
# ==============================================================================

def build_historical_response_table(event_number: int,
                                     article_date: str,
                                     consensus_df,
                                     n_rows: int = 12) -> str:
    """
    Build a plain-text table of historical DXY reactions for the same
    event type, to inject into Agent 2's prompt.

    Queries consensus_df for all rows matching the base event number
    that occurred strictly before article_date, takes the last n_rows,
    and formats them as a fixed-width table with columns:
        date | actual | forecast | surprise | magnitude | pct_1h | pct_4h | pct_1d

    Returns empty string if fewer than 3 matching rows are available
    (not enough history to be useful).
    """
    if consensus_df is None:
        return ""

    # Resolve to base event number (beat/miss both map to same base)
    base_numbers = {8: 7, 11: 10, 14: 13, 16: 15, 18: 17}
    lookup_base  = base_numbers.get(event_number, event_number)

    try:
        cutoff = pd.Timestamp(article_date, tz="UTC")
    except Exception:
        return ""

    # Ensure release_date_utc is tz-aware for comparison
    rdu = pd.to_datetime(consensus_df["release_date_utc"], utc=True, errors="coerce")
    mask = (consensus_df["base_event_number"] == lookup_base) & (rdu < cutoff)
    rows = consensus_df[mask].copy()
    rows["_rdu"] = rdu[mask]
    rows = rows.sort_values("_rdu").tail(n_rows)

    if len(rows) < 3:
        return ""

    header = (f"{'date':<12}  {'actual':>8}  {'forecast':>9}  "
              f"{'surprise':>9}  {'magnitude':>10}  "
              f"{'pct_1h':>7}  {'pct_4h':>7}  {'pct_1d':>7}")
    sep    = "-" * len(header)
    lines  = [
        f"Historical DXY reactions — {EVENT_NAMES.get(event_number, 'event')} "
        f"(last {len(rows)} releases before {article_date[:10]})",
        header, sep,
    ]

    for _, r in rows.iterrows():
        pct_1h = f"{r['pct_1h']:+.3f}%" if pd.notna(r.get("pct_1h")) else "    N/A"
        pct_4h = f"{r['pct_4h']:+.3f}%" if pd.notna(r.get("pct_4h")) else "    N/A"
        pct_1d = f"{r['pct_1d']:+.3f}%" if pd.notna(r.get("pct_1d")) else "    N/A"
        mag    = f"{r['surprise_magnitude']:+.4f}" if pd.notna(r.get("surprise_magnitude")) else "     N/A"
        lines.append(
            f"{str(r['release_date']):<12}  "
            f"{str(r.get('actual_raw',''))[:8]:>8}  "
            f"{str(r.get('forecast_raw',''))[:9]:>9}  "
            f"{str(r.get('surprise_direction',''))[:9]:>9}  "
            f"{mag:>10}  "
            f"{pct_1h:>7}  {pct_4h:>7}  {pct_1d:>7}"
        )

    return "\n".join(lines)


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def analyze_article_dxy(client, title, date, content,
                         classifier_system_prompt: str = "",
                         criticality_system_prompt: str = "",
                         consensus_df=None,
                         macro_row: dict = None,
                         gt_df=None,
                         verbose=True):
    """
    Runs the 2-agent pipeline on a single article.

    Agent 1 (Classifier)  -- always runs; uses cached taxonomy + Pool A few-shot system prompt
    Agent 2 (Criticality) -- skipped when event_number == 0 (Irrelevant);
                             receives consensus history context when consensus_df provided;
                             receives macro regime context when macro_row provided;
                             receives Pool B tier few-shot block when gt_df provided

    gt_df: the full ground truth DataFrame from load_gt_dataframe(). When provided,
           build_tier_few_shot_block() is called to inject same-tier GT examples
           into Agent 2's prompt, anchoring criticality/direction predictions to
           observed outcomes for articles of this tier.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:75]}")
        print(f"{'='*60}")

    # --- Agent 1: Classify ---
    if verbose: print("\n[Agent 1] Classification...")
    a1 = agent_1_classify(
        client, title, date, content,
        system_prompt=classifier_system_prompt,
    )

    event_number  = a1.get("event_number", IRRELEVANT_EVENT_NUMBER)
    is_irrelevant = (event_number == IRRELEVANT_EVENT_NUMBER)
    tier          = EVENT_TIERS.get(event_number, 4)

    if verbose:
        if is_irrelevant:
            print(f"IRRELEVANT | content_type={a1.get('content_type')}")
        else:
            print(f"#{event_number}: {EVENT_NAMES.get(event_number)}")
            print(f"   content_type : {a1.get('content_type')}")
            print(f"   confidence   : {a1.get('confidence')}")
            print(f"   tier         : {TIER_LABELS.get(tier)}")

    if is_irrelevant:
        return {"agent1": a1, "agent2": None}

    # --- Build consensus context for Agent 2 ---
    consensus_record  = lookup_consensus_history(event_number, consensus_df,
                            article_date=date) if consensus_df is not None else None
    history_table     = build_historical_response_table(event_number, date,
                            consensus_df) if consensus_df is not None else ""
    consensus_context = build_consensus_context_with_history(consensus_record,
                            history_table)

    # --- Build macro context for Agent 2 ---
    macro_context = build_macro_context(macro_row, event_number) if macro_row else ""

    # --- Build Pool B tier few-shot block for Agent 2 ---
    tier_few_shot_block = build_tier_few_shot_block(gt_df, tier) if gt_df is not None else ""

    if verbose and consensus_context:
        print(f"   consensus context: {len(history_table.splitlines())} history rows")
    if verbose and macro_context:
        print(f"   macro context: injected ({len(macro_context)} chars)")
    if verbose and tier_few_shot_block:
        tier_rows_count = tier_few_shot_block.count("-- Example")
        print(f"   tier few-shot: {tier_rows_count} Tier {tier} examples injected (Pool B)")

    # --- Agent 2: Criticality ---
    if verbose: print("\n[Agent 2] Criticality assessment...")
    a2 = agent_2_criticality(
        client, title, date, content,
        event_number=event_number,
        content_type=a1.get("content_type", "other"),
        system_prompt=criticality_system_prompt,
        consensus_context=consensus_context,
        macro_context=macro_context,
        tier_few_shot_block=tier_few_shot_block,
    )

    if verbose:
        level = a2.get("criticality_level", "not high")
        marker = "HIGH" if level == "high" else "not high"
        print(f"[{marker}] criticality_level={level}")
        if a2.get("reasoning"):
            print(f"   reasoning: {a2.get('reasoning')[:120]}")

    return {"agent1": a1, "agent2": a2}


# ==============================================================================
# CSV PROCESSING PIPELINE
# ==============================================================================

def process_csv_to_csv(
    input_csv_path,
    output_csv_path,
    ground_truth_csv_path=None,
    max_few_shot_examples=12,
    test_row_index=None,
    verbose=True,
):
    """
    Read articles from CSV, run 2-agent pipeline, write results CSV.
    ALL rows are output -- irrelevant articles included with blank agent 2 columns.

    Two-pool few-shot architecture (when ground_truth_csv_path is provided):

      Pool A -- Agent 1 (Classifier):
        A balanced cross-event sample of GT articles is built ONCE at startup
        and injected into the classifier system prompt. Anthropic caches this
        server-side so repeated calls cost ~10% of normal token price.

      Pool B -- Agent 2 (Criticality/Direction):
        All GT rows whose event maps to the SAME TIER as the classified article
        are injected into Agent 2's user prompt at call time. Sorted by date
        descending so the most recent (regime-relevant) examples appear first.

    Pool B is fully active once true_criticality and true_direction columns
    are present in the GT CSV. true_criticality ('high'/'not high') is required
    for criticality few-shot; true_direction is optional and sparse is fine --
    only rows where it is non-null are shown with direction labels.

    Columns derived via pandas after the run (zero LLM cost):
      event_name       -- looked up from EVENT_NAMES by event_number
      event_tier_label -- looked up from TIER_LABELS by event_tier
      is_relevant      -- True where event_number != 0
      is_critical      -- True where criticality_level == "high"
      processed_at     -- timestamp for the batch
    """

    client = get_client()

    # --- Load GT DataFrame once (shared by Pool A and Pool B) ---
    gt_df = None
    if ground_truth_csv_path:
        gt_df          = load_gt_dataframe(ground_truth_csv_path)
        few_shot_block = build_classifier_few_shot_block(gt_df, max_examples=max_few_shot_examples)
        tier_counts    = gt_df["gt_tier"].value_counts().sort_index()
        print(f"Pool A (Agent 1): {len(few_shot_block)} chars "
              f"~{len(few_shot_block)//4} tokens (cached after first call)")
        labeled      = gt_df[gt_df.get("gt_labeled", pd.Series(False, index=gt_df.index))]
        tier_labeled = labeled["gt_tier"].value_counts().sort_index()
        tier_str = ", ".join(
            f"T{t}={n} ({(labeled[labeled['gt_tier']==t]['true_criticality']=='high').sum()}H)"
            for t, n in tier_labeled.items()
        ) if not tier_labeled.empty else "none"
        print(f"Pool B (Agent 2): {len(labeled)} labeled rows with macro snapshots "
              f"[{tier_str}] (H=high criticality)\n")
    else:
        few_shot_block = ""
        print("No ground truth CSV provided -- running without few-shot examples.\n")

    # Classifier: taxonomy + Pool A few-shot examples (cached system prompt)
    classifier_system_prompt  = build_system_prompt(few_shot_block)
    # Criticality: separate system prompt; Pool B injected per-article in user prompt
    criticality_system_prompt = CRITICALITY_SYSTEM_PROMPT

    # --- Load consensus table once (used by Agent 2 for historical context) ---
    consensus_df = load_consensus_table("data/consensus_table.csv")
    print(f"📊 Consensus table loaded: {len(consensus_df)} rows\n")

    # --- Load input ---
    print(f"📂 Reading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"✅ Loaded {len(df)} rows")

    if test_row_index is not None:
        if test_row_index >= len(df):
            raise ValueError(
                f"test_row_index={test_row_index} is out of range. "
                f"CSV has {len(df)} rows (valid: 0–{len(df)-1})."
            )
        print(f"⚡ TEST MODE: processing row {test_row_index} only\n")
        df = df.iloc[[test_row_index]]

    results = []
    start_time = time.time()

    for idx, row in df.iterrows():

        title      = str(row.get("title", ""))
        date       = str(row.get("article_published_utc", row.get("published", row.get("article_date", ""))))
        article_id = row.get("id", idx)

        raw_full_text = row.get("full_text", "")
        content = (
            str(raw_full_text).strip()
            if pd.notna(raw_full_text) and str(raw_full_text).strip()
            else title
        )

        # Extract macro variables for this article row
        macro_row = extract_macro_row(row)

        # Progress + ETA
        n_done  = len(results)
        n_total = len(df)
        if n_done > 0:
            elapsed   = time.time() - start_time
            rate      = n_done / elapsed          # articles per second
            remaining = (n_total - n_done) / rate
            eta_min   = remaining / 60
            print(f"\n[{n_done}/{n_total}] ETA ~{eta_min:.1f} min remaining")

        analysis = analyze_article_dxy(
            client, title, date, content,
            classifier_system_prompt=classifier_system_prompt,
            criticality_system_prompt=criticality_system_prompt,
            consensus_df=consensus_df,
            macro_row=macro_row,
            gt_df=gt_df,
            verbose=verbose,
        )

        a1 = analysis["agent1"]
        a2 = analysis["agent2"]   # None when irrelevant

        event_num = a1.get("event_number", IRRELEVANT_EVENT_NUMBER)
        tier      = EVENT_TIERS.get(event_num) if event_num != IRRELEVANT_EVENT_NUMBER else None

        out_row = {
            # ---- original columns ----
            "id":              article_id,
            "title":           title,
            "link":            row.get("link", ""),
            "published":       row.get("published", ""),
            "source":          row.get("source", ""),
            "query":           row.get("query", ""),
            "priority_weight": row.get("priority_weight", ""),
            "article_date":    row.get("article_date", ""),
            "prediction_date": row.get("prediction_date", ""),
            "fetched_at":      row.get("fetched_at", ""),
            "title_len":       row.get("title_len", len(title)),
            "article_published_utc": row.get("article_published_utc", ""),
            # ---- agent 1 raw output ----
            "event_number":              event_num,
            "content_type":              a1.get("content_type"),
            "classification_confidence": a1.get("confidence"),
            "event_tier":                tier,
            # ---- agent 2 raw output (None for irrelevant) ----
            "criticality_level":    a2.get("criticality_level")    if a2 else None,
            "reasoning":            a2.get("reasoning")            if a2 else None,
            "direction":            a2.get("direction")            if a2 else None,
            "direction_chain":      a2.get("direction_chain")      if a2 else None,
            "direction_confidence": a2.get("direction_confidence") if a2 else None,
            "direction_source":     a2.get("direction_source")     if a2 else None,
            "direction_conflict":   a2.get("direction_conflict")   if a2 else None,
            "table_used":           a2.get("table_used")           if a2 else None,
        }

        results.append(out_row)

        # Rate-limit pacing — sleep between articles so we stay well under
        # the 450k input tokens/min ceiling. Skip sleep after the last article.
        if n_done < n_total - 1:
            time.sleep(INTER_ARTICLE_DELAY)

    # --- Build output DataFrame ---
    out_df = pd.DataFrame(results)

    # --- Derive columns via pandas (no LLM cost) ---
    out_df["event_name"]       = out_df["event_number"].map(EVENT_NAMES)
    out_df["event_tier_label"] = out_df["event_tier"].map(TIER_LABELS)
    out_df["is_relevant"]      = out_df["event_number"] != IRRELEVANT_EVENT_NUMBER
    out_df["is_critical"]      = out_df["criticality_level"] == "high"
    out_df["processed_at"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_df.to_csv(output_csv_path, index=False)

    # --- Summary ---
    relevant_count = out_df["is_relevant"].sum()
    critical_count = out_df["is_critical"].sum()
    rows_done      = len(out_df)

    print(f"\n{'='*60}")
    print(f"Saved -> {output_csv_path}")
    print(f"\nSUMMARY")
    print(f"   Model                  : {MODEL}")
    if gt_df is not None:
        labeled_ct = gt_df.get("gt_labeled", pd.Series(False)).sum()
        has_crit   = "true_criticality" in gt_df.columns
        has_dir    = "true_direction" in gt_df.columns
        label_note = (
            f"{labeled_ct} labeled rows (criticality + direction)"
            if (has_crit and has_dir)
            else f"{labeled_ct} labeled rows (criticality only)"
            if has_crit
            else "taxonomy only (no labels)"
        )
        print(f"   Few-shot Pool A (A1)   : {max_few_shot_examples} examples (classifier calibration)")
        print(f"   Few-shot Pool B (A2)   : tier-matched GT ({len(gt_df)} rows, {label_note}, date-sorted)")
    else:
        print(f"   Few-shot               : No")
    print(f"   Rows processed         : {rows_done}")
    print(f"   Relevant               : {relevant_count}")
    print(f"   Irrelevant             : {rows_done - relevant_count}")
    print(f"   Critical               : {critical_count}")
    if relevant_count > 0:
        print(f"   Critical rate          : {critical_count / relevant_count * 100:.1f}% of relevant")
        rel = out_df[out_df["is_relevant"]]
        print(f"\nEvent distribution (relevant articles):")
        for event, count in rel["event_name"].value_counts().items():
            print(f"   {count:>3}x  {event}")
        print(f"\nTier distribution:")
        for label, count in rel["event_tier_label"].value_counts().items():
            print(f"   {count:>3}x  {label}")

    return out_df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    output_df = process_csv_to_csv(
        input_csv_path="data/aug_mar_cutoff_cln.csv",
        output_csv_path="data/results_gt_ext200.csv",
        ground_truth_csv_path="data/gt_cum200.csv",  # GT with true_criticality + true_direction
        # Pool A: examples injected into Agent 1 classifier system prompt (cached)
        # Balanced sample across event types -- 2 per event type up to this cap
        max_few_shot_examples=42,
        # Pool B: all same-tier GT rows injected into Agent 2 user prompt per article
        # Controlled by GT_TIER_CAPS at top of file (default: T1=16, T2/T3=all)
        test_row_index=None,        # set to an int to test a single row
        verbose=True,
    )
