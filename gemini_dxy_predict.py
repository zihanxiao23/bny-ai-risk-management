import os
import pandas as pd
import json
import time
from datetime import datetime
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# CLIENT SETUP
# Uses GEMINI_API_KEY from .env file in the same directory as this script.
# Create a .env file with:  GEMINI_API_KEY=your_key_here
# Get your key from: https://aistudio.google.com/apikey
# ==============================================================================

def get_client():
    """Initialize Gemini client using .env API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
    return genai.Client(api_key=api_key)

# Model selection — free tier daily limits as of early 2026 (post Dec 2025 cuts):
#   gemini-1.5-flash-8b   → 15 RPM, 1,500 RPD  ← best free tier headroom; lightweight
#   gemini-1.5-flash      → 15 RPM, 1,500 RPD  (heavier, same limits)
#   gemini-2.5-flash-lite → 15 RPM,    20 RPD  (Google cut this 92% in Dec 2025 — avoid)
#   gemini-2.5-flash      → 10 RPM,   250 RPD
#
# gemini-1.5-flash-8b is the "lite" of the 1.5 family: smaller model, same free
# tier RPD headroom, perfectly capable for classification tasks like this.
# Switch to a 2.5 model only after enabling Cloud Billing (Tier 1).
MODEL = "gemini-2.5-flash-lite"  # Free tier: 15 RPM, 1,500 RPD


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


# ==============================================================================
# SHARED HELPER: call Gemini and parse JSON response
# ==============================================================================

class RateLimitExceeded(Exception):
    """Raised when Gemini returns a daily/per-minute quota error.
    Triggers immediate CSV flush and clean exit in process_csv_to_csv."""
    pass


def call_gemini(client, prompt: str) -> dict:
    """
    Send a prompt to Gemini and return parsed JSON.
    Raises RateLimitExceeded on 429/RESOURCE_EXHAUSTED so the caller can
    immediately write partial results to CSV rather than losing all progress.
    All other errors are re-raised normally.
    """
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 512,
            }
        )
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            raise RateLimitExceeded(err)
        raise

    text = response.text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```json")[-1].split("```")[0].strip()

    return json.loads(text)


# ==============================================================================
# AGENT 1: EVENT CLASSIFIER  (merged from old Agents 1 + 2)
#
# Replaces the old two-agent relevance + classification pipeline with a single
# call that simultaneously decides relevance AND classifies the event type.
#
# Key design decision: "Irrelevant" is a first-class category (event_number=0),
# not a pre-filter flag. This means:
#   - The model has the full event taxonomy in context when deciding irrelevance,
#     so it can't accidentally discard a rare-earth export control or a BOJ shock
#     just because it looks exotic.
#   - "Other / Mixed" (event_number=28) is reserved for articles that ARE
#     DXY-relevant but don't fit any named category.
#   - Everything flows through the same output schema — no branching on is_relevant.
# ==============================================================================

def agent_1_classify(client, title, date, content):
    """
    Merged relevance + classification agent.

    Returns the single best-matching event category for this article.
    Two special categories handle the relevance decision:

      event_number=0,  event_name="Irrelevant"
        → Article has no bearing on DXY or FX rates. Skip Agent 2 (criticality).
        → Use for: pure opinion/commentary on already-known policy, corporate
          earnings with no macro signal, crypto/commodities with no dollar link,
          technical chart reviews, lifestyle/general news.

      event_number=28, event_name="Other / Mixed"
        → Article IS relevant to DXY but doesn't map to any named category.
        → Use for: FX market colour, dollar sentiment pieces with a concrete
          event anchor, sovereign credit news outside the named categories,
          multi-topic articles where no single category dominates.

    For all other categories (1–39 excluding 28), pick the SINGLE best match.

    Returns:
      {
        "event_number": int,          # 0 = Irrelevant, 1–39 otherwise
        "event_name": str,            # exact name from list, or "Irrelevant"
        "content_type": str,          # hard_news | opinion | analysis | recap | other
        "is_event_vs_opinion": bool,  # true = factual sourced event; false = opinion/analysis
        "vs_consensus": str,          # beat | miss | in-line | not-applicable
        "confidence": str             # high | medium | low
      }
    """

    prompt = f"""You are classifying a financial news article for DXY (US Dollar Index) analysis.

Your task: assign the article to the SINGLE best-matching category from the list below.

DXY basket for reference: EUR 57.6% | JPY 13.6% | GBP 11.9% | CAD 9.1% | SEK 4.2% | CHF 3.6%

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

════════════════════════════════════════════════════════════════
NAMED EVENT CATEGORIES
════════════════════════════════════════════════════════════════

## US MONETARY POLICY
1.  Fed Rate Hike (actual decision announced)
2.  Fed Rate Cut (actual decision announced)
3.  Hawkish Pivot / Surprise (guidance or statement tighter than market expected)
4.  Dovish Pivot / Surprise (guidance or statement looser than market expected)
5.  Fed QE / Balance Sheet Expansion
6.  Fed QT / Tapering Announcement

## US INFLATION DATA
7.  CPI/PCE Above Consensus
8.  CPI/PCE Below Consensus
9.  Inflation In-Line (only use if surrounding context makes it notably significant)

## US EMPLOYMENT DATA
10. NFP/Jobs Beat (above consensus)
11. NFP/Jobs Miss (below consensus)
12. Unemployment Rate Surprise

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
39. Other major central bank rate change (RBA, SNB, BoC, etc.)

════════════════════════════════════════════════════════════════
OUTPUT FIELDS
════════════════════════════════════════════════════════════════

event_number     : integer — 0 for Irrelevant, 28 for Other/Mixed, 1–39 for named events
event_name       : exact category name from the list above (e.g. "NFP/Jobs Beat", "Irrelevant")
content_type     : hard_news | opinion | analysis | recap | other
is_event_vs_opinion:
  true  = article reports a factual, sourced event (decision made, data released,
          official statement, or credibly sourced reporting on an imminent decision)
  false = article expresses opinion, analysis, forecast, or commentary
  Note: if event_number is 0 (Irrelevant), set this to false.
vs_consensus     : beat | miss | in-line | not-applicable
confidence       : high | medium | low — your confidence in the event_number assignment

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "event_number": <integer>,
  "event_name": "<exact category name>",
  "content_type": "<hard_news|opinion|analysis|recap|other>",
  "is_event_vs_opinion": <true|false>,
  "vs_consensus": "<beat|miss|in-line|not-applicable>",
  "confidence": "<high|medium|low>"
}}"""

    try:
        result = call_gemini(client, prompt)
        # Normalise: Irrelevant must always be opinion=false
        if result.get("event_number") == IRRELEVANT_EVENT_NUMBER:
            result["is_event_vs_opinion"] = False
        return result
    except RateLimitExceeded:
        raise  # propagate immediately — caller will flush CSV and exit
    except Exception as e:
        print(f"  ⚠️  Agent 1 (Classifier) Error: {e}")
        return {
            "event_number": IRRELEVANT_EVENT_NUMBER,
            "event_name": "Irrelevant",
            "content_type": "other",
            "is_event_vs_opinion": False,
            "vs_consensus": "not-applicable",
            "confidence": "low",
        }


# ==============================================================================
# AGENT 2: CRITICALITY ASSESSOR  (was Agent 3)
#
# Tier is a STARTING POINT (prior), not a hard ceiling.
# The LLM can escalate any tier upward if the article signals an unusually
# large or market-disruptive event. Downgrade is also allowed for thin/vague articles.
# Only called when event_number != 0 (i.e. article is not Irrelevant).
# Does NOT generate pip estimates or surprise factors.
# ==============================================================================

def agent_2_criticality(client, title, date, content, event_info):
    """
    Determines whether this article is actionable for an FX analyst.

    Tier sets the DEFAULT criticality — but the LLM can override in either
    direction based on what the article actually describes:

    Default by tier (when article is factual + concrete):
      Tier 1 → high
      Tier 2 → medium
      Tier 3 → low   (but can escalate to medium or high if event is large enough)
      Tier 4 → low   (but can escalate if article signals genuine market disruption)

    Hard floors (cannot be escalated regardless of content):
      - Opinion/analysis/forecast → always low, never critical
      - Vague/lacking detail with no concrete signal → drop one level from default

    Escalation triggers (Tier 3/4 events that warrant upgrading):
      - Explicit emergency language: "emergency meeting", "unscheduled", "surprise cut/hike"
      - Described as historic, unprecedented, or record-breaking in scope
      - Concrete large numbers that suggest macro-scale disruption
        (e.g. tariffs raised to 50%+, $500B+ fiscal package, bank with $1T+ assets failing)
      - Language indicating immediate market reaction: "dollar plunged/surged X%",
        "yields spiked X bps", "circuit breakers triggered"
      - Geopolitical events explicitly tied to USD safe-haven flows or sanctions
        that could impair dollar liquidity
      - Fed Chair or Treasury Secretary directly contradicting current policy in real-time

    Also extracts any specific data value referenced in the article for analyst
    cross-reference (NOT used to assess surprise — that requires consensus data).

    Returns: {"is_critical": bool, "criticality_level": str,
              "tier_override": bool, "override_reason": str | null,
              "data_point_referenced": bool, "reported_value": str | null}
    """

    event_number        = event_info.get("event_number", OTHER_EVENT_NUMBER)
    event_name          = event_info.get("event_name", "Other / Mixed")
    is_event_vs_opinion = event_info.get("is_event_vs_opinion", False)
    tier                = EVENT_TIERS.get(event_number, 4)
    tier_label          = TIER_LABELS.get(tier)

    # Compute the default criticality so the model has an explicit anchor
    if not is_event_vs_opinion:
        default_level = "low"
    elif tier == 1:
        default_level = "high"
    elif tier == 2:
        default_level = "medium"
    else:
        default_level = "low"

    prompt = f"""You are an FX desk analyst reviewing a news article.
The article has been pre-classified:

  Event type           : {event_name} (Category #{event_number})
  Research-based tier  : {tier_label}
  Is a factual event   : {is_event_vs_opinion}
  Default criticality  : {default_level}  ← starting point based on event tier

The DEFAULT criticality above is your starting point. Your job is to CONFIRM it
or OVERRIDE it based on what the article actually describes.

── HARD FLOORS (cannot be overridden upward) ──────────────────────────────────
• If is_event_vs_opinion is FALSE (opinion/analysis/forecast/commentary):
  → always low, never critical, no override possible.
• If article is a recap of an event already fully priced in by markets:
  → downgrade to low.

── DOWNGRADE TRIGGERS (lower the default one level) ──────────────────────────
• Article is vague, speculative, or lacks concrete details to act on.
• Event is described as expected/anticipated with no new information.

── ESCALATION TRIGGERS (raise the default one or two levels) ─────────────────
These signals warrant escalating a Tier 3 or Tier 4 event to medium or high:
• Emergency/unscheduled nature: "emergency meeting", "unscheduled rate decision",
  "surprise cut/hike outside of scheduled meetings"
• Historic or unprecedented scope described explicitly in the article
• Large concrete numbers suggesting macro-scale disruption:
  - Tariffs raised to 50%+ on major trading partners
  - Fiscal package of $500B+ or more
  - Major bank failure ($500B+ assets)
  - Currency move of 2%+ in a single session described in article
• Immediate market reaction described: "dollar plunged/surged X%", "yields spiked
  X bps", "equities fell X%", "circuit breakers triggered"
• USD safe-haven demand explicitly described as materialising (not predicted)
• Direct contradiction of current Fed/Treasury policy by a senior official in real-time
• Sanctions or asset freezes that directly impair dollar liquidity or reserve status

Apply judgment: a large tariff escalation or an unexpected central bank emergency
action is genuinely high-impact even though their event type defaults to Tier 3.

── OUTPUT FIELDS ──────────────────────────────────────────────────────────────
is_critical    : true if criticality_level is "high" or "medium", false if "low"
criticality_level : your final assessed level — "high", "medium", or "low"
tier_override  : true if you are changing from the default ({default_level}), false if confirming it
override_reason: one sentence explaining WHY you overrode, or null if no override
data_point_referenced: true if article states a specific numeric value
  (e.g. "CPI at 3.2%", "NFP added 180k", "Fed cut 25bps", "tariffs raised to 25%")
reported_value : the specific value (e.g. "3.2%", "180k", "25bps cut"), or null if none

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "is_critical": true or false,
  "criticality_level": "high/medium/low",
  "tier_override": true or false,
  "override_reason": "one sentence or null",
  "data_point_referenced": true or false,
  "reported_value": "specific value string or null"
}}"""

    try:
        result = call_gemini(client, prompt)
        # Enforce hard floor in code: opinion/analysis can never be critical
        if not is_event_vs_opinion:
            result["is_critical"]       = False
            result["criticality_level"] = "low"
            result["tier_override"]     = False
            result["override_reason"]   = None
        return result
    except RateLimitExceeded:
        raise  # propagate immediately — caller will flush CSV and exit
    except Exception as e:
        print(f"  ⚠️  Agent 2 (Criticality) Error: {e}")
        return {
            "is_critical": False, "criticality_level": "low",
            "tier_override": False, "override_reason": None,
            "data_point_referenced": False, "reported_value": None,
        }


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def analyze_article_dxy(client, title, date, content, verbose=True):
    """
    Runs the 2-agent pipeline on a single article.

    Agent 1 (Classifier) → always runs; returns event category or "Irrelevant"
    Agent 2 (Criticality) → skipped when event_number == 0 (Irrelevant)

    Always returns a result dict — irrelevant articles have a3=None.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:75]}")
        print(f"{'='*60}")

    # --- Agent 1: Classify (merged relevance + event type) ---
    if verbose: print("\n🏷️  Agent 1: Classification...")
    a1 = agent_1_classify(client, title, date, content)

    is_irrelevant = (a1.get("event_number") == IRRELEVANT_EVENT_NUMBER)
    tier = EVENT_TIERS.get(a1.get("event_number", OTHER_EVENT_NUMBER), 4)

    if verbose:
        if is_irrelevant:
            print(f"⬜ IRRELEVANT | content_type={a1.get('content_type')}")
        else:
            print(f"📋 #{a1.get('event_number')}: {a1.get('event_name')}")
            print(f"   content_type        : {a1.get('content_type')}")
            print(f"   is_event_vs_opinion : {a1.get('is_event_vs_opinion')}")
            print(f"   vs_consensus        : {a1.get('vs_consensus')}")
            print(f"   confidence          : {a1.get('confidence')}")
            print(f"   tier (lookup)       : {TIER_LABELS.get(tier)}")

    if is_irrelevant:
        return {"relevant": False, "agent1": a1, "agent2": None}

    # --- Agent 2: Criticality ---
    if verbose: print("\n⚡ Agent 2: Criticality assessment...")
    a2 = agent_2_criticality(client, title, date, content, a1)

    if verbose:
        icon = "🔴" if a2.get("is_critical") else "🟡"
        print(f"{icon} is_critical={a2.get('is_critical')} | level={a2.get('criticality_level')}")
        if a2.get("tier_override"):
            print(f"   ⚠️  Tier override : {a2.get('override_reason')}")
        if a2.get("data_point_referenced"):
            print(f"   Reported value  : {a2.get('reported_value')}")

    return {"relevant": True, "agent1": a1, "agent2": a2}


# ==============================================================================
# CSV PROCESSING PIPELINE
# ==============================================================================

def process_csv_to_csv(
    input_csv_path,
    output_csv_path,
    test_row_index=None,
    verbose=True,
):
    """
    Read articles from CSV, run 2-agent pipeline, write full results CSV.
    ALL rows are output — irrelevant articles included with blank criticality
    columns so nothing is silently dropped.

    Args:
        input_csv_path  : Path to input CSV file.
        output_csv_path : Path for output CSV file.
        test_row_index  : Integer (0-based) row to test. None = process all rows.
                          Examples: 0 = first row, 5 = sixth row, 99 = 100th row
        verbose         : Print step-by-step agent output.
    """

    client = get_client()

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

    results        = []
    relevant_count = 0
    critical_count = 0
    rate_limit_hit = False

    def _flush_csv(rows, path, df_len):
        """Write whatever results have been collected so far and print summary."""
        if not rows:
            print("⚠️  No rows to write — rate limit hit before any article was processed.")
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        out.to_csv(path, index=False)
        processed = len(rows)
        skipped   = df_len - processed
        print(f"\n{'='*60}")
        print(f"💾 Saved → {path}")
        print(f"   Rows written : {processed} of {df_len}")
        if skipped:
            print(f"   ⚠️  {skipped} rows NOT processed (rate limit hit mid-run)")
        return out

    try:
        for idx, row in df.iterrows():

            title      = str(row.get("title", ""))
            date       = str(row.get("published", row.get("article_date", "")))
            article_id = row.get("id", idx)

            # Use full_text column; fall back to title if empty/missing
            raw_full_text = row.get("full_text", "")
            content = (
                str(raw_full_text).strip()
                if pd.notna(raw_full_text) and str(raw_full_text).strip()
                else title
            )

            analysis = analyze_article_dxy(client, title, date, content, verbose=verbose)

            a1 = analysis["agent1"]
            a2 = analysis["agent2"]   # None when irrelevant

            event_num = a1.get("event_number", IRRELEVANT_EVENT_NUMBER)
            tier      = EVENT_TIERS.get(event_num, 4) if event_num != IRRELEVANT_EVENT_NUMBER else None

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
                # ---- agent 1: classification ----
                "is_relevant":               analysis["relevant"],   # derived: event != Irrelevant
                "event_number":              a1.get("event_number"),
                "event_name":                a1.get("event_name"),
                "content_type":              a1.get("content_type"),
                "is_event_vs_opinion":       a1.get("is_event_vs_opinion"),
                "vs_consensus":              a1.get("vs_consensus"),
                "classification_confidence": a1.get("confidence"),
                "event_tier":                tier,
                "event_tier_label":          TIER_LABELS.get(tier) if tier else None,
                # ---- agent 2: criticality (blank for irrelevant articles) ----
                "is_critical":           a2.get("is_critical")           if a2 else None,
                "criticality_level":     a2.get("criticality_level")     if a2 else None,
                "tier_override":         a2.get("tier_override")         if a2 else None,
                "override_reason":       a2.get("override_reason")       if a2 else None,
                "data_point_referenced": a2.get("data_point_referenced") if a2 else None,
                "reported_value":        a2.get("reported_value")        if a2 else None,
                # ---- metadata ----
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            results.append(out_row)

            if analysis["relevant"]:
                relevant_count += 1
            if a2 and a2.get("is_critical"):
                critical_count += 1

            # Rate limiting: Gemini Flash-Lite free tier = 15 RPM
            # 2 agents = 2 calls per relevant row, 1 call for irrelevant rows
            # 5s inter-row pause keeps well within limits
            if test_row_index is None:
                time.sleep(5.0)

    except RateLimitExceeded as rle:
        rate_limit_hit = True
        print(f"\n🚫 RATE LIMIT HIT — stopping immediately.")
        print(f"   Error: {str(rle)[:200]}")
        print(f"   Flushing {len(results)} completed rows to CSV now...")

    # --- Write output (partial or complete) ---
    out_df = _flush_csv(results, output_csv_path, len(df))

    if out_df.empty:
        return out_df

    rows_done = len(results)
    print(f"\n📊 SUMMARY {'(PARTIAL — rate limit)' if rate_limit_hit else ''}")
    print(f"   Rows processed  : {rows_done} of {len(df)}")
    print(f"   Relevant         : {relevant_count}")
    print(f"   Irrelevant       : {rows_done - relevant_count}")
    print(f"   Critical         : {critical_count}")
    if relevant_count > 0:
        print(f"   Critical rate    : {critical_count / relevant_count * 100:.1f}% of relevant")

    if relevant_count > 0:
        rel = out_df[out_df["is_relevant"] == True]
        print(f"\n📈 Event distribution (relevant articles):")
        for event, count in rel["event_name"].value_counts().items():
            print(f"   {count:>3}x  {event}")
        print(f"\n🏷️  Tier distribution:")
        for label, count in rel["event_tier_label"].value_counts().items():
            print(f"   {count:>3}x  {label}")

    if rate_limit_hit:
        print(f"\n💡 TIP: Quota resets at midnight Pacific Time.")
        print(f"   Re-run with test_row_index={rows_done} to resume from where you left off,")
        print(f"   or concatenate the partial CSV with a second run.")

    return out_df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    output_df = process_csv_to_csv(
        input_csv_path="data/dxy_training/gemeni/cnbc_apr_2_4_full_text.csv",
        output_csv_path="data/dxy_training/claude/cnbc_apr2_4_results.csv",
        test_row_index=1,     # ← change to any row index you want to test
                                 #   0 = first row, 5 = sixth row, etc.
                                 #   set to None to process ALL rows
        verbose=True,
    )
