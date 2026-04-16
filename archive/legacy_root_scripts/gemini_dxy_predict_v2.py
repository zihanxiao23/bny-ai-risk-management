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

MODEL = "gemini-2.5-flash-lite"  # Free tier: 15 RPM, 1,000 RPD, 250k TPM


# ==============================================================================
# TIER LOOKUP TABLE
# Based on DXY driver hierarchy research:
#   Fed Board IFDP (2017, 2021, 2024), BIS WP 695, J.P. Morgan AM (2025),
#   ICE USDX White Paper, NBER WP 32329, CEPR/Eichengreen (2023)
#
# Tier 1 — Primary Structural: Fed policy, rate differentials, core macro data
#           Largest magnitude, longest-lasting DXY impact
# Tier 2 — Secondary Structural: risk appetite, capital flows, fiscal policy
#           Medium-term, recurring influence
# Tier 3 — Geopolitical & Sentiment: trade/tariff, CB divergence, institutional risk
#           Event-driven, acute but shorter-lived volatility
# Tier 4 — Commodity & Technical: oil/gold prices, positioning, PPP valuation
#           Shorter-term, reflexive, lower structural importance
# ==============================================================================

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
    9:  2,   # Inflation In-Line (priced in, lower structural impact → Tier 2)
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

    # Tier 4: Ambiguous / low structural importance
    28: 4,   # Other / Mixed
}

TIER_LABELS = {
    1: "Tier 1 — Primary Structural (Fed/Core Macro)",
    2: "Tier 2 — Secondary Structural (Risk/Flows/Fiscal)",
    3: "Tier 3 — Geopolitical & Sentiment",
    4: "Tier 4 — Commodity/Technical or Ambiguous",
}


# ==============================================================================
# SHARED HELPER: call Gemini and parse JSON response
# ==============================================================================

def call_gemini(client, prompt: str) -> dict:
    """Send a prompt to Gemini Flash-Lite and return parsed JSON."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 256,
        }
    )

    text = response.text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```json")[-1].split("```")[0].strip()

    return json.loads(text)


# ==============================================================================
# AGENT 1: RELEVANCE FILTER
# Simplified: binary yes/no + content_type. No surprise factor, no reasoning.
# ==============================================================================

def agent_1_relevance_filter(client, title, date, content):
    """
    Binary filter: does this article describe a market-moving event for DXY/FX?
    Returns: {"is_relevant": bool, "content_type": str}
    content_type: "hard_news" | "opinion" | "analysis" | "recap" | "other"
    """

    prompt = f"""You are screening financial news. Decide: does this article describe a
concrete event or policy development that could move DXY (US Dollar Index) or FX rates?

DXY basket: EUR 57.6% | JPY 13.6% | GBP 11.9% | CAD 9.1% | SEK 4.2% | CHF 3.6%

PASS (is_relevant: true) — hard news that shifts market expectations:
- Fed decisions, FOMC statements, or central bank rate changes (ECB, BOJ, BOE, etc.)
- US macro data releases: CPI, PCE, NFP, GDP, retail sales, ISM/PMI
- Trade/tariff policy announcements or reversals with concrete details
- Debt ceiling, fiscal crisis, sovereign credit events, major stimulus
- Geopolitical shocks with clear USD safe-haven or risk-off implications
- Congressional or executive actions affecting fiscal/monetary/trade policy
- Banking or financial system stress events
- Sourced reporting on upcoming decisions ("sources say...", "officials considering...")

FAIL (is_relevant: false) — noise:
- Analyst opinions, forecasts, or price targets with no new factual event
- Commentary or editorials on existing known policy
- Technical analysis or chart reviews
- Recaps of events already happened and fully priced in
- Corporate earnings (unless clearly signalling a macro inflection)
- Pure speculation with no sourced basis

content_type must be one of: hard_news | opinion | analysis | recap | other

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "is_relevant": true or false,
  "content_type": "hard_news/opinion/analysis/recap/other"
}}"""

    try:
        return call_gemini(client, prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 1 Error: {e}")
        return {"is_relevant": False, "content_type": "other"}


# ==============================================================================
# AGENT 2: EVENT CLASSIFIER
# Outputs event_type + is_event_vs_opinion flag
# ==============================================================================

def agent_2_event_classifier(client, title, date, content):
    """
    Classifies the article into an event category and flags event vs. opinion.
    Returns: {"event_number": int, "event_name": str, "vs_consensus": str,
              "is_event_vs_opinion": bool, "confidence": str}

    is_event_vs_opinion:
      true  = article reports a factual, sourced event (decision made, data released,
              official statement, credibly sourced reporting)
      false = article expresses opinion, analysis, forecast, or commentary
    """

    prompt = f"""Classify this article into the SINGLE best-matching DXY event category.
Also flag whether the article reports a factual event or expresses opinion/analysis.

## US MONETARY POLICY
1.  Fed Rate Hike (actual decision)
2.  Fed Rate Cut (actual decision)
3.  Hawkish Pivot / Surprise (guidance tighter than expected)
4.  Dovish Pivot / Surprise (guidance looser than expected)
5.  Fed QE / Balance Sheet Expansion
6.  Fed QT / Tapering Announcement

## US INFLATION DATA
7.  CPI/PCE Above Consensus
8.  CPI/PCE Below Consensus
9.  Inflation In-Line (only classify if other context makes it notable)

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
19. Trade Tariff / Policy Shock
20. Trade Balance Surprise
21. Debt Ceiling / Fiscal Crisis
22. Fiscal Stimulus / Spending Package

## MARKET STRUCTURE EVENTS
23. US Banking / Financial System Stress
24. Risk-Off Shock (flight to USD safety)
25. Treasury Yield Spike / Bond Market Stress
26. US Political Shock (election, impeachment, major policy reversal)
27. Foreign Central Bank Shock (ECB/BOJ/BOE surprise affecting rate differential)
28. Other / Mixed

## POLICY EXPECTATION SHIFTS
29. Trade Policy Reversal Signal (tariff removal/reduction becoming likely)
30. Trade Policy Escalation Signal (new tariffs or expansion becoming likely)
31. Fed Rate Path Repricing (market odds of cuts/hikes shifting without a decision)
32. Fiscal Policy Probability Shift (stimulus/austerity becoming more/less likely)
33. Legal/Judicial Event Affecting Economic Policy
34. Legislative Progress/Blockage (bill with fiscal consequence)
35. Executive Policy Signal (White House trial balloon, leaked plan, EO signal)

## FOREIGN CENTRAL BANK RATE CHANGES
36. Rate hike or cut from ECB
37. Rate hike or cut from BOJ
38. Rate hike or cut from BOE
39. Rate hike or cut from other major central bank (RBA, SNB, etc.)

is_event_vs_opinion:
  true  = article reports a factual, sourced event
  false = article expresses opinion, analysis, forecast, or commentary

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "event_number": <integer 1-39>,
  "event_name": "Exact name from list above",
  "vs_consensus": "beat/miss/in-line/not-applicable",
  "is_event_vs_opinion": true or false,
  "confidence": "high/medium/low"
}}"""

    try:
        return call_gemini(client, prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 2 Error: {e}")
        return {"event_number": 28, "event_name": "Other / Mixed",
                "vs_consensus": "not-applicable", "is_event_vs_opinion": False,
                "confidence": "low"}


# ==============================================================================
# AGENT 3: CRITICALITY ASSESSOR
# Tier is a STARTING POINT (prior), not a hard ceiling.
# The LLM can escalate any tier upward if the article signals an unusually
# large or market-disruptive event. Downgrade is also allowed for thin/vague articles.
# Does NOT generate pip estimates or surprise factors.
# ==============================================================================

def agent_3_criticality_assessor(client, title, date, content, event_info):
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

    event_number        = event_info.get("event_number", 28)
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
        # Enforce hard floor: opinion/analysis can never be critical
        if not is_event_vs_opinion:
            result["is_critical"]       = False
            result["criticality_level"] = "low"
            result["tier_override"]     = False
            result["override_reason"]   = None
        return result
    except Exception as e:
        print(f"  ⚠️  Agent 3 Error: {e}")
        return {"is_critical": False, "criticality_level": "low",
                "tier_override": False, "override_reason": None,
                "data_point_referenced": False, "reported_value": None}


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def analyze_article_dxy(client, title, date, content, verbose=True):
    """
    Runs the 3-agent funnel on a single article.
    Always returns a result dict — irrelevant articles get None for agents 2 & 3.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:75]}")
        print(f"{'='*60}")

    # --- Agent 1: Relevance ---
    if verbose: print("\n🔍 Agent 1: Relevance filter...")
    a1 = agent_1_relevance_filter(client, title, date, content)

    if verbose:
        icon = "✅" if a1.get("is_relevant") else "❌"
        print(f"{icon} is_relevant={a1.get('is_relevant')} | content_type={a1.get('content_type')}")

    if not a1.get("is_relevant", False):
        return {"relevant": False, "agent1": a1, "agent2": None, "agent3": None}

    # --- Agent 2: Classification ---
    if verbose: print("\n🏷️  Agent 2: Event classification...")
    a2 = agent_2_event_classifier(client, title, date, content)

    tier = EVENT_TIERS.get(a2.get("event_number", 28), 4)
    if verbose:
        print(f"📋 #{a2.get('event_number')}: {a2.get('event_name')}")
        print(f"   is_event_vs_opinion : {a2.get('is_event_vs_opinion')}")
        print(f"   vs_consensus        : {a2.get('vs_consensus')}")
        print(f"   confidence          : {a2.get('confidence')}")
        print(f"   tier (lookup)       : {TIER_LABELS.get(tier)}")

    # --- Agent 3: Criticality ---
    if verbose: print("\n⚡ Agent 3: Criticality assessment...")
    a3 = agent_3_criticality_assessor(client, title, date, content, a2)

    if verbose:
        icon = "🔴" if a3.get("is_critical") else "🟡"
        print(f"{icon} is_critical={a3.get("is_critical")} | level={a3.get("criticality_level")}")
        if a3.get("tier_override"):
            print(f"   ⚠️  Tier override : {a3.get("override_reason")}")
        if a3.get("data_point_referenced"):
            print(f"   Reported value  : {a3.get("reported_value")}")

    return {"relevant": True, "agent1": a1, "agent2": a2, "agent3": a3}


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
    Read articles from CSV, run 3-agent pipeline, write full results CSV.
    ALL rows are output — irrelevant articles included with blank classification
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

    results = []
    relevant_count = 0
    critical_count = 0

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
        a2 = analysis["agent2"]
        a3 = analysis["agent3"]

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
            # ---- agent 1 outputs ----
            "is_relevant":  analysis["relevant"],
            "content_type": a1.get("content_type") if a1 else None,
            # ---- agent 2 outputs (blank for irrelevant articles) ----
            "event_number":              a2.get("event_number")         if a2 else None,
            "event_name":                a2.get("event_name")           if a2 else None,
            "event_tier":                EVENT_TIERS.get(a2.get("event_number", 28), 4) if a2 else None,
            "event_tier_label":          TIER_LABELS.get(EVENT_TIERS.get(a2.get("event_number", 28), 4)) if a2 else None,
            "vs_consensus":              a2.get("vs_consensus")         if a2 else None,
            "is_event_vs_opinion":       a2.get("is_event_vs_opinion")  if a2 else None,
            "classification_confidence": a2.get("confidence")           if a2 else None,
            # ---- agent 3 outputs (blank for irrelevant articles) ----
            "is_critical":           a3.get("is_critical")           if a3 else None,
            "criticality_level":     a3.get("criticality_level")     if a3 else None,
            "tier_override":         a3.get("tier_override")         if a3 else None,
            "override_reason":       a3.get("override_reason")       if a3 else None,
            "data_point_referenced": a3.get("data_point_referenced") if a3 else None,
            "reported_value":        a3.get("reported_value")        if a3 else None,
            # ---- metadata ----
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        results.append(out_row)

        if analysis["relevant"]:
            relevant_count += 1
        if a3 and a3.get("is_critical"):
            critical_count += 1

        # Rate limiting: Gemini Flash-Lite free tier = 15 RPM
        # 3 agents = 3 calls per row; 5s inter-row pause keeps well within limits
        if test_row_index is None:
            time.sleep(5.0)

    # --- Write output ---
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"💾 Saved → {output_csv_path}")
    print(f"\n📊 SUMMARY")
    print(f"   Rows processed  : {len(df)}")
    print(f"   Relevant         : {relevant_count}")
    print(f"   Not relevant     : {len(df) - relevant_count}")
    print(f"   Critical         : {critical_count}")
    if relevant_count > 0:
        print(f"   Critical rate    : {critical_count / relevant_count * 100:.1f}% of relevant")

    if relevant_count > 0:
        rel = out_df[out_df["is_relevant"] == True]
        print(f"\n📈 Event distribution:")
        for event, count in rel["event_name"].value_counts().items():
            print(f"   {count:>3}x  {event}")
        print(f"\n🏷️  Tier distribution:")
        for label, count in rel["event_tier_label"].value_counts().items():
            print(f"   {count:>3}x  {label}")

    return out_df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    output_df = process_csv_to_csv(
        input_csv_path="data/dxy_training/cnbc_apr_2_4_full_text.csv",
        output_csv_path="data/dxy_training/cnbc_apr2_4_gemini_results.csv",
        test_row_index=None,     # ← change to any row index you want to test
                              #   0 = first row, 5 = sixth row, etc.
                              #   set to None to process ALL rows
        verbose=True,
    )
