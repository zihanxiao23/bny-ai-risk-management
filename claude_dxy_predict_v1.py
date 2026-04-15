import os
import anthropic
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv

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


MODEL = "claude-3-haiku-20240307"  # Cheapest: $0.25/$1.25 per M tokens


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

# Canonical name for each event_number — used to derive event_name post-hoc via pandas map()
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


# ==============================================================================
# SHARED HELPER: call Claude and parse JSON response
# ==============================================================================

def call_claude(client, prompt: str) -> dict:
    """
    Send a prompt to Claude and return parsed JSON.
    Uses the messages API with the prompt as a user message.
    All API errors are re-raised for the caller to handle.
    """
    message = client.messages.create(
        model=MODEL,
        max_tokens=128,
        temperature=0.1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = message.content[0].text.strip()

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
    Classify article into the DXY event taxonomy.

    Returns only the fields the model needs to produce:
      event_number : 0 = Irrelevant, 28 = Other/Mixed, 1–39 = named event
      content_type : hard_news | opinion | analysis | recap | other
      confidence   : high | medium | low

    event_name, event_tier, event_tier_label, and is_relevant are derived
    from event_number via lookup tables after the run — no LLM cost.
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

event_number : integer — 0 for Irrelevant, 28 for Other/Mixed, 1–39 for named events
content_type : hard_news | opinion | analysis | recap | other
confidence   : high | medium | low — your confidence in the event_number assignment

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "event_number": <integer>,
  "content_type": "<hard_news|opinion|analysis|recap|other>",
  "confidence": "<high|medium|low>"
}}"""

    try:
        return call_claude(client, prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 1 (Classifier) Error: {e}")
        return {
            "event_number": IRRELEVANT_EVENT_NUMBER,
            "content_type": "other",
            "confidence":   "low",
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

def agent_2_criticality(client, title, date, content, event_number, content_type):
    """
    Assess whether this article is actionable for an FX analyst.

    Receives event_number and content_type directly (no event_info dict).
    Returns only criticality_level — is_critical is derived post-hoc via pandas.

    Default by tier (factual content only):
      Tier 1 → high  |  Tier 2 → medium  |  Tier 3/4 → low
    Hard floor: opinion / analysis → always low, no escalation possible.

    Escalation triggers (Tier 3/4 → medium or high):
      - "emergency meeting", "unscheduled", "surprise cut/hike"
      - Historic or unprecedented scope explicitly described
      - Tariffs 50%+, fiscal $500B+, bank failure $500B+ assets
      - "dollar plunged/surged X%", "yields spiked X bps", circuit breakers
      - USD safe-haven demand described as materialising (not predicted)
      - Senior official directly contradicting current Fed/Treasury policy
      - Sanctions / asset freezes impairing dollar liquidity
    """

    tier       = EVENT_TIERS.get(event_number, 4)
    tier_label = TIER_LABELS.get(tier)
    event_name = EVENT_NAMES.get(event_number, "Other / Mixed")
    is_opinion = content_type in ("opinion", "analysis")

    if is_opinion:
        default_level = "low"
    elif tier == 1:
        default_level = "high"
    elif tier == 2:
        default_level = "medium"
    else:
        default_level = "low"

    prompt = f"""You are an FX desk analyst assessing whether a news article is actionable.
The article has been pre-classified:

  Event type          : {event_name} (Category #{event_number})
  Research-based tier : {tier_label}
  Content type        : {content_type}
  Default criticality : {default_level}  ← starting point based on event tier

Your job: CONFIRM or OVERRIDE the default based on what the article actually describes.

── HARD FLOOR ─────────────────────────────────────────────────────────────────
• content_type is "opinion" or "analysis" → always low, no escalation possible.
• Recap of an event already fully priced in → downgrade to low.

── DOWNGRADE TRIGGERS ─────────────────────────────────────────────────────────
• Article is vague or lacks concrete details.
• Event described as expected/anticipated with no new information.

── ESCALATION TRIGGERS (Tier 3/4 only) ───────────────────────────────────────
• "emergency meeting", "unscheduled rate decision", "surprise cut/hike"
• Explicitly described as historic or unprecedented in scope
• Tariffs raised to 50%+ | Fiscal package $500B+ | Bank failure $500B+ assets
• "dollar plunged/surged X%", "yields spiked X bps", "circuit breakers triggered"
• USD safe-haven demand described as materialising (not merely predicted)
• Senior Fed/Treasury official directly contradicting current policy in real-time
• Sanctions or asset freezes impairing dollar liquidity or reserve status

Apply judgment: a large tariff escalation or unexpected central bank emergency
action is genuinely high-impact even though their event type defaults to Tier 3.

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "criticality_level": "<high|medium|low>"
}}"""

    try:
        result = call_claude(client, prompt)
        if is_opinion:
            result["criticality_level"] = "low"
        return result
    except Exception as e:
        print(f"  ⚠️  Agent 2 (Criticality) Error: {e}")
        return {"criticality_level": "low"}


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def analyze_article_dxy(client, title, date, content, verbose=True):
    """
    Runs the 2-agent pipeline on a single article.

    Agent 1 (Classifier)  → always runs; returns event_number + content_type
    Agent 2 (Criticality) → skipped when event_number == 0 (Irrelevant)
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:75]}")
        print(f"{'='*60}")

    # --- Agent 1 ---
    if verbose: print("\n🏷️  Agent 1: Classification...")
    a1 = agent_1_classify(client, title, date, content)

    event_number  = a1.get("event_number", IRRELEVANT_EVENT_NUMBER)
    is_irrelevant = (event_number == IRRELEVANT_EVENT_NUMBER)
    tier          = EVENT_TIERS.get(event_number, 4)

    if verbose:
        if is_irrelevant:
            print(f"⬜ IRRELEVANT | content_type={a1.get('content_type')}")
        else:
            print(f"📋 #{event_number}: {EVENT_NAMES.get(event_number)}")
            print(f"   content_type : {a1.get('content_type')}")
            print(f"   confidence   : {a1.get('confidence')}")
            print(f"   tier         : {TIER_LABELS.get(tier)}")

    if is_irrelevant:
        return {"agent1": a1, "agent2": None}

    # --- Agent 2 ---
    if verbose: print("\n⚡ Agent 2: Criticality assessment...")
    a2 = agent_2_criticality(
        client, title, date, content,
        event_number=event_number,
        content_type=a1.get("content_type", "other"),
    )

    if verbose:
        level = a2.get("criticality_level", "low")
        icon  = "🔴" if level == "high" else ("🟡" if level == "medium" else "⚪")
        print(f"{icon} criticality_level={level}")

    return {"agent1": a1, "agent2": a2}


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
    Read articles from CSV, run 2-agent pipeline, write results CSV.
    ALL rows are output — irrelevant articles included with blank agent 2 columns.

    Columns added via pandas after the run (zero LLM cost):
      event_name       — looked up from EVENT_NAMES by event_number
      event_tier_label — looked up from TIER_LABELS by event_tier
      is_relevant      — True where event_number != 0
      is_critical      — True where criticality_level in ("high", "medium")
      processed_at     — single timestamp for the whole batch

    Args:
        input_csv_path  : Path to input CSV.
        output_csv_path : Path for output CSV.
        test_row_index  : 0-based row index to test single article; None = all rows.
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

    for idx, row in df.iterrows():

        title      = str(row.get("title", ""))
        date       = str(row.get("published", row.get("article_date", "")))
        article_id = row.get("id", idx)

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
            # ---- agent 1 raw output ----
            "event_number":              event_num,
            "content_type":              a1.get("content_type"),
            "classification_confidence": a1.get("confidence"),
            "event_tier":                tier,
            # ---- agent 2 raw output (None for irrelevant) ----
            "criticality_level": a2.get("criticality_level") if a2 else None,
        }

        results.append(out_row)

    # --- Build DataFrame ---
    out_df = pd.DataFrame(results)

    # --- Derive columns via pandas (no LLM cost) ---
    out_df["event_name"]       = out_df["event_number"].map(EVENT_NAMES)
    out_df["event_tier_label"] = out_df["event_tier"].map(TIER_LABELS)
    out_df["is_relevant"]      = out_df["event_number"] != IRRELEVANT_EVENT_NUMBER
    out_df["is_critical"]      = out_df["criticality_level"].isin(["high", "medium"])
    out_df["processed_at"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_df.to_csv(output_csv_path, index=False)

    # --- Summary ---
    relevant_count = out_df["is_relevant"].sum()
    critical_count = out_df["is_critical"].sum()
    rows_done      = len(out_df)

    print(f"\n{'='*60}")
    print(f"💾 Saved → {output_csv_path}")
    print(f"\n📊 SUMMARY")
    print(f"   Model           : {MODEL}")
    print(f"   Rows processed  : {rows_done}")
    print(f"   Relevant        : {relevant_count}")
    print(f"   Irrelevant      : {rows_done - relevant_count}")
    print(f"   Critical        : {critical_count}")
    if relevant_count > 0:
        print(f"   Critical rate   : {critical_count / relevant_count * 100:.1f}% of relevant")
        rel = out_df[out_df["is_relevant"]]
        print(f"\n📈 Event distribution (relevant articles):")
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
        input_csv_path="data/dxy_training/gemeni/cnbc_apr_2_4_full_text.csv",
        output_csv_path="data/dxy_training/claude/cnbc_apr2_4_results.csv",
        test_row_index=None,     # ← change to any row index you want to test
                              #   0 = first row, 5 = sixth row, etc.
                              #   set to None to process ALL rows
        verbose=True,
    )
