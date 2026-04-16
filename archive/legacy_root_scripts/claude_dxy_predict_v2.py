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


MODEL = "claude-haiku-4-5-20251001"


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


# ==============================================================================
# FEW-SHOT EXAMPLE LOADING
#
# Loads labeled ground truth articles (e.g. Apr 2–4) to inject into prompts
# as few-shot examples for all subsequent classifications.
#
# Expected ground truth CSV columns:
#   title               — article headline (join key if no 'id')
#   full_text           — article body (or 'content' as fallback)
#   gt_content_type     — one of: hard_news | opinion | analysis | recap | other
#   gt_event_name       — canonical name from EVENT_NAMES (e.g. "Trade Tariff / Policy Shock")
#   gt_criticality_level— one of: high | medium | low
#
# Sampling strategy: balanced across event types so the model sees diverse examples,
# not just whatever event happens to dominate the first rows.
# ==============================================================================

def load_few_shot_examples(ground_truth_csv_path, max_examples=12):
    """
    Load and sample labeled ground truth articles for use as few-shot examples.

    Samples up to 2 per unique event type to maximise taxonomy coverage,
    then caps at max_examples total. Shuffles so no event type dominates
    the top of the prompt.

    Returns a list of dicts ready for build_few_shot_block().
    """
    print(f"📂 Loading few-shot ground truth: {ground_truth_csv_path}")
    gt_df = pd.read_csv(ground_truth_csv_path)

    required = {"title", "gt_content_type", "gt_event_number", "gt_criticality_level"}
    missing = required - set(gt_df.columns)
    if missing:
        raise ValueError(
            f"Ground truth CSV is missing required columns: {missing}\n"
            f"Found columns: {list(gt_df.columns)}"
        )

    # Derive gt_event_name from gt_event_number via lookup table
    gt_df["gt_event_name"] = gt_df["gt_event_number"].map(EVENT_NAMES)
    unknown = gt_df["gt_event_name"].isna()
    if unknown.any():
        bad = gt_df.loc[unknown, "gt_event_number"].unique().tolist()
        raise ValueError(f"Ground truth contains unrecognised gt_event_number values: {bad}")

    # Balanced sample: up to 2 per event type, then cap at max_examples
    sampled = (
        gt_df.groupby("gt_event_number", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 2), random_state=42))
        .sample(frac=1, random_state=42)   # shuffle order
        .head(max_examples)
        .reset_index(drop=True)
    )

    # Resolve content column
    content_col = (
        "full_text" if "full_text" in gt_df.columns
        else "content" if "content" in gt_df.columns
        else None
    )

    examples = []
    for _, row in sampled.iterrows():
        if content_col:
            content_preview = str(row.get(content_col, ""))[:600].strip()
        else:
            content_preview = ""   # fall back to title only

        examples.append({
            "title":             str(row["title"]),
            "content_preview":   content_preview,
            "content_type":      str(row["gt_content_type"]),
            "event_name":        str(row["gt_event_name"]),   # resolved from number
            "criticality_level": str(row["gt_criticality_level"]),
        })

    print(f"✅ {len(examples)} few-shot examples loaded across ")
        #   f"{sampled['gt_event_number'].nunique()} event types")
    return examples


def build_few_shot_block(examples):
    """
    Render few-shot examples into a stable string for prompt injection.

    IMPORTANT FOR CACHING: this string must be byte-for-byte identical on every
    call within a run. Build it once in process_csv_to_csv() and reuse the same
    object — never regenerate it per article.
    """
    if not examples:
        return ""

    lines = [
        "════════════════════════════════════════════════════════════════",
        "FEW-SHOT EXAMPLES — real articles labeled by a human analyst",
        "Use these to calibrate your output format and judgment.",
        "════════════════════════════════════════════════════════════════\n",
    ]

    for i, ex in enumerate(examples, 1):
        lines.append(f"── Example {i} ──────────────────────────────────────────────")
        lines.append(f"Title   : {ex['title']}")
        if ex["content_preview"]:
            lines.append(f"Content : {ex['content_preview']}...")
        lines.append(f"LABELS:")
        lines.append(f"  content_type      → {ex['content_type']}")
        lines.append(f"  event_name        → {ex['event_name']}")
        lines.append(f"  criticality_level → {ex['criticality_level']}\n")

    lines.append("════════════════════════════════════════════════════════════════")
    lines.append("Now classify the new article below using the same judgment.\n")

    return "\n".join(lines)


# ==============================================================================
# SHARED HELPER: call Claude with optional cached system prompt
#
# The system_prompt parameter receives the static taxonomy + few-shot block.
# Anthropic's prompt caching stores this server-side after the first call,
# cutting subsequent reads by ~90% in token cost.
#
# Cache hit requires the system prompt to be byte-for-byte identical — this is
# why build_few_shot_block() is called once per run, not per article.
# ==============================================================================

def call_claude(client, user_prompt: str, system_prompt: str = "") -> dict:
    """
    Send a prompt to Claude and return parsed JSON.

    If system_prompt is provided it is sent with cache_control so Anthropic
    caches it server-side across calls. The user_prompt contains the
    per-article content that varies each call.
    """
    kwargs = {
        "model":      MODEL,
        "max_tokens": 128,
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
#
# Combines the static taxonomy (identical every run) with the few-shot block
# (built once from ground truth CSV). This entire string is what gets cached.
#
# Keeping taxonomy in the system prompt rather than the user prompt means:
#   1. It's cached and not re-billed on cache hits.
#   2. The model treats it as standing instructions, not per-article context.
# ==============================================================================

TAXONOMY_SYSTEM_PROMPT = """You are an expert FX analyst classifying financial news articles for DXY (US Dollar Index) impact analysis.

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
39. Other major central bank rate change (RBA, SNB, BoC, etc.)"""


def build_system_prompt(few_shot_block: str = "") -> str:
    """
    Combine the static taxonomy with the (optional) few-shot block.
    Called once per run; the result is passed to every call_claude() call.
    """
    if few_shot_block:
        return TAXONOMY_SYSTEM_PROMPT + "\n\n" + few_shot_block
    return TAXONOMY_SYSTEM_PROMPT


# ==============================================================================
# AGENT 1: EVENT CLASSIFIER
#
# Classifies article into the DXY event taxonomy.
# Taxonomy + few-shot examples live in the cached system prompt.
# Only the per-article content is in the user message.
# ==============================================================================

def agent_1_classify(client, title, date, content, system_prompt: str = ""):
    """
    Classify article into the DXY event taxonomy.

    Returns:
      event_number : 0 = Irrelevant, 28 = Other/Mixed, 1–39 = named event
      content_type : hard_news | opinion | analysis | recap | other
      confidence   : high | medium | low

    event_name, event_tier, event_tier_label, and is_relevant are derived
    from event_number via lookup tables — no LLM cost.
    """

    user_prompt = f"""Classify the article below using the taxonomy and examples in your instructions.

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "event_number": <integer>,
  "content_type": "<hard_news|opinion|analysis|recap|other>",
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
#
# Assesses whether this article is actionable for an FX analyst.
# Tier is a starting point (prior); the model can escalate or downgrade.
# Only called when event_number != 0 (not Irrelevant).
# ==============================================================================

CRITICALITY_SYSTEM_PROMPT = """You are an FX desk analyst assessing whether a news article is actionable for DXY trading.

── HARD FLOOR ─────────────────────────────────────────────────────────────────
• content_type is "opinion" or "analysis" → always low or null, no escalation possible.
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
action is genuinely high-impact even though their event type defaults to Tier 3."""


def agent_2_criticality(client, title, date, content, event_number, content_type,
                         system_prompt: str = ""):
    """
    Assess criticality of a relevant article.

    Default by tier (factual content only):
      Tier 1 → high  |  Tier 2 → medium  |  Tier 3/4 → low
    Hard floor: opinion / analysis → always low.
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

    user_prompt = f"""Assess the criticality of this article for DXY/FX trading.

Pre-classification:
  Event type          : {event_name} (Category #{event_number})
  Research-based tier : {tier_label}
  Content type        : {content_type}
  Default criticality : {default_level}  ← starting point; confirm or override

Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "criticality_level": "<high|medium|low>",
  "reasoning": "<1-2 sentences explaining why this event/criticality were chosen>",
  "direction": "<up|down, would this move dx-y index up or down?>"

}}"""

    try:
        # Criticality agent uses its own system prompt (not the taxonomy/few-shot one)
        result = call_claude(client, user_prompt, system_prompt=system_prompt)
        if is_opinion:
            result["criticality_level"] = "low"   # enforce hard floor
        return result
    except Exception as e:
        print(f"  ⚠️  Agent 2 (Criticality) Error: {e}")
        return {"criticality_level": "low"}


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def analyze_article_dxy(client, title, date, content,
                         classifier_system_prompt: str = "",
                         criticality_system_prompt: str = "",
                         verbose=True):
    """
    Runs the 2-agent pipeline on a single article.

    Agent 1 (Classifier)  → always runs; uses cached taxonomy + few-shot system prompt
    Agent 2 (Criticality) → skipped when event_number == 0 (Irrelevant)
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:75]}")
        print(f"{'='*60}")

    # --- Agent 1: Classify ---
    if verbose: print("\n🏷️  Agent 1: Classification...")
    a1 = agent_1_classify(
        client, title, date, content,
        system_prompt=classifier_system_prompt,
    )

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

    # --- Agent 2: Criticality ---
    if verbose: print("\n⚡ Agent 2: Criticality assessment...")
    a2 = agent_2_criticality(
        client, title, date, content,
        event_number=event_number,
        content_type=a1.get("content_type", "other"),
        system_prompt=criticality_system_prompt,
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
    ground_truth_csv_path=None,   # ← path to Apr 2–4 labeled CSV
    max_few_shot_examples=12,     # ← how many examples to inject (balanced sample)
    test_row_index=None,
    verbose=True,
):
    """
    Read articles from CSV, run 2-agent pipeline, write results CSV.
    ALL rows are output — irrelevant articles included with blank agent 2 columns.

    When ground_truth_csv_path is provided:
      - Labeled examples are loaded and injected into the classifier system prompt
        as few-shot examples, giving the model real calibration data.
      - The system prompt (taxonomy + examples) is sent with cache_control so
        Anthropic caches it server-side — repeated reads cost ~10% of normal price.

    Columns derived via pandas after the run (zero LLM cost):
      event_name       — looked up from EVENT_NAMES by event_number
      event_tier_label — looked up from TIER_LABELS by event_tier
      is_relevant      — True where event_number != 0
      is_critical      — True where criticality_level in ("high", "medium")
      processed_at     — timestamp for the batch

    Args:
        input_csv_path        : Path to input articles CSV.
        output_csv_path       : Path for output results CSV.
        ground_truth_csv_path : (Optional) Path to labeled ground truth CSV for few-shot examples.
        max_few_shot_examples : Max number of examples to sample from ground truth (default 12).
        test_row_index        : 0-based index to test a single row; None = process all.
        verbose               : Print step-by-step agent output.
    """

    client = get_client()

    # --- Build system prompts (once per run) ---
    few_shot_block = ""
    if ground_truth_csv_path:
        examples       = load_few_shot_examples(ground_truth_csv_path, max_examples=max_few_shot_examples)
        few_shot_block = build_few_shot_block(examples)
        print(f"📝 Few-shot block: {len(few_shot_block)} chars, "
              f"~{len(few_shot_block)//4} tokens (cached after first call)\n")
    else:
        print("ℹ️  No ground truth CSV provided — running without few-shot examples.\n")

    # Classifier: taxonomy + few-shot examples (this is the big cached prompt)
    classifier_system_prompt  = build_system_prompt(few_shot_block)
    # Criticality: separate shorter system prompt, also benefits from caching
    criticality_system_prompt = CRITICALITY_SYSTEM_PROMPT

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

        analysis = analyze_article_dxy(
            client, title, date, content,
            classifier_system_prompt=classifier_system_prompt,
            criticality_system_prompt=criticality_system_prompt,
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
            # ---- agent 1 raw output ----
            "event_number":              event_num,
            "content_type":              a1.get("content_type"),
            "classification_confidence": a1.get("confidence"),
            "event_tier":                tier,
            # ---- agent 2 raw output (None for irrelevant) ----
            "criticality_level": a2.get("criticality_level") if a2 else None,
            "reasoning":    a2.get("reasoning") if a2 else None,
            "direction": a2.get("direction") if a2 else None,
        }

        results.append(out_row)

    # --- Build output DataFrame ---
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
    print(f"   Few-shot        : {'Yes (' + str(max_few_shot_examples) + ' examples)' if ground_truth_csv_path else 'No'}")
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
        input_csv_path="data/dxy_training/claude/dec_jan_cln.csv",
        output_csv_path="data/dxy_training/claude/dec_jan_res_m_4_5.csv",
        ground_truth_csv_path="data/dxy_training/claude/gt_example_ft.csv",
        max_few_shot_examples=39,    
        test_row_index=None,         # set to e.g. 0 to test a single row first
        verbose=True,
    )