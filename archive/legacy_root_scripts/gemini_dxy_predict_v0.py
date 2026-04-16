import pandas as pd
import json
import time
from datetime import datetime
from google import genai
import os
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
# SHARED HELPER: call Gemini and parse JSON response
# ==============================================================================

def call_gemini(client, prompt: str) -> dict:
    """
    Send a prompt to Gemini Flash-Lite and return parsed JSON.
    We instruct the model to return only JSON in the prompt and parse ourselves.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 1024,
        }
    )

    text = response.text.strip()

    # Strip markdown code fences if present (```json ... ```)
    if "```" in text:
        text = text.split("```json")[-1].split("```")[0].strip()

    return json.loads(text)


# ==============================================================================
# AGENT 1: RELEVANCE FILTER
# ==============================================================================

def agent_1_relevance_filter(client, title, date, content):
    """
    Agent 1: Determines if article impacts DXY (US Dollar Index)
    Returns: {"is_relevant": bool, "surprise_factor": str, "reasoning": str}
    """

    prompt = f"""You are a senior FX strategist screening news for market-moving relevance to the DXY (US Dollar Index).

DXY basket weights: EUR 57.6% | JPY 13.6% | GBP 11.9% | CAD 9.1% | SEK 4.2% | CHF 3.6%

## RELEVANT — likely to move DXY:
- Fed decisions, speeches, or minutes that SURPRISE markets (unexpected tone shift, dot plot changes)
- US macro data that DEVIATES from consensus (CPI, PCE, NFP, GDP, retail sales, ISM)
- New US trade/tariff policy announcements with concrete details
- Debt ceiling escalation, sovereign credit events, or major fiscal packages
- Sudden USD liquidity events (banking stress, repo spikes, TGA drawdowns)
- Geopolitical shock that triggers genuine safe-haven USD flows (war escalation, sanctions)
- Major central bank actions in EUR/JPY/GBP that shift rate differentials vs USD
- Legal/court rulings or signals that change probability of fiscal/trade policy
- Congressional developments affecting tariffs, stimulus, or debt ceiling odds
- Leaked or reported Fed intentions ahead of official decisions
- Election polling shifts that reprice future fiscal/monetary policy
- Executive orders, executive action signals, or policy reversals in progress
- Regulatory signals that shift probability of a known upcoming decision
- "Sources say...", "Officials considering...", "Expected to..." framing around
  any monetary, fiscal, or trade policy

## NOT RELEVANT — likely noise:
- Data releases that met consensus with no revision surprises
- Fed speakers reiterating the existing party line
- Analyst forecasts, price targets, or model predictions
- Technical analysis, chart commentary, or historical reviews
- Domestic political news without fiscal/monetary consequence
- Corporate earnings unless they signal broad economic inflection
- Emerging market or minor currency news without USD spillover

## BORDERLINE — flag but be cautious:
- Geopolitical events without a clear USD safe-haven narrative
- Soft data (sentiment surveys) unless they dramatically miss
- Secondary Fed officials making off-script comments

## Key heuristic:
Ask yourself: "Would a macro hedge fund desk update their DXY position after reading this?"
If yes → relevant.

Ask: "Does this article change the PROBABILITY of a known policy outcome?"
If yes → relevant, even if nothing has been officially decided yet.

Article:
Title: {title}
Date: {date}
Content: {content[:2000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "is_relevant": true or false,
  "surprise_factor": "high/medium/low/none",
  "reasoning": "One sentence: what happened and why it does/doesn't move DXY"
}}"""

    try:
        return call_gemini(client, prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 1 Error: {e}")
        return {"is_relevant": False, "surprise_factor": "none", "reasoning": f"Parse error: {e}"}


# ==============================================================================
# AGENT 2: EVENT CLASSIFIER
# ==============================================================================

def agent_2_event_classifier(client, title, date, content):
    """
    Agent 2: Classifies article into one of the FX event categories.
    Returns: {"event_number": int, "event_name": str, "vs_consensus": str,
              "confidence": str, "reasoning": str}
    """

    prompt = f"""Classify this article into the SINGLE best-matching DXY event category.

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

Article:
Title: {title}
Date: {date}
Content: {content[:2000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "event_number": <integer 1-39>,
  "event_name": "Exact name from list above",
  "vs_consensus": "beat/miss/in-line/not-applicable",
  "confidence": "high/medium/low",
  "reasoning": "One sentence: what the article describes and why this category fits"
}}"""

    try:
        return call_gemini(client, prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 2 Error: {e}")
        return {"event_number": 28, "event_name": "Other / Mixed", "vs_consensus": "not-applicable",
                "confidence": "low", "reasoning": f"Parse error: {e}"}


# ==============================================================================
# AGENT 3: IMPACT PREDICTOR
# ==============================================================================

def agent_3_impact_predictor(client, title, date, content, event_info):
    """
    Agent 3: Predicts direction and magnitude of DXY impact.
    Returns: {"direction": str, "magnitude": str, "confidence": str,
              "pip_estimate": int, "primary_driver": str, "key_risk": str,
              "reasoning": str}
    """

    prompt = f"""You are predicting the directional impact on DXY (US Dollar Index) from a known macro event.

Event: {event_info.get('event_name', 'Unknown')} (Category #{event_info.get('event_number', '?')})
Consensus beat/miss: {event_info.get('vs_consensus', 'unknown')}

## DXY DIRECTION LOGIC

USD BULLISH (DXY UP) scenarios:
- Fed hikes, hawkish surprises, or QT → higher US rates attract capital
- Inflation above expected → forces Fed hawkishness
- Strong jobs/GDP → economy doesn't need cuts
- Risk-off / safe-haven flows → USD demand spike
- Foreign CB dovish surprise → rate differential narrows in USD's favor

USD BEARISH (DXY DOWN) scenarios:
- Fed cuts, dovish pivots, QE → rate differential narrows
- Weak data → Fed forced to cut sooner
- Fiscal crisis / debt downgrade → reserve currency credibility hit
- Risk-on rally → capital flows away from safe-haven USD
- Strong foreign CB hawkish surprise → EUR/JPY/GBP strengthens vs USD

NEUTRAL scenarios:
- Data in-line with consensus (fully priced in)
- Contradictory signals (e.g., strong jobs + falling wages)
- Event is relevant but impact is ambiguous

## PIP MAGNITUDE GUIDE (DXY typically moves in 20-300 pip ranges)
- 200-300 pips: Fed surprise rate decision, major crisis (2008-level), NFP >200k miss
- 100-200 pips: Clear CPI/PCE shock, strong NFP surprise, hawkish/dovish pivot
- 50-100 pips:  Moderate data surprise, Fed speaker making unexpected remarks
- 20-50 pips:   Minor beat/miss, secondary data, soft geopolitical tension
- <20 pips:     In-line data, noise

## IMPORTANT: EUR is 57.6% of DXY. A EUR/USD move is nearly mirrored in DXY (inverted).

Article:
Title: {title}
Date: {date}
Content: {content[:2000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "direction": "UP/DOWN/NEUTRAL",
  "magnitude": "HIGH/MEDIUM/LOW",
  "confidence": "high/medium/low",
  "pip_estimate": <integer between -300 and 300, negative=DXY down, positive=DXY up>,
  "primary_driver": "One phrase — the specific mechanism (e.g. 'rate differential widening')",
  "key_risk": "What could invalidate this prediction (1 sentence)",
  "reasoning": "2-3 sentences walking through the logic with reference to DXY basket composition"
}}"""

    try:
        return call_gemini(client, prompt)
    except Exception as e:
        print(f"  ⚠️  Agent 3 Error: {e}")
        return {"direction": "NEUTRAL", "magnitude": "LOW", "confidence": "low",
                "pip_estimate": 0, "primary_driver": "error", "key_risk": "error",
                "reasoning": f"Parse error: {e}"}


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def analyze_article_dxy(client, title, date, content, verbose=True):
    """
    Runs the 3-agent funnel on a single article.
    Returns a combined result dict, or None if Agent 1 filters it out.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:75]}")
        print(f"{'='*60}")

    # --- Agent 1: Relevance ---
    if verbose: print("\n🔍 Agent 1: Checking relevance...")
    a1 = agent_1_relevance_filter(client, title, date, content)

    if not a1.get("is_relevant", False):
        if verbose:
            print(f"❌ NOT RELEVANT: {a1.get('reasoning', '')}")
        return None

    if verbose:
        print(f"✅ RELEVANT (surprise={a1.get('surprise_factor','?')}): {a1.get('reasoning', '')}")

    # --- Agent 2: Classification ---
    if verbose: print("\n🏷️  Agent 2: Classifying event...")
    a2 = agent_2_event_classifier(client, title, date, content)

    if verbose:
        print(f"📋 Event #{a2.get('event_number')}: {a2.get('event_name')}")
        print(f"   vs consensus : {a2.get('vs_consensus')}")
        print(f"   confidence   : {a2.get('confidence')}")
        print(f"   reasoning    : {a2.get('reasoning')}")

    # --- Agent 3: Impact ---
    if verbose: print("\n📊 Agent 3: Predicting DXY impact...")
    a3 = agent_3_impact_predictor(client, title, date, content, a2)

    if verbose:
        arrow = "📈" if a3.get("direction") == "UP" else "📉" if a3.get("direction") == "DOWN" else "➡️"
        print(f"{arrow} Direction    : {a3.get('direction')}")
        print(f"   Magnitude    : {a3.get('magnitude')}")
        print(f"   Pip estimate : {a3.get('pip_estimate', 0):+d}")
        print(f"   Confidence   : {a3.get('confidence')}")
        print(f"   Driver       : {a3.get('primary_driver')}")
        print(f"   Key risk     : {a3.get('key_risk')}")
        print(f"   Reasoning    : {a3.get('reasoning')}")

    return {
        "article": {"title": title, "date": date},
        "agent1_relevance": a1,
        "agent2_classification": a2,
        "agent3_impact": a3,
        "final_verdict": {
            "is_relevant": True,
            "event": a2.get("event_name"),
            "dxy_direction": a3.get("direction"),
            "dxy_magnitude": a3.get("magnitude"),
            "pip_estimate": a3.get("pip_estimate"),
            "overall_confidence": a2.get("confidence"),
        },
    }


# ==============================================================================
# CSV PROCESSING PIPELINE
# ==============================================================================

def process_csv_to_csv(
    input_csv_path,
    output_csv_path,
    content_column=None,
    test_first_row_only=False,
    verbose=True,
):
    """
    Read articles from CSV, run through 3-agent DXY pipeline, write results CSV.

    Args:
        input_csv_path      : Path to input CSV file.
        output_csv_path     : Path for output CSV file.
        content_column      : Column name to use as article body (falls back to title).
        test_first_row_only : If True, only processes the first row — use for testing.
        verbose             : Print step-by-step agent output.
    """

    client = get_client()

    print(f"📂 Reading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"✅ Loaded {len(df)} rows")

    if test_first_row_only:
        print("⚡ TEST MODE: processing first row only\n")
        df = df.head(1)

    results = []
    filtered_count = 0

    for idx, row in df.iterrows():

        title = str(row.get("title", ""))
        date  = str(row.get("published", row.get("article_date", "")))
        article_id = row.get("id", idx)

        content = (
            str(row.get(content_column, title))
            if content_column and content_column in df.columns
            else title
        )

        analysis = analyze_article_dxy(client, title, date, content, verbose=verbose)

        if analysis:
            results.append({
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
                # ---- agent outputs ----
                "is_relevant":       True,
                "surprise_factor":   analysis["agent1_relevance"].get("surprise_factor"),
                "relevance_reasoning": analysis["agent1_relevance"].get("reasoning"),
                "event_number":      analysis["agent2_classification"].get("event_number"),
                "event_name":        analysis["agent2_classification"].get("event_name"),
                "vs_consensus":      analysis["agent2_classification"].get("vs_consensus"),
                "event_confidence":  analysis["agent2_classification"].get("confidence"),
                "event_reasoning":   analysis["agent2_classification"].get("reasoning"),
                "dxy_direction":     analysis["agent3_impact"].get("direction"),
                "dxy_magnitude":     analysis["agent3_impact"].get("magnitude"),
                "dxy_pip_estimate":  analysis["agent3_impact"].get("pip_estimate"),
                "primary_driver":    analysis["agent3_impact"].get("primary_driver"),
                "key_risk":          analysis["agent3_impact"].get("key_risk"),
                "impact_confidence": analysis["agent3_impact"].get("confidence"),
                "impact_reasoning":  analysis["agent3_impact"].get("reasoning"),
                "processed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            filtered_count += 1

        # Rate limiting — Gemini Flash-Lite free tier: 15 RPM, 1,000 RPD
        # With 3 agents per article = 3 calls per row.
        # 15 RPM = 1 request per 4 seconds minimum.
        # 3 calls × 4 s = 12 s theoretical minimum; 5 s inter-row pause is safe buffer.
        if not test_first_row_only:
            time.sleep(5.0)

    # --- Write output ---
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv_path, index=False)

        print(f"\n{'='*60}")
        print(f"💾 Saved → {output_csv_path}")
        print(f"\n📊 SUMMARY")
        print(f"   Rows processed : {len(df)}")
        print(f"   Relevant        : {len(results)}")
        print(f"   Filtered out    : {filtered_count}")
        if len(df) > 0:
            print(f"   Relevance rate  : {len(results)/len(df)*100:.1f}%")

        if len(out_df) > 0:
            print(f"\n📈 Event distribution:")
            for event, count in out_df["event_name"].value_counts().items():
                print(f"   {count:>3}x  {event}")

        return out_df

    else:
        print("\n⚠️  No relevant articles found — output CSV not written.")
        return pd.DataFrame()


# ==============================================================================
# MAIN — test mode: first row only
# ==============================================================================

if __name__ == "__main__":

    output_df = process_csv_to_csv(
        input_csv_path="data/dxy_training/dxy_condensed.csv",
        output_csv_path="data/dxy_training/gemini_test.csv",
        content_column=None,        # set to e.g. "description" if you have a body column
        test_first_row_only=True,   # ← flip to False for full run
        verbose=True,
    )