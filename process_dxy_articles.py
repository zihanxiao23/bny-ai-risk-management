import pandas as pd
import json
import time
from datetime import datetime
import openai
import os
from dotenv import load_dotenv


load_dotenv()

# ==============================================================================
# PASTE ALL THE AGENT FUNCTIONS HERE (from previous response)
# ==============================================================================
# ==============================================================================
# AGENT 1: RELEVANCE FILTER
# ==============================================================================
def agent_1_relevance_filter(client, model_type, title, date, content):
    """
    Agent 1: Determines if article impacts DXY (US Dollar Index)
    Returns: {"is_relevant": bool, "reasoning": str}
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
Ask yourself: "Would a macro hedge fund desk update their DXY position after reading this?" If yes → relevant.

## Key heuristic for expectations:
Ask: "Does this article change the PROBABILITY of a known policy outcome?"
If yes → relevant, even if nothing has been officially decided yet.

Article:
Title: {title}
Date: {date}
Content: {content[:2000]}

Return ONLY valid JSON:
{{
  "is_relevant": true/false,
  "surprise_factor": "high/medium/low/none",
  "reasoning": "One sentence: what happened and why it does/doesn't move DXY"
}}
"""

    try:
        if model_type == "openai":
            response = client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            text = response.choices[0].message.content.strip()
        elif model_type == "gemini":
            response = client.generate_content(prompt)
            text = response.text.strip()
        
        # Clean response
        if '```' in text:
            text = text.split('```json')[-1].split('```')[0].strip()
        
        return json.loads(text)
    
    except Exception as e:
        print(f"Agent 1 Error: {e}")
        return {"is_relevant": False, "reasoning": f"Error: {e}"}


# ==============================================================================
# AGENT 2: EVENT CLASSIFIER
# ==============================================================================
def agent_2_event_classifier(client, model_type, title, date, content):
    """
    Agent 2: Classifies article into one of 25 FX events
    Returns: {"event_number": int, "event_name": str, "confidence": str}
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

## POLICY EXPECTATION SHIFTS (often the most impactful DXY movers)
29. Trade Policy Reversal Signal (tariff removal/reduction becoming likely)
30. Trade Policy Escalation Signal (new tariffs or expansion becoming likely)
31. Fed Rate Path Repricing (market odds of cuts/hikes shifting without a decision)
32. Fiscal Policy Probability Shift (stimulus/austerity becoming more/less likely)
33. Legal/Judicial Event Affecting Economic Policy (court ruling, injunction, challenge)
34. Legislative Progress/Blockage (bill passing or failing that has fiscal consequence)
35. Executive Policy Signal (White House trial balloon, leaked plan, executive order signal)

## Interest rate changes from other central banks that impact DXY through rate differentials
36. Rate hike or cut from ECB
37. Rate hike or cut from BOJ
38. Rate hike or cut from BOE
39. Rate hike or cut from other major central bank (RBA, SNB, etc.)

Article:
Title: {title}
Date: {date}
Content: {content[:2000]}

Return ONLY valid JSON:
{{
  "event_number": 1-28,
  "event_name": "Exact name from list above",
  "vs_consensus": "beat/miss/in-line/not-applicable",
  "confidence": "high/medium/low",
  "reasoning": "One sentence: what the article describes and why this category fits"
}}
"""

    try:
        if model_type == "openai":
            response = client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            text = response.choices[0].message.content.strip()
        elif model_type == "gemini":
            response = client.generate_content(prompt)
            text = response.text.strip()
        
        if '```' in text:
            text = text.split('```json')[-1].split('```')[0].strip()
        
        return json.loads(text)
    
    except Exception as e:
        print(f"Agent 2 Error: {e}")
        return {"event_number": None, "event_name": None, "confidence": "low", "reasoning": f"Error: {e}"}


# ==============================================================================
# AGENT 3: IMPACT PREDICTOR
# ==============================================================================
def agent_3_impact_predictor(client, model_type, title, date, content, event_info):
    """
    Agent 3: Predicts impact on DXY
    Returns: {"direction": str, "magnitude": str, "confidence": str, "pip_estimate": int}
    """
    
    prompt = f"""You are predicting the directional impact on DXY (US Dollar Index) from a known macro event.

Event: {event_info['event_name']} (Category #{event_info['event_number']})
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
- <20 pips:     In-line data, noise — reconsider if Agent 1 passed this through

## IMPORTANT: EUR is 57.6% of DXY. A EUR/USD move is nearly mirrored in DXY (inverted).

Article:
Title: {title}
Date: {date}
Content: {content[:2000]}

Return ONLY valid JSON:
{{
  "direction": "UP/DOWN/NEUTRAL",
  "magnitude": "HIGH/MEDIUM/LOW",
  "confidence": "high/medium/low",
  "pip_estimate": integer between -300 and 300 (negative=DXY down, positive=DXY up),
  "primary_driver": "One phrase — the specific mechanism (e.g., 'rate differential widening', 'safe-haven demand')",
  "key_risk": "What could invalidate this prediction (1 sentence)",
  "reasoning": "2-3 sentences walking through the logic with reference to DXY basket composition"
}}
"""

    try:
        if model_type == "openai":
            response = client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            text = response.choices[0].message.content.strip()
        elif model_type == "gemini":
            response = client.generate_content(prompt)
            text = response.text.strip()
        
        if '```' in text:
            text = text.split('```json')[-1].split('```')[0].strip()
        
        return json.loads(text)
    
    except Exception as e:
        print(f"Agent 3 Error: {e}")
        return {"direction": "NEUTRAL", "magnitude": "LOW", "confidence": "low", 
                "pip_estimate": 0, "reasoning": f"Error: {e}"}


# ==============================================================================
# ORCHESTRATOR: MULTI-AGENT FUNNEL
# ==============================================================================
def analyze_article_dxy(client, model_type, title, date, content, verbose=True):
    """
    Multi-agent funnel for DXY analysis
    
    Returns complete analysis or None if filtered out at Agent 1
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {title[:80]}")
        print(f"{'='*60}\n")
    
    # STAGE 1: Relevance Filter
    if verbose: print("🔍 Agent 1: Checking relevance...")
    agent1_result = agent_1_relevance_filter(client, model_type, title, date, content)
    
    if not agent1_result.get("is_relevant", False):
        if verbose:
            print(f"❌ NOT RELEVANT: {agent1_result.get('reasoning', 'No reason given')}")
        return None
    
    if verbose:
        print(f"✅ RELEVANT: {agent1_result.get('reasoning', '')}\n")
    
    # STAGE 2: Event Classification
    if verbose: print("🏷️  Agent 2: Classifying event...")
    agent2_result = agent_2_event_classifier(client, model_type, title, date, content)
    
    if verbose:
        print(f"📋 Event: {agent2_result.get('event_name', 'Unknown')}")
        print(f"   Confidence: {agent2_result.get('confidence', 'unknown')}")
        print(f"   Reasoning: {agent2_result.get('reasoning', '')}\n")
    
    # STAGE 3: Impact Prediction
    if verbose: print("📊 Agent 3: Predicting DXY impact...")
    agent3_result = agent_3_impact_predictor(client, model_type, title, date, content, agent2_result)
    
    if verbose:
        direction_emoji = "📈" if agent3_result.get('direction') == 'UP' else "📉" if agent3_result.get('direction') == 'DOWN' else "➡️"
        print(f"{direction_emoji} Direction: {agent3_result.get('direction', 'UNKNOWN')}")
        print(f"   Magnitude: {agent3_result.get('magnitude', 'UNKNOWN')}")
        print(f"   Pip Estimate: {agent3_result.get('pip_estimate', 0):+d}")
        print(f"   Confidence: {agent3_result.get('confidence', 'unknown')}")
        print(f"   Reasoning: {agent3_result.get('reasoning', '')}")
    
    # COMBINE ALL RESULTS
    return {
        "article": {
            "title": title,
            "date": date
        },
        "agent1_relevance": agent1_result,
        "agent2_classification": agent2_result,
        "agent3_impact": agent3_result,
        "final_verdict": {
            "is_relevant": True,
            "event": agent2_result.get('event_name'),
            "dxy_direction": agent3_result.get('direction'),
            "dxy_magnitude": agent3_result.get('magnitude'),
            "pip_estimate": agent3_result.get('pip_estimate'),
            "overall_confidence": agent2_result.get('confidence')
        }
    }



# ==============================================================================
# EXCEL PROCESSING PIPELINE
# ==============================================================================

def process_csv_to_csv(input_csv_path, output_csv_path, model_type='openai', 
                       content_column=None, verbose=True):
    """
    Read articles from CSV, process through multi-agent system, output results to new CSV
    """
    
    # Setup client
    if model_type == "openai":
        client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # required but ignored locally
)
    elif model_type == "gemini":
        import google.generativeai as genai
        api_key = userdata.get('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-2.0-flash',
               generation_config={"response_mime_type": "application/json"})
    
    # Read input CSV
    print(f"📂 Reading CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)  # Changed from read_excel
    print(f"✅ Loaded {len(df)} articles\n")
    
    # Prepare results list
    results = []
    filtered_count = 0
    
    # Process each article
    for idx, row in df.iterrows():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing {idx+1}/{len(df)}: {row['title'][:60]}...")
            print(f"{'='*70}")
        
        # Extract article data
        title = str(row.get('title', ''))
        date = str(row.get('published', row.get('article_date', '')))
        article_id = row.get('id', idx)
        
        # Get content from specified column or use title as fallback
        if content_column and content_column in df.columns:
            content = str(row.get(content_column, title))
        else:
            content = title  # Use title if no content column
        
        # Run through multi-agent system
        analysis = analyze_article_dxy(
            client, 
            model_type, 
            title, 
            date, 
            content, 
            verbose=verbose
        )
        
        # If article passed Agent 1 (is relevant)
        if analysis:
            result_row = {
                # Original columns
                'id': article_id,
                'title': title,
                'link': row.get('link', ''),
                'published': row.get('published', ''),
                'source': row.get('source', ''),
                'query': row.get('query', ''),
                'priority_weight': row.get('priority_weight', ''),
                'article_date': row.get('article_date', ''),
                'prediction_date': row.get('prediction_date', ''),
                'fetched_at': row.get('fetched_at', ''),
                'title_len': row.get('title_len', len(title)),
                
                # Multi-agent outputs
                'is_relevant': True,
                'relevance_reasoning': analysis['agent1_relevance']['reasoning'],
                'event_number': analysis['agent2_classification']['event_number'],
                'event_name': analysis['agent2_classification']['event_name'],
                'event_confidence': analysis['agent2_classification']['confidence'],
                'event_reasoning': analysis['agent2_classification']['reasoning'],
                'dxy_direction': analysis['agent3_impact']['direction'],
                'dxy_magnitude': analysis['agent3_impact']['magnitude'],
                'dxy_pip_estimate': analysis['agent3_impact']['pip_estimate'],
                'impact_confidence': analysis['agent3_impact']['confidence'],
                'impact_reasoning': analysis['agent3_impact']['reasoning'],
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            results.append(result_row)
        else:
            filtered_count += 1
            if verbose:
                print(f"❌ Article filtered out at Agent 1 (not DXY-relevant)")
        
        # Rate limiting
        if model_type == "gemini":
            time.sleep(4.1)
        else:
            time.sleep(1.0)
    
    # Create output DataFrame
    if results:
        output_df = pd.DataFrame(results)
        
        # Save to CSV
        print(f"\n{'='*70}")
        print(f"💾 Saving results to: {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)  # Changed from to_excel
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total articles processed: {len(df)}")
        print(f"   Relevant articles (passed Agent 1): {len(results)}")
        print(f"   Filtered out: {filtered_count}")
        print(f"   Relevance rate: {len(results)/len(df)*100:.1f}%")
        print(f"\n✅ Output saved to: {output_csv_path}")
        
        # Show distribution of events
        if len(output_df) > 0:
            print(f"\n📈 Event Distribution:")
            event_counts = output_df['event_name'].value_counts()
            for event, count in event_counts.items():
                print(f"   {event}: {count}")
        
        return output_df
    else:
        print(f"\n⚠️ No relevant articles found!")
        return pd.DataFrame()

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    
    # Example 1: Basic usage
    output_df = process_csv_to_csv(
        input_csv_path='dxy_condensed.csv',
        output_csv_path='articles_analyzed.csv',
        model_type='openai',
        verbose=True
    )
    
    """# Example 2: If you have a content column
    output_df = process_csv_to_csv(
        input_csv_path='dxy_condensed.csv',
        output_csv_path='articles_analyzed.csv',
        model_type='openai',
        content_column='description',  # or 'content', 'body', etc.
        verbose=True
    )
    """
    """
    # Example 3: Silent mode for production
    output_df = process_csv_to_csv(
        input_csv_path='dxy_condensed.csv',
        output_csv_path='articles_analyzed.csv',
        model_type='openai',
        verbose=False
    )
    """