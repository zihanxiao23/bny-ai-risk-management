import os
import anthropic
import pandas as pd
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from key_pipeline.predict.consensus_extract_hf import (
    load_consensus_table,
    lookup_consensus_history,
    build_consensus_context_with_history,
)

MACRO_COLUMNS = [
    "us_2yr_yield",
    "vix",
    "cpi_yoy",
    "dxy_zscore_52w_broad"
]

# ==============================================================================
# RAG EMBEDDING BACKEND
#
# Uses sentence-transformers (local, free, no extra API key).
# Install: pip install sentence-transformers
#
# To swap to OpenAI embeddings instead:
#   1. pip install openai
#   2. Set OPENAI_API_KEY in your .env
#   3. Replace _embed_texts() below with the openai version (commented out)
# ==============================================================================

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

_RAG_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, 80MB, strong semantic similarity
_rag_encoder    = None                   # lazy-loaded singleton

def _get_encoder():
    global _rag_encoder
    if _rag_encoder is None:
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers\n"
                "Or set USE_RAG=False in process_csv_to_csv() to disable RAG."
            )
        print(f"🔧 Loading RAG embedding model: {_RAG_MODEL_NAME}")
        _rag_encoder = _ST(_RAG_MODEL_NAME)
    return _rag_encoder

def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings → (N, D) float32 array, L2-normalised."""
    encoder = _get_encoder()
    return encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

# --- OpenAI alternative (uncomment to use) ---
# import openai as _oai
# def _embed_texts(texts: list[str]) -> np.ndarray:
#     client = _oai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
#     vecs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
#     arr  = np.array(vecs, dtype=np.float32)
#     return arr / np.linalg.norm(arr, axis=1, keepdims=True)  # normalise

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

# Content types that can never produce high criticality (hard floors)
NON_ACTIONABLE_CONTENT_TYPES = {"opinion", "analysis", "preview", "recap"}


# ==============================================================================
# FEW-SHOT EXAMPLE LOADING
#
# Loads labeled ground truth articles to inject into prompts as few-shot examples.
#
# Expected ground truth CSV columns:
#   title               — article headline (join key if no 'id')
#   full_text           — article body (or 'content' as fallback)
#   gt_content_type     — one of: hard_news | opinion | analysis | recap | preview | other
#   gt_event_name       — canonical name from EVENT_NAMES (e.g. "Trade Tariff / Policy Shock")
#   gt_criticality_level— one of: high | medium | low
#
# Sampling strategy: balanced across event types so the model sees diverse examples,
# not just whatever event happens to dominate the first rows.
# ==============================================================================

def load_few_shot_examples(ground_truth_csv_path, max_examples=42):
    """
    Load ALL labeled ground truth articles for use as a RAG knowledge base.

    When USE_RAG=True (default), retrieval handles relevance selection at
    runtime — so we load every example here rather than sampling.
    max_examples is kept as a hard cap for safety but defaults to 42 so
    all rows are included.

    Returns a list of dicts ready for build_rag_index() and build_rag_few_shot_block().
    """
    print(f"📂 Loading RAG ground truth: {ground_truth_csv_path}")
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

    # Load all rows (RAG retrieval selects the relevant ones per article)
    sampled = gt_df.head(max_examples).reset_index(drop=True)

    # Resolve content column
    content_col = (
        "full_text" if "full_text" in gt_df.columns
        else "content" if "content" in gt_df.columns
        else None
    )

    # Market movement columns — present in new GT CSV, optional for backwards compat
    _MOVE_COLS = ["pct_1h", "pct_2h", "pct_4h", "pct_1d"]

    def _fmt_pct(val):
        """Format a percentage move for display, or return 'N/A' if missing."""
        if val is None:
            return "N/A"
        try:
            import math
            f = float(val)
            if math.isnan(f):
                return "N/A"
            return f"{f:+.3f}%"
        except (TypeError, ValueError):
            return "N/A"

    examples = []
    for _, row in sampled.iterrows():
        if content_col:
            content_preview = str(row.get(content_col, ""))[:600].strip()
        else:
            content_preview = ""   # fall back to title only

        # Collect actual DXY market moves if present in the ground truth CSV
        moves = {col: _fmt_pct(row.get(col)) for col in _MOVE_COLS
                 if col in gt_df.columns}

        examples.append({
            "title":             str(row["title"]),
            "content_preview":   content_preview,
            "content_type":      str(row["gt_content_type"]),
            "event_name":        str(row["gt_event_name"]),   # resolved from number
            "criticality_level": str(row["gt_criticality_level"]),
            "moves":             moves,   # dict of pct_1h/2h/4h/1d, may be empty
        })

    print(f"✅ {len(examples)} few-shot examples loaded")
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
        "",
        "The actual_dxy_* fields show how DXY actually moved after each",
        "article was published. Use them to understand what criticality",
        "levels correspond to in terms of realized market impact.",
        "These outcomes are shown for calibration only — you will NOT",
        "have this information at prediction time.",
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
        lines.append(f"  criticality_level → {ex['criticality_level']}")
        moves = ex.get("moves", {})
        if moves:
            lines.append(f"ACTUAL DXY MOVES (realized after publication):")
            if "pct_1h" in moves:
                lines.append(f"  actual_dxy_1h → {moves['pct_1h']}")
            if "pct_2h" in moves:
                lines.append(f"  actual_dxy_2h → {moves['pct_2h']}")
            if "pct_4h" in moves:
                lines.append(f"  actual_dxy_4h → {moves['pct_4h']}")
            if "pct_1d" in moves:
                lines.append(f"  actual_dxy_1d → {moves['pct_1d']}")
        lines.append("")

    lines.append("════════════════════════════════════════════════════════════════")
    lines.append("Now classify the new article below using the same judgment.\n")

    return "\n".join(lines)


# ==============================================================================
# RAG INDEX — built once per run from all 42 GT articles
# ==============================================================================

def build_rag_index(examples: list[dict]) -> dict:
    """
    Encode all GT examples into a searchable embedding index.

    We embed: title + first 600 chars of content (same as what the model sees).
    Called once at startup; the index is reused for every article in the batch.

    Returns a dict with keys: embeddings (np.ndarray), examples (list[dict]).
    """
    texts = [
        f"{ex['title']}. {ex['content_preview']}"
        for ex in examples
    ]

    embeddings = _embed_texts(texts)

    # --- NEW: build macro matrix ---
    macro_matrix = np.array([
        [ex.get(col, 0.0) for col in MACRO_COLUMNS]
        for ex in examples
    ], dtype=np.float32)

    # normalize
    macro_mean = macro_matrix.mean(axis=0)
    macro_std = macro_matrix.std(axis=0) + 1e-8
    macro_matrix = (macro_matrix - macro_mean) / macro_std

    return {
        "embeddings": embeddings,
        "examples": examples,

        # --- NEW ---
        "macro_matrix": macro_matrix,
        "macro_mean": macro_mean,
        "macro_std": macro_std,
    }

def compute_macro_similarity(query_vec, macro_matrix, mean, std):
    query_vec = np.array(query_vec, dtype=np.float32)
    query_vec = (query_vec - mean) / std

    dist = np.linalg.norm(macro_matrix - query_vec, axis=1)
    return -dist

def retrieve_similar_examples(
    title,
    content,
    rag_index,
    query_macro,        # --- NEW ARG ---
    k=6,
    alpha=0.7,
    beta=0.3
):
    query_text = f"{title}. {content[:600]}"
    query_emb = _embed_texts([query_text])[0]

    # semantic similarity (existing)
    semantic_scores = rag_index["embeddings"] @ query_emb

    # --- NEW: macro similarity ---
    macro_scores = compute_macro_similarity(
        query_macro,
        rag_index["macro_matrix"],
        rag_index["macro_mean"],
        rag_index["macro_std"]
    )

    # --- NEW: combined score ---
    final_scores = alpha * semantic_scores + beta * macro_scores

    top_k = min(k, len(rag_index["examples"]))
    idx = np.argsort(final_scores)[::-1][:top_k]

    return [
        (rag_index["examples"][i], float(final_scores[i]))
        for i in idx
    ]


def build_rag_few_shot_block(retrieved: list[tuple[dict, float]]) -> str:
    """
    Render RAG-retrieved examples into a prompt block for the user message.

    Similar to build_few_shot_block() but:
      - Includes the cosine similarity score so the model knows how close
        each example is to the current article.
      - Goes in the USER prompt (not the system prompt) so the static
        taxonomy system prompt stays byte-identical and remains cached.

    retrieved: list of (example_dict, similarity_score) from retrieve_similar_examples()
    """
    if not retrieved:
        return ""

    lines = [
        "════════════════════════════════════════════════════════════════",
        "RETRIEVED EXAMPLES — most similar labeled articles from ground truth",
        "Similarity score shown (1.0 = identical, 0.0 = unrelated).",
        "Use these to calibrate your classification of the article below.",
        "════════════════════════════════════════════════════════════════\n",
    ]

    for i, (ex, sim) in enumerate(retrieved, 1):
        lines.append(f"── Retrieved Example {i}  (similarity: {sim:.3f}) ────────────────")
        lines.append(f"Title   : {ex['title']}")
        if ex.get("content_preview"):
            lines.append(f"Content : {ex['content_preview']}...")
        lines.append("LABELS:")
        lines.append(f"  content_type      → {ex['content_type']}")
        lines.append(f"  event_name        → {ex['event_name']}")
        lines.append(f"  criticality_level → {ex['criticality_level']}")
        moves = ex.get("moves", {})
        if moves:
            lines.append("ACTUAL DXY MOVES (realized after publication):")
            for key in ("pct_1h", "pct_2h", "pct_4h", "pct_1d"):
                if key in moves:
                    lines.append(f"  actual_dxy_{key[4:]} → {moves[key]}")
        lines.append("")

    lines.append("════════════════════════════════════════════════════════════════")
    lines.append("Now classify the NEW article below using the same judgment.\n")

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


def build_system_prompt() -> str:
    """
    Build the static classifier system prompt (taxonomy only).

    The few-shot examples have moved to the per-article USER prompt via RAG,
    so this system prompt is now byte-identical across all calls and benefits
    fully from Anthropic server-side prompt caching.
    """
    return TAXONOMY_SYSTEM_PROMPT


# ==============================================================================
# AGENT 1: EVENT CLASSIFIER
# ==============================================================================

def agent_1_classify(client, title, date, content,
                      system_prompt: str = "",
                      rag_few_shot_block: str = ""):
    """
    Classify article into the DXY event taxonomy.

    rag_few_shot_block: pre-rendered retrieved examples string from
    build_rag_few_shot_block(). Injected at the top of the user prompt
    (not the system prompt) so the taxonomy system prompt stays cached.

    Returns:
      event_number : 0 = Irrelevant, 28 = Other/Mixed, 1–39 = named event
      content_type : hard_news | opinion | analysis | recap | preview | other
      confidence   : high | medium | low
    """

    few_shot_header = rag_few_shot_block if rag_few_shot_block else ""

    user_prompt = f"""{few_shot_header}Classify the article below using the taxonomy and examples in your instructions.

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
  2. If table present: directional consistency from historical rows
     of the same surprise direction as the current article
  3. If table absent: theoretical implied direction from event taxonomy
  4. Article text — use only to identify context not in structured data

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

── CONTENT TYPE FLOORS (non-negotiable) ────────────────────────
- opinion, analysis, preview, recap → ALWAYS low, no exceptions
- Article written BEFORE a scheduled event → ALWAYS low
- Post-event recap of something already fully priced in → low
- If your reasoning would include the phrases "already priced in",
  "widely expected", "no new information", or "no surprise" → must
  return low

── SURROGATE INDICATORS (ALWAYS LOW) ───────────────────────────
ADP private payrolls, Challenger job cuts, JOLTS job openings,
University of Michigan sentiment, Conference Board consumer confidence.
Never assign medium or high to a surrogate regardless of surprise size.
If the event is classified as event 28 (Other/Mixed) and content is a
surrogate indicator, return low.

── GEOGRAPHY CHECK ──────────────────────────────────────────────
DXY basket weights: EUR 57.6% | JPY 13.6% | GBP 11.9% | CAD 9.1% |
                    SEK 4.2%  | CHF 3.6%
Tariffs or trade events affecting non-basket economies → low unless
the article explicitly links them to USD reserve status or broad dollar
demand. ECB, BOJ, BOE, BOC policy actions are always DXY-relevant.
PBOC, RBA, RBNZ actions → low unless causing broad safe-haven USD demand.

── SOURCE QUALITY ───────────────────────────────────────────────
Authoritative primary sources: AP, Reuters, Bloomberg, WSJ, CNBC,
Fed.gov, BLS.gov, BEA.gov, Treasury.gov, BIS, IMF official releases.
Secondary sources reporting a primary decision → cap at medium and note
in reasoning. If source credibility is uncertain → downgrade one level.

── DOWNGRADE TRIGGERS ───────────────────────────────────────────
- Article is vague or lacks concrete details → downgrade
- Event described as expected with no new information → downgrade
- "The Fed is expected to...", "Markets anticipate...", "Analysts
  predict..." → downgrade
- Data release described as delayed, revised, or partially priced in
  → medium at most

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

Key signals to check before forming your judgment:

DXY POSITIONING: If dxy_zscore_52w_broad > 1.5, DXY is extended and
  crowded — downgrade direction confidence for bullish events (less room
  to run). If < -1.5, DXY is oversold — bullish events have more
  follow-through potential.

CFTC POSITIONING: If cftc_net_usd_zscore > 1.5, USD longs are crowded
  — same downgrade logic. Near zero = neutral, moves can extend either way.

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

# Columns from the input CSV that are not macro variables
_ARTICLE_COLS = {
    "id", "title", "link", "published", "source", "query", "priority_weight",
    "article_date", "prediction_date", "fetched_at", "title_len", "actual_link",
    "full_text", "article_published_raw", "article_published_utc", "found_via",
}

# Macro columns that are always null — skip entirely
_NULL_MACRO_COLS = {"gold_20d_return"}


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
    dxy_lines = []
    if _v("dxy_zscore_52w_broad"):
        dxy_lines.append(f"  DXY 52w z-score         : {_v('dxy_zscore_52w_broad')}")
    if _v("dxy_20d_return_broad"):
        dxy_lines.append(f"  DXY 20d return          : {_v('dxy_20d_return_broad')}%")
    if _v("dxy_pct_above_200d_broad"):
        dxy_lines.append(f"  DXY % above 200d MA     : {_v('dxy_pct_above_200d_broad')}%")
    if _v("cftc_net_usd_zscore"):
        dxy_lines.append(f"  CFTC net USD z-score    : {_v('cftc_net_usd_zscore')}")
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
    Excludes article-specific columns and known-null macro columns.
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
                         macro_context: str = ""):
    """
    Assess criticality of a relevant article.

    Default by tier (factual hard_news only):
      Tier 1 → high  |  Tier 2 → medium  |  Tier 3/4 → low

    Hard floors (enforced both in prompt and in Python post-call):
      preview / recap / analysis / opinion → always low

    consensus_context: pre-formatted historical DXY reaction table string.
    macro_context:     pre-formatted point-in-time macro regime string.
    Both are injected before the article block when provided.
    """

    tier       = EVENT_TIERS.get(event_number, 4)
    tier_label = TIER_LABELS.get(tier)
    event_name = EVENT_NAMES.get(event_number, "Other / Mixed")
    is_non_actionable = content_type in NON_ACTIONABLE_CONTENT_TYPES

    if is_non_actionable:
        default_level = "low"
    elif tier == 1:
        default_level = "high"
    elif tier == 2:
        default_level = "medium"
    else:
        default_level = "low"

    if consensus_context:
        table_reminder = ("A historical response table is provided. "
                          "Read it before forming any judgment on criticality or direction.")
    else:
        table_reminder = "No historical data available. Use qualitative reasoning only."

    consensus_block = f"\n{consensus_context}\n" if consensus_context else ""
    macro_block     = f"\n{macro_context}\n"     if macro_context     else ""

    user_prompt = f"""Assess the criticality of this article for DXY/FX trading.

{table_reminder}

Pre-classification:
  Event type          : {event_name} (Category #{event_number})
  Research-based tier : {tier_label}
  Content type        : {content_type}
  Default criticality : {default_level}  ← starting point; confirm or override
{macro_block}{consensus_block}
Article:
Title: {title}
Date: {date}
Content: {content[:3000]}

Return ONLY valid JSON with no commentary before or after:
{{
  "criticality_level":    "<high|medium|low>",
  "reasoning":            "<1-2 sentences — cite table patterns if table was used, cite article content if not>",
  "direction":            "<up|down|neutral|unclear>",
  "direction_chain":      "<chain of reasoning as described in your instructions>",
  "direction_confidence": "<high|medium|low>",
  "table_used":           <true|false>
}}"""

    try:
        result = call_claude(client, user_prompt, system_prompt=system_prompt,
                             max_tokens=512)
        # Enforce hard floor in Python — model cannot override this
        if is_non_actionable:
            result["criticality_level"] = "low"
        return result
    except Exception as e:
        print(f"  ⚠️  Agent 2 (Criticality) Error: {e}")
        return {"criticality_level": "low"}


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
                         rag_few_shot_block: str = "",
                         verbose=True):
    """
    Runs the 2-agent pipeline on a single article.

    Agent 1 (Classifier)  → always runs; uses cached taxonomy system prompt +
                            per-article RAG examples in the user prompt
    Agent 2 (Criticality) → skipped when event_number == 0 (Irrelevant);
                            receives consensus history context when consensus_df provided;
                            receives macro regime context when macro_row provided
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
        rag_few_shot_block=rag_few_shot_block,
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

    # --- Build consensus context for Agent 2 ---
    consensus_record  = lookup_consensus_history(event_number, consensus_df,
                            article_date=date) if consensus_df is not None else None
    history_table     = build_historical_response_table(event_number, date,
                            consensus_df) if consensus_df is not None else ""
    consensus_context = build_consensus_context_with_history(consensus_record,
                            history_table)

    # --- Build macro context for Agent 2 ---
    macro_context = build_macro_context(macro_row, event_number) if macro_row else ""

    if verbose and consensus_context:
        print(f"   📈 consensus context: {len(history_table.splitlines())} history rows")
    if verbose and macro_context:
        print(f"   🌐 macro context: injected ({len(macro_context)} chars)")

    # --- Agent 2: Criticality ---
    if verbose: print("\n⚡ Agent 2: Criticality assessment...")
    a2 = agent_2_criticality(
        client, title, date, content,
        event_number=event_number,
        content_type=a1.get("content_type", "other"),
        system_prompt=criticality_system_prompt,
        consensus_context=consensus_context,
        macro_context=macro_context,
    )

    if verbose:
        level = a2.get("criticality_level", "low")
        icon  = "🔴" if level == "high" else ("🟡" if level == "medium" else "⚪")
        print(f"{icon} criticality_level={level}")
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
    max_few_shot_examples=42,
    rag_k=6,
    use_rag=True,
    test_row_index=None,
    verbose=True,
):
    """
    Read articles from CSV, run 2-agent pipeline, write results CSV.
    ALL rows are output — irrelevant articles included with blank agent 2 columns.

    RAG MODE (use_rag=True, default):
      - All 42 GT examples are embedded once at startup into a vector index.
      - For each article, the top rag_k most similar examples are retrieved
        and injected into the Agent 1 USER prompt (not system prompt).
      - The taxonomy system prompt is cached server-side as before.
      - Each article gets its own tailored few-shot examples rather than
        a static random sample.

    LEGACY MODE (use_rag=False):
      - Behaves exactly like the original: random balanced sample goes
        into the system prompt as a static few-shot block.

    Columns derived via pandas after the run (zero LLM cost):
      event_name          — looked up from EVENT_NAMES by event_number
      event_tier_label    — looked up from TIER_LABELS by event_tier
      is_relevant         — True where event_number != 0
      is_critical         — True where criticality_level in ("high", "medium")
      rag_top_match       — title of the closest retrieved GT example (RAG mode only)
      rag_top_similarity  — cosine similarity of the closest match (RAG mode only)
      processed_at        — timestamp for the batch
    """

    client = get_client()

    # --- Build system prompt (taxonomy only — cached across all calls) ---
    classifier_system_prompt  = build_system_prompt()
    criticality_system_prompt = CRITICALITY_SYSTEM_PROMPT

    # --- Build RAG index or legacy few-shot block ---
    rag_index  = None
    few_shot_block = ""   # only used in legacy mode

    if ground_truth_csv_path:
        examples = load_few_shot_examples(
            ground_truth_csv_path, max_examples=max_few_shot_examples
        )

        if use_rag:
            rag_index = build_rag_index(examples)
            print(f"🔍 RAG active — retrieving top {rag_k} examples per article\n")
        else:
            # Legacy: static balanced sample in the system prompt
            few_shot_block = build_few_shot_block(examples)
            classifier_system_prompt = build_system_prompt() + "\n\n" + few_shot_block
            print(f"📝 Legacy few-shot block: {len(few_shot_block)} chars, "
                  f"~{len(few_shot_block)//4} tokens (cached after first call)\n")
    else:
        print("ℹ️  No ground truth CSV provided — running without few-shot examples.\n")

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

        # Extract macro variables for this article row
        macro_row = extract_macro_row(row)

        # --- RAG: retrieve the k most similar GT examples for this article ---
        rag_few_shot_block = ""
        rag_top_match      = ""
        rag_top_similarity = None

        if rag_index is not None:
            # Build the macro vector aligned with MACRO_COLUMNS (the same
            # 4-feature schema the index was built from). Missing/NaN macros
            # default to 0 so the row still retrieves.
            query_macro_vec = [
                float(macro_row.get(col, 0.0)) if macro_row.get(col) is not None
                and not pd.isna(macro_row.get(col)) else 0.0
                for col in MACRO_COLUMNS
            ]
            retrieved = retrieve_similar_examples(
                title=title,
                content=content,
                rag_index=rag_index,
                query_macro=query_macro_vec,
                k=rag_k,
            )
            rag_few_shot_block = build_rag_few_shot_block(retrieved)

            if retrieved:
                rag_top_match, rag_top_similarity = retrieved[0][0]["title"], retrieved[0][1]

            if verbose and retrieved:
                print(f"🔍 RAG: {len(retrieved)} examples retrieved "
                      f"(best: {rag_top_similarity:.3f} — {rag_top_match[:50]})")

        analysis = analyze_article_dxy(
            client, title, date, content,
            classifier_system_prompt=classifier_system_prompt,
            criticality_system_prompt=criticality_system_prompt,
            consensus_df=consensus_df,
            macro_row=macro_row,
            rag_few_shot_block=rag_few_shot_block,
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
            # ---- RAG metadata ----
            "rag_top_match":        rag_top_match,
            "rag_top_similarity":   round(rag_top_similarity, 4) if rag_top_similarity is not None else None,
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
    print(f"   Mode            : {'RAG (k=' + str(rag_k) + ')' if use_rag and rag_index else 'Legacy few-shot' if few_shot_block else 'No examples'}")
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
        input_csv_path="data/aug_mar_cutoff_cln.csv",
        output_csv_path="data/results_rag.csv",
        ground_truth_csv_path="data/gt_ext_ft_macros_labeled.csv",
        max_few_shot_examples=42,   # load all 42 — RAG selects the relevant ones
        rag_k=6,                    # retrieve top 6 per article (try 5–8)
        use_rag=True,               # set False to fall back to legacy static few-shot
        test_row_index=1,        # set to an int to test a single row
        verbose=True,
    )