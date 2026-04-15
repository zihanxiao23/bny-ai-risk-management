"""
Re-run Agent 2 only on relevant rows that are missing criticality/direction.

Reads:  data/apr_jan_res.csv    (existing results with gaps)
        data/apr_jan_cln.csv    (source of full_text)
Writes: data/apr_jan_res.csv    (in-place, gaps filled)
"""

import pandas as pd
from datetime import datetime
from key_pipeline.archived.claude_dxy_predict_v6 import (
    agent_2_criticality,
    CRITICALITY_SYSTEM_PROMPT,
    get_client,
    EVENT_NAMES,
    EVENT_TIERS,
    TIER_LABELS,
    IRRELEVANT_EVENT_NUMBER,
)

def main(RESULTS_CSV, SOURCE_CSV, RES_FILLED):
    res = pd.read_csv(RESULTS_CSV)
    src = pd.read_csv(SOURCE_CSV)

    # Rows that need Agent 2: relevant but missing direction
    needs_a2 = (
        res["is_relevant"].astype(bool)
        & res["direction"].isna()
    )
    print(f"Rows needing Agent 2: {needs_a2.sum()} / {len(res)}")

    if needs_a2.sum() == 0:
        print("Nothing to do.")
        return

    # Build a lookup from title → full_text using source CSV
    text_lookup = src.set_index("title")["full_text"].to_dict()

    client = get_client()
    criticality_system_prompt = CRITICALITY_SYSTEM_PROMPT

    idx_list = res[needs_a2].index.tolist()
    for i, idx in enumerate(idx_list, 1):
        row        = res.loc[idx]
        title      = str(row["title"])
        date       = str(row.get("published", row.get("article_date", "")))
        event_num  = int(row["event_number"])
        content_type = str(row.get("content_type", "other"))

        raw_text   = text_lookup.get(title, "")
        content    = str(raw_text).strip() if pd.notna(raw_text) and str(raw_text).strip() else title

        print(f"[{i}/{needs_a2.sum()}] {title[:70]}")

        a2 = agent_2_criticality(
            client, title, date, content,
            event_number=event_num,
            content_type=content_type,
            system_prompt=criticality_system_prompt,
        )

        res.at[idx, "criticality_level"] = a2.get("criticality_level")
        res.at[idx, "reasoning"]         = a2.get("reasoning")
        res.at[idx, "direction"]         = a2.get("direction")

    # Recompute derived columns
    res["is_critical"]  = res["criticality_level"].isin(["high", "medium"])
    res["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    res.to_csv(RES_FILLED, index=False)
    print(f"\nSaved {len(res)} rows → {RES_FILLED}")
    print(f"Critical: {res['is_critical'].sum()} / {res['is_relevant'].astype(bool).sum()} relevant")


if __name__ == "__main__":
    RESULTS_CSV = "data/apr_jan_res.csv"
    SOURCE_CSV  = "data/apr_jan_cln.csv"

    RES_FILLED = "data/apr_jan_res_filled.csv"
    main(RESULTS_CSV, SOURCE_CSV, RES_FILLED)
