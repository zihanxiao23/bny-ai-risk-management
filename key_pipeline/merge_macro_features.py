"""
Merge macro feature columns from articles_with_macro.csv into apr_jan_cln.csv,
joining on 'id'. Output overwrites apr_jan_cln.csv in place.
"""

import pandas as pd

INPUT_BASE  = "data/apr_jan_cln.csv"
INPUT_MACRO = "data/articles_with_macro.csv"

MACRO_COLS = [
    "id",
    "us_2yr_yield", "us_10yr_yield", "us_30yr_yield", "real_yield_10yr",
    "breakeven_5yr", "breakeven_10yr", "yield_curve_2s10s",
    "us_de_2yr_spread", "cpi_yoy", "core_cpi_yoy", "pce_yoy",
    "core_pce_yoy", "pce_gap", "unemployment_rate", "nfp_3m_avg",
    "jolts_openings", "initial_claims_4wma", "sahm_indicator",
    "retail_sales_mom", "gdp_qoq", "ig_oas", "hy_oas", "hy_ig_spread",
    "fed_funds_rate", "regime_flag", "months_since_last_change", "vix",
    "vix_regime", "vix_20d_change", "sp500_20d_return", "gold_20d_return",
    "wti_20d_return", "copper_3m_return", "dxy_zscore_52w_broad",
    "dxy_pct_above_200d_broad", "dxy_20d_return_broad",
    "cftc_net_usd_zscore",
]

def main():
    base  = pd.read_csv(INPUT_BASE)
    macro = pd.read_csv(INPUT_MACRO)

    print(f"Base  : {len(base):,} rows  — {INPUT_BASE}")
    print(f"Macro : {len(macro):,} rows  — {INPUT_MACRO}")

    # Validate requested columns exist in macro file
    available = set(macro.columns)
    missing   = [c for c in MACRO_COLS if c != "id" and c not in available]
    if missing:
        raise ValueError(f"Columns not found in {INPUT_MACRO}:\n  {missing}")

    macro_slim = macro[MACRO_COLS].drop_duplicates(subset=["id"])

    # Warn about any id collisions that would have been silently dropped
    if len(macro_slim) < len(macro[["id"]].drop_duplicates()):
        print("⚠️  Duplicate ids in macro file — keeping first occurrence per id")

    merged = base.merge(macro_slim, on="id", how="left")

    matched = merged[MACRO_COLS[1]].notna().sum()   # spot-check first feature col
    print(f"Matched: {matched}/{len(merged)} rows have macro data")

    merged.to_csv(INPUT_BASE, index=False)
    print(f"✅ Saved {len(merged)} rows → {INPUT_BASE}")

if __name__ == "__main__":
    main()
