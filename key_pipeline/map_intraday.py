"""
Merge LLM-classified articles with minute-level DXY intraday prices and
compute forward returns at 1h, 2h, 3h, 4h, and 1d horizons.

Output is compatible with eval_signal.py.

Usage
-----
    python map_intraday.py                        # uses defaults defined in main()
    python map_intraday.py \\
        --results   data/apr_jan_res.csv \\
        --dxy       data/dxy_intraday_min.csv \\
        --output    data/apr_jan_mapped.csv
"""

import argparse
import pandas as pd
import numpy as np
from label_true_criticality import trailing_sd_at

ET = "America/New_York"
ROLLING_DAYS_DEFAULT = 7

HORIZONS = {
    "Open_5m":  pd.Timedelta(minutes=5),
    "Open_15m": pd.Timedelta(minutes=15),
    "Open_1h":  pd.Timedelta(hours=1),
    "Open_2h":  pd.Timedelta(hours=2),
    "Open_4h":  pd.Timedelta(hours=4),
    "Open_1d":  pd.Timedelta(days=1),
}

OUTPUT_COLS = [
    "id", "title", "link", "published", "source",
    "content_type", "event_number", "event_tier", "event_name", "event_tier_label",
    "criticality_level", "reasoning", "direction",
    "direction_confidence", "direction_source", "direction_conflict",
    "table_used", "classification_confidence",
    "is_relevant", "is_critical",
    "article_published_utc",
    "Time", "Open",
    "Open_5m", "Open_15m", "Open_1h", "Open_2h", "Open_4h", "Open_1d",
    "pct_5m",  "pct_15m",  "pct_1h",  "pct_2h",  "pct_4h",  "pct_1d",
    "sd_15m", "true_criticality", "true_direction",
]


def load_dxy(path: str) -> pd.DataFrame:
    """Load DXY intraday CSV. Timestamps are ET — convert to UTC-naive for joining."""
    dxy = pd.read_csv(path)
    dxy["Time"] = (
        pd.to_datetime(dxy["Time"])
          .dt.tz_localize(ET)
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )
    return dxy.sort_values("Time").reset_index(drop=True)


def load_results(path: str) -> pd.DataFrame:
    """Load classified articles. article_published_utc is true UTC — make UTC-naive, floor to minute.
    Year fix: scrape_publish_time stamps January articles as 2025; they are actually 2026."""
    res = pd.read_csv(path)
    pub = pd.to_datetime(res["article_published_utc"], utc=True, format="mixed").dt.tz_localize(None).dt.floor("min")

    jan_2025 = (pub.dt.month == 1) & (pub.dt.year == 2025)
    pub.loc[jan_2025] += pd.DateOffset(years=1)
    n_fixed = jan_2025.sum()
    if n_fixed:
        print(f"  Year fix: {n_fixed} January 2025 timestamps advanced to 2026")

    res["article_published_utc"] = pub
    return res


def compute_forward_returns(dxy: pd.DataFrame) -> pd.DataFrame:
    """Add Open_Nh / pct_Nh columns via index reindex at each horizon offset."""
    time_indexed = dxy.set_index("Time")["Open"]
    for col, delta in HORIZONS.items():
        dxy[col] = time_indexed.reindex(time_indexed.index + delta).values
    for fwd_col, pct_col in [
        ("Open_5m",  "pct_5m"),  ("Open_15m", "pct_15m"),
        ("Open_1h",  "pct_1h"),  ("Open_2h",  "pct_2h"),
        ("Open_4h",  "pct_4h"),  ("Open_1d",  "pct_1d"),
    ]:
        dxy[pct_col] = (dxy[fwd_col] - dxy["Open"]) / dxy["Open"] * 100
    return dxy


def main(results_csv: str, dxy_csv: str, output_csv: str,
         rolling_days: int = ROLLING_DAYS_DEFAULT):
    print(f"Loading results  : {results_csv}")
    res = load_results(results_csv)
    print(f"  {len(res)} articles, "
          f"{res['article_published_utc'].notna().sum()} with publish timestamps")

    print(f"Loading DXY      : {dxy_csv}")
    dxy = load_dxy(dxy_csv)
    print(f"  {len(dxy)} minute rows  "
          f"({dxy['Time'].min()} → {dxy['Time'].max()})")

    print("Computing forward returns...")
    dxy = compute_forward_returns(dxy)

    DXY_COLS = ["Time", "Open"] + list(HORIZONS.keys()) + ["pct_5m", "pct_15m", "pct_1h", "pct_2h", "pct_4h", "pct_1d"]
    mgd = res.merge(dxy[DXY_COLS], how="left", left_on="article_published_utc", right_on="Time")

    matched = mgd["pct_1h"].notna().sum()
    print(f"Merged: {len(res)} articles → {matched} matched to a DXY minute ({len(res)-matched} outside price range)")

    # Rolling-SD based true_criticality (same logic as label_true_criticality.py)
    print(f"Computing rolling {rolling_days}-day SD at article timestamps...")
    mgd["sd_15m"] = trailing_sd_at(mgd["article_published_utc"], dxy, rolling_days).values

    has_data = mgd["pct_15m"].notna() & mgd["sd_15m"].notna()
    is_high  = has_data & (mgd["pct_15m"].abs() > mgd["sd_15m"])

    mgd["true_criticality"] = None
    mgd.loc[has_data & ~is_high, "true_criticality"] = "not high"
    mgd.loc[is_high, "true_criticality"] = "high"

    mgd["true_direction"] = mgd["pct_15m"].apply(
        lambda x: "up" if pd.notna(x) and x > 0 else ("down" if pd.notna(x) else None)
    )

    high_n    = (mgd["true_criticality"] == "high").sum()
    nothigh_n = (mgd["true_criticality"] == "not high").sum()
    null_n    = mgd["true_criticality"].isna().sum()
    print(f"  true_criticality: {high_n} high / {nothigh_n} not high / {null_n} null (unmatched)")

    # Keep only OUTPUT_COLS that exist (gracefully handle missing cols)
    out_cols = [c for c in OUTPUT_COLS if c in mgd.columns]
    mgd[out_cols].to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge classified articles with DXY intraday prices")
    parser.add_argument("--results", default="data/results_gt_ext200.csv",
                        help="Path to classified articles CSV (output of claude_dxy_predict.py)")
    parser.add_argument("--dxy",     default="data/dxy_intraday_min.csv",
                        help="Path to merged DXY intraday CSV (output of merge_dxy_intraday.py)")
    parser.add_argument("--output",  default="data/results_gt_ext200_mapped.csv",
                        help="Path for output mapped CSV")
    parser.add_argument("--rolling-days", type=int, default=ROLLING_DAYS_DEFAULT,
                        help="Trailing window in days for rolling |pct_15m| SD threshold")
    args = parser.parse_args()

    main(args.results, args.dxy, args.output, rolling_days=args.rolling_days)
