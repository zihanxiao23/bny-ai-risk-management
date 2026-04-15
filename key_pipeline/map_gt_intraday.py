"""
Map DXY intraday prices onto gt_example_ft.csv, adding:
  dxy_open, dxy_1h, dxy_2h, dxy_4h, dxy_1d
  pct_1h, pct_2h, pct_4h, pct_1d

Timezone logic mirrors map_intraday.py (validated):
  - DXY CSV timestamps are ET  → tz_localize(ET) → tz_convert(UTC) → tz_localize(None)
  - article_published_utc is true UTC → tz_localize(None) → floor("min")
  - January 2025 timestamps are advanced to 2026 (scraper year bug)

Writes result in-place to data/gt_example_ft.csv.
"""

import pandas as pd

GT_PATH  = "data/gt_example_ft.csv"
DXY_PATH = "data/dxy_intraday_min.csv"
ET       = "America/New_York"

HORIZONS = {
    "dxy_1h": pd.Timedelta(hours=1),
    "dxy_2h": pd.Timedelta(hours=2),
    "dxy_4h": pd.Timedelta(hours=4),
    "dxy_1d": pd.Timedelta(days=1),
}


def load_dxy(path: str) -> pd.DataFrame:
    """DXY timestamps are ET — localize then convert to UTC-naive."""
    dxy = pd.read_csv(path)
    dxy["Time"] = (
        pd.to_datetime(dxy["Time"])
          .dt.tz_localize(ET)
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )
    return dxy[["Time", "Open"]].sort_values("Time").reset_index(drop=True)


def compute_forward_returns(dxy: pd.DataFrame) -> pd.DataFrame:
    """Add forward price and pct change columns at each horizon."""
    time_idx = dxy.set_index("Time")["Open"]
    for col, delta in HORIZONS.items():
        dxy[col] = time_idx.reindex(time_idx.index + delta).values
    for fwd_col, pct_col in [
        ("dxy_1h", "pct_1h"), ("dxy_2h", "pct_2h"),
        ("dxy_4h", "pct_4h"), ("dxy_1d", "pct_1d"),
    ]:
        dxy[pct_col] = (dxy[fwd_col] - dxy["Open"]) / dxy["Open"] * 100
    return dxy


def load_gt(path: str) -> pd.DataFrame:
    """
    Load ground truth CSV. article_published_utc is true UTC —
    strip tz and floor to minute for joining. Apply January 2025 → 2026 fix.
    """
    gt = pd.read_csv(path)

    pub = pd.to_datetime(gt["article_published_utc"], utc=True, errors="coerce")
    pub = pub.dt.tz_localize(None).dt.floor("min")

    jan_2025 = (pub.dt.month == 1) & (pub.dt.year == 2025)
    n_fixed  = jan_2025.sum()
    if n_fixed:
        pub.loc[jan_2025] += pd.DateOffset(years=1)
        print(f"  Year fix: {n_fixed} January 2025 timestamps advanced to 2026")

    gt["_join_key"] = pub
    return gt


def main():
    print(f"Loading DXY intraday: {DXY_PATH}")
    dxy = load_dxy(DXY_PATH)
    dxy = compute_forward_returns(dxy)
    print(f"  {len(dxy):,} minute rows  ({dxy['Time'].min()} → {dxy['Time'].max()})")

    print(f"Loading ground truth: {GT_PATH}")
    gt = load_gt(GT_PATH)
    print(f"  {len(gt)} rows, {gt['_join_key'].notna().sum()} with publish timestamps")

    # Drop any existing DXY columns so we get a clean merge
    drop_cols = ["dxy_open", "dxy_1h", "dxy_2h", "dxy_4h", "dxy_1d",
                 "pct_1h", "pct_2h", "pct_4h", "pct_1d", "Time"]
    gt = gt.drop(columns=[c for c in drop_cols if c in gt.columns])

    dxy_cols = ["Time", "Open", "dxy_1h", "dxy_2h", "dxy_4h", "dxy_1d",
                "pct_1h", "pct_2h", "pct_4h", "pct_1d"]
    merged = gt.merge(
        dxy[dxy_cols].rename(columns={"Open": "dxy_open"}),
        how="left",
        left_on="_join_key",
        right_on="Time",
    ).drop(columns=["_join_key", "Time"])

    matched = merged["pct_1h"].notna().sum()
    print(f"  Matched {matched}/{len(merged)} rows to a DXY minute")

    merged.to_csv(GT_PATH, index=False)
    print(f"✅ Saved → {GT_PATH}")


if __name__ == "__main__":
    main()
