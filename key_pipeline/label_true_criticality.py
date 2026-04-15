"""
Label true_criticality and true_direction on any CSV using the DXY move 15 minutes after each article.

  true_criticality = "high"     if |pct_15m| > rolling SD of |pct_15m| over the
                                  prior N days (default 7), computed per timestamp
  true_criticality = "not high" otherwise
  true_direction   = "up" | "down" | NaN (NaN when no price match)

Timezone logic matches map_intraday.py:
  - DXY CSV timestamps are ET → tz_localize(ET) → tz_convert(UTC) → tz_localize(None)
  - article_published_utc is true UTC → tz_localize(None) → floor("min")
  - January 2025 timestamps are advanced to 2026 (scraper year bug)

Usage
-----
    python label_true_criticality.py                         # uses defaults
    python label_true_criticality.py \\
        --input  data/my_articles.csv \\
        --col    article_published_utc \\
        --dxy    data/dxy_intraday_min.csv \\
        --output data/my_articles_labeled.csv
"""

import argparse
import numpy as np
import pandas as pd

ET = "America/New_York"
DXY_DEFAULT           = "data/dxy_intraday_min.csv"
INPUT_DEFAULT         = "data/gt_100_new_ft_pub_macros.csv"
COL_DEFAULT           = "article_published_utc"
ROLLING_DAYS_DEFAULT  = 7


def load_dxy(path: str) -> pd.DataFrame:
    dxy = pd.read_csv(path)
    dxy["Time"] = (
        pd.to_datetime(dxy["Time"])
          .dt.tz_localize(ET)
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )
    dxy = dxy[["Time", "Open"]].sort_values("Time").reset_index(drop=True)
    # forward price 15 minutes later
    time_idx = dxy.set_index("Time")["Open"]
    dxy["Open_15m"] = time_idx.reindex(time_idx.index + pd.Timedelta(minutes=15)).values
    dxy["pct_15m"]  = (dxy["Open_15m"] - dxy["Open"]) / dxy["Open"] * 100
    return dxy


def trailing_sd_at(query_times: pd.Series, dxy: pd.DataFrame, rolling_days: int) -> pd.Series:
    """
    For each timestamp T in `query_times`, return the sample SD (ddof=1) of
    |pct_15m| over dxy rows whose Time falls in the half-open window
    [T - rolling_days, T).

    Implementation: precompute cumulative count / Σx / Σx² over the dxy minute
    series, then resolve each query in O(1) via np.searchsorted. Total cost is
    O(N + M) where N is the number of dxy minutes and M is the number of
    queries — vs. the previous approach which spent O(N) computing a rolling
    SD at every dxy minute even though we only need values at M minutes.
    """
    times = dxy["Time"].to_numpy()                       # sorted datetime64[ns]
    abs_pct = dxy["pct_15m"].abs().to_numpy()
    valid = ~np.isnan(abs_pct)
    abs_pct_filled = np.where(valid, abs_pct, 0.0)

    # Prepend a 0 so cum[i] = sum of first i entries → range sum is cum[b] - cum[a]
    cum_n  = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    cum_x  = np.concatenate(([0.0], np.cumsum(abs_pct_filled)))
    cum_x2 = np.concatenate(([0.0], np.cumsum(abs_pct_filled ** 2)))

    window = pd.Timedelta(days=rolling_days).to_timedelta64()
    q = query_times.to_numpy()
    q_valid = ~pd.isna(query_times).to_numpy()

    # Indices into `times` for the half-open window [q - window, q)
    a = np.searchsorted(times, np.where(q_valid, q - window, times[0]), side="left")
    b = np.searchsorted(times, np.where(q_valid, q,          times[0]), side="left")

    n  = cum_n[b]  - cum_n[a]
    s  = cum_x[b]  - cum_x[a]
    s2 = cum_x2[b] - cum_x2[a]

    with np.errstate(invalid="ignore", divide="ignore"):
        # Sample variance: (Σx² - (Σx)² / n) / (n - 1)
        var = (s2 - (s * s) / np.where(n > 0, n, 1)) / np.where(n > 1, n - 1, 1)
        sd = np.sqrt(np.where((n > 1) & q_valid, var, np.nan))

    return pd.Series(sd, index=query_times.index)


def load_articles(path: str, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pub = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("min")

    jan_2025 = (pub.dt.month == 1) & (pub.dt.year == 2025)
    n_fixed  = jan_2025.sum()
    if n_fixed:
        pub.loc[jan_2025] += pd.DateOffset(years=1)
        print(f"  Year fix: {n_fixed} January 2025 timestamps advanced to 2026")

    df["_join_key"] = pub
    return df


def main(input_csv: str, time_col: str, dxy_csv: str, output_csv: str,
         rolling_days: int = ROLLING_DAYS_DEFAULT):
    print(f"Loading DXY      : {dxy_csv}  (rolling SD window: {rolling_days}d)")
    dxy = load_dxy(dxy_csv)
    print(f"  {len(dxy):,} minute rows  ({dxy['Time'].min()} → {dxy['Time'].max()})")

    print(f"Loading articles : {input_csv}  (time col: '{time_col}')")
    df = load_articles(input_csv, time_col)
    print(f"  {len(df)} rows, {df['_join_key'].notna().sum()} with timestamps")

    # Drop any existing columns we're about to add
    for col in ["Open_15m", "pct_15m", "sd_15m", "true_criticality", "true_direction"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    merged = df.merge(
        dxy[["Time", "Open_15m", "pct_15m"]],
        how="left",
        left_on="_join_key",
        right_on="Time",
    ).drop(columns=["Time"])

    matched = merged["pct_15m"].notna().sum()
    print(f"  Matched {matched}/{len(merged)} rows to a DXY minute")

    # Compute trailing SD only at the article timestamps (O(N + M), not O(N*W))
    merged["sd_15m"] = trailing_sd_at(merged["_join_key"], dxy, rolling_days).values
    merged = merged.drop(columns=["_join_key"])

    sd_available = merged["sd_15m"].notna().sum()
    print(f"  Rolling SD computed for {sd_available}/{len(merged)} article timestamps  "
          f"(median SD = {merged['sd_15m'].median():.4f}%)")

    # Rows with both pct_15m and sd_15m → label; unmatched rows → null
    has_data = merged["pct_15m"].notna() & merged["sd_15m"].notna()
    is_high  = has_data & (merged["pct_15m"].abs() > merged["sd_15m"])

    merged["true_criticality"] = None
    merged.loc[has_data & ~is_high, "true_criticality"] = "not high"
    merged.loc[is_high, "true_criticality"] = "high"

    merged["true_direction"] = merged["pct_15m"].apply(
        lambda x: "up" if pd.notna(x) and x > 0 else ("down" if pd.notna(x) else None)
    )

    high_n    = (merged["true_criticality"] == "high").sum()
    nothigh_n = (merged["true_criticality"] == "not high").sum()
    null_n    = merged["true_criticality"].isna().sum()
    up_n      = (merged["true_direction"] == "up").sum()
    down_n    = (merged["true_direction"] == "down").sum()
    print(f"  Labeled: {high_n} high  /  {nothigh_n} not high  /  {null_n} null (unmatched)")
    print(f"  Direction: {up_n} up  /  {down_n} down  /  {null_n} null")

    merged.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_DEFAULT)
    parser.add_argument("--col",    default=COL_DEFAULT)
    parser.add_argument("--dxy",    default=DXY_DEFAULT)
    parser.add_argument("--output", default=None,
                        help="Output path. Defaults to input path with '_labeled' suffix.")
    parser.add_argument("--rolling-days", type=int, default=ROLLING_DAYS_DEFAULT,
                        help="Trailing window in days for the rolling |pct_15m| SD threshold.")
    args = parser.parse_args()

    output = args.output
    if output is None:
        stem = args.input.replace(".csv", "")
        output = f"{stem}_labeled.csv"

    main(args.input, args.col, args.dxy, output, rolling_days=args.rolling_days)
