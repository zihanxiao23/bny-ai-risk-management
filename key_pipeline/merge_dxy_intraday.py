"""
Merge all per-period DXY intraday CSVs into a single sorted file.

Reads:  data/dxy_intraday/*.csv
Writes: data/dxy_intraday_min.csv
"""

import glob
import pandas as pd

INPUT_GLOB  = "data/dxy_intraday/*.csv"
OUTPUT_CSV  = "data/dxy_intraday_min.csv"


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f}")

    frames = []
    for f in files:
        df = pd.read_csv(f)

        # Some files store timestamps in UTC under a "Time_UTC" column instead
        # of the local-time "Time" column used by the per-period Barchart
        # exports. Convert UTC → America/New_York (Barchart's local tz) and
        # drop tz so it matches the rest.
        if "Time" not in df.columns and "Time_UTC" in df.columns:
            df = df[df["Time_UTC"].astype(str).str.match(r"^\d")].copy()
            ts_utc = pd.to_datetime(df["Time_UTC"], format="mixed", utc=True)
            df["Time"] = ts_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
            df = df.drop(columns=["Time_UTC"])
        else:
            # Drop Barchart footer rows (e.g. "Downloaded from Barchart.com...")
            df = df[df["Time"].astype(str).str.match(r"^\d")].copy()
            df["Time"] = pd.to_datetime(df["Time"], format="mixed")

        print(f"  {f.split('/')[-1]}: {len(df)} rows  "
              f"({df['Time'].min()} → {df['Time'].max()})")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    total_raw = len(combined)

    # Pass 1: drop fully identical rows (same timestamp + same OHLC data)
    combined  = combined.drop_duplicates()
    after_p1  = len(combined)

    # Pass 2: drop rows sharing a timestamp but with differing values (keep first)
    combined  = combined.drop_duplicates(subset=["Time"])
    after_p2  = len(combined)

    combined  = combined.sort_values("Time").reset_index(drop=True)

    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\nRaw rows across all files : {total_raw}")
    print(f"After exact-row dedup     : {after_p1}  ({total_raw - after_p1} removed)")
    print(f"After timestamp dedup     : {after_p2}  ({after_p1 - after_p2} removed)")
    print(f"Date range: {combined['Time'].min()} → {combined['Time'].max()}")
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
