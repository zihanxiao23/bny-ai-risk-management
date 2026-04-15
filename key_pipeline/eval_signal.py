"""
LLM Signal Evaluation — DXY Directional Accuracy
==================================================
Evaluates whether Claude's article classifications (direction + criticality)
predict DXY price moves at multiple horizons.

Methodology
-----------
1. No session filter  : DXY trades ~23 hours/day so all articles with matched
                        intraday data are included regardless of publish time.
                        Use --ny-session to restrict to 8am–5pm ET if desired.
2. Deduplication      : keep the earliest article per (event_name, date) to
                        avoid inflating counts when the same event generates
                        multiple CNBC articles on the same day
3. Null distribution  : SD thresholds derived from ALL articles with matched
                        intraday data — moves below 1 SD are treated as noise
4. Three evaluations  :
     a. Raw directional accuracy (hit rate, no threshold)
     b. SD-gated accuracy — only count rows where |move| > threshold
     c. Signal rate      — % of critical vs irrelevant articles that produce
                           a statistically significant move

Input
-----
data/dxy_training/claude/output_mgd_try.csv   (output of IntradayMapping.ipynb)

Usage
-----
    python eval_signal.py
    python eval_signal.py --input path/to/other.csv
    python eval_signal.py --no-dedup             # skip event deduplication
    python eval_signal.py --ny-session           # restrict to 8am–5pm ET only
"""

import argparse
import sys
import pytz
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "data/dxy_training/claude/output_mgd_try.csv"
HORIZONS      = ["pct_1h", "pct_2h", "pct_4h", "pct_1d"]
ET            = pytz.timezone("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["pub_utc"] = pd.to_datetime(df["article_published_utc"], utc=True, errors="coerce")
    df = df[df["pub_utc"].notna()].copy()

    df["pub_et"]   = df["pub_utc"].dt.tz_convert(ET)
    df["hour_et"]  = df["pub_et"].dt.hour + df["pub_et"].dt.minute / 60
    df["date_et"]  = df["pub_et"].dt.date.astype(str)
    df["ny_session"] = (df["hour_et"] >= 8) & (df["hour_et"] < 17)

    return df


def dedup_by_event_date(df: pd.DataFrame) -> pd.DataFrame:
    """Keep earliest article per (event_name, date_et) to avoid counting
    the same macro event multiple times on the same day."""
    before = len(df)
    df = (
        df.sort_values("pub_et")
          .drop_duplicates(subset=["event_name", "date_et"], keep="first")
    )
    print(f"  Dedup: {before} → {len(df)} rows "
          f"(removed {before - len(df)} same-event duplicates)")
    return df


def null_distribution(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-horizon SD from all matched articles (not just critical)."""
    return {h: df[h].dropna().std() for h in HORIZONS}


def directional_accuracy(subset: pd.DataFrame, horizon: str) -> tuple[int, int]:
    """Returns (hits, n) for raw direction match."""
    sub = subset[subset[horizon].notna()].copy()
    sub["actual"] = sub[horizon].apply(lambda x: "up" if x > 0 else "down")
    valid = sub[sub["direction"].isin(["up", "down"])]
    hits  = (valid["direction"] == valid["actual"]).sum()
    return int(hits), len(valid)


def sd_gated_accuracy(subset: pd.DataFrame, horizon: str,
                      sd: float) -> tuple[int, int, int]:
    """Returns (hits, signal_n, total_n) where signal = |move| > sd."""
    sub = subset[subset[horizon].notna() & subset["direction"].isin(["up", "down"])].copy()
    sub["actual"] = sub[horizon].apply(lambda x: "up" if x > 0 else "down")
    signal = sub[sub[horizon].abs() > sd]
    hits   = (signal["direction"] == signal["actual"]).sum()
    return int(hits), len(signal), len(sub)


def signal_rate(subset: pd.DataFrame, horizon: str, sd: float) -> float:
    """% of articles in subset producing |move| > sd."""
    col = subset[horizon].dropna()
    if len(col) == 0:
        return float("nan")
    return (col.abs() > sd).mean() * 100


def sep(char="=", width=68):
    print(char * width)


# ── Report sections ───────────────────────────────────────────────────────────

def print_null_distribution(sd_map: dict, df: pd.DataFrame):
    sep()
    print("NULL DISTRIBUTION  (all articles with matched pct data)")
    sep()
    print(f"  {'Horizon':<8}  {'n':>3}  {'mean':>9}  {'sd':>8}  "
          f"{'1SD range':>22}  {'2SD range':>22}")
    print("  " + "-" * 76)
    for h in HORIZONS:
        col  = df[h].dropna()
        mu   = col.mean()
        sd   = sd_map[h]
        r1   = f"[{mu-sd:+.4f}%, {mu+sd:+.4f}%]"
        r2   = f"[{mu-2*sd:+.4f}%, {mu+2*sd:+.4f}%]"
        print(f"  {h:<8}  {len(col):>3}  {mu:>+9.5f}%  {sd:>7.5f}%  "
              f"{r1:>22}  {r2:>22}")
    print()


def print_raw_accuracy(critical_ny: pd.DataFrame, session_label: str):
    sep()
    print(f"RAW DIRECTIONAL ACCURACY  (critical{session_label}, no SD threshold)")
    sep()
    print(f"  {'Horizon':<10} {'n':>4}  {'Hits':>5}  {'Hit Rate':>9}  {'Mean |move|':>12}")
    print("  " + "-" * 46)
    for h in HORIZONS:
        hits, n = directional_accuracy(critical_ny, h)
        sub     = critical_ny[critical_ny[h].notna() & critical_ny["direction"].isin(["up","down"])]
        mean_abs = sub[h].abs().mean() if len(sub) else float("nan")
        pct = f"{hits/n*100:.1f}%" if n else "n/a"
        print(f"  {h:<10} {n:>4}  {hits:>5}  {pct:>9}  {mean_abs:>11.4f}%")
    print()


def print_sd_accuracy(critical_ny: pd.DataFrame, sd_map: dict, session_label: str):
    for multiplier, label in [(1, "1 SD"), (2, "2 SD")]:
        sep()
        print(f"SD-GATED ACCURACY @ {label}  (critical{session_label}, |move| > threshold)")
        sep()
        print(f"  {'Horizon':<10} {'SD thresh':>10}  {'n_total':>8}  "
              f"{'n_signal':>9}  {'signal%':>8}  {'dir_acc':>8}")
        print("  " + "-" * 60)
        for h in HORIZONS:
            sd            = sd_map[h] * multiplier
            hits, sig_n, total_n = sd_gated_accuracy(critical_ny, h, sd)
            sig_pct       = f"{sig_n/total_n*100:.0f}%" if total_n else "n/a"
            dir_acc       = f"{hits/sig_n*100:.1f}% ({hits}/{sig_n})" if sig_n else "n/a"
            print(f"  {h:<10} {sd:>9.4f}%  {total_n:>8}  {sig_n:>9}  "
                  f"{sig_pct:>8}  {dir_acc:>8}")
        print()


def print_signal_rate_comparison(critical_ny: pd.DataFrame,
                                 irrelevant_ny: pd.DataFrame,
                                 sd_map: dict, session_label: str):
    sep()
    print(f"SIGNAL RATE  —  critical vs irrelevant{session_label}")
    print("(% of articles producing |move| > threshold)")
    sep()
    print(f"  {'Horizon':<10}  {'1SD crit':>9}  {'1SD irrel':>10}  "
          f"{'2SD crit':>9}  {'2SD irrel':>10}")
    print("  " + "-" * 52)
    for h in HORIZONS:
        c1 = signal_rate(critical_ny,   h, sd_map[h])
        i1 = signal_rate(irrelevant_ny, h, sd_map[h])
        c2 = signal_rate(critical_ny,   h, sd_map[h] * 2)
        i2 = signal_rate(irrelevant_ny, h, sd_map[h] * 2)
        def fmt(v): return f"{v:.1f}% (n={int(critical_ny[h].notna().sum())})" if h == h else f"{v:.1f}%"
        nc = critical_ny[h].notna().sum()
        ni = irrelevant_ny[h].notna().sum()
        print(f"  {h:<10}  {c1:>7.1f}%   {i1:>8.1f}%   {c2:>7.1f}%   {i2:>8.1f}%"
              f"   (n_crit={nc}, n_irrel={ni})")
    print()


def print_magnitude_comparison(critical_ny: pd.DataFrame,
                                irrelevant_ny: pd.DataFrame, session_label: str):
    sep()
    print(f"AVERAGE MOVE MAGNITUDE  —  critical vs irrelevant{session_label}")
    sep()
    print(f"  {'Horizon':<10}  {'crit mean|move|':>16}  {'irrel mean|move|':>17}  {'ratio':>6}")
    print("  " + "-" * 56)
    for h in HORIZONS:
        c = critical_ny[h].dropna().abs().mean()
        i = irrelevant_ny[h].dropna().abs().mean()
        ratio = f"{c/i:.2f}x" if i else "n/a"
        print(f"  {h:<10}  {c:>15.4f}%  {i:>16.4f}%  {ratio:>6}")
    print()


def print_per_article_detail(critical_ny: pd.DataFrame, sd_map: dict, session_label: str):
    sep()
    print(f"PER-ARTICLE DETAIL  (critical{session_label}, sorted by publish time)")
    sep()

    h1, h4 = "pct_1h", "pct_4h"
    sd1, sd4 = sd_map[h1], sd_map[h4]

    def flag(move, pred, sd):
        if pd.isna(move): return "N/A  "
        actual = "up" if move > 0 else "down"
        sig    = abs(move) > sd
        match  = actual == pred
        if not sig:   return "noise"
        return "HIT  " if match else "MISS "

    print(f"  {'pub_et':<17} {'pred':<5} {'pct_1h':>8}  {'1h':>6}  "
          f"{'pct_4h':>8}  {'4h':>6}  event")
    print("  " + "-" * 80)
    for _, row in critical_ny.sort_values("pub_et").iterrows():
        if not row["direction"] in ("up", "down"):
            continue
        pub  = row["pub_et"].strftime("%m-%d %H:%M")
        p1h  = f"{row[h1]:+.4f}%" if pd.notna(row[h1]) else "    N/A"
        p4h  = f"{row[h4]:+.4f}%" if pd.notna(row[h4]) else "    N/A"
        f1h  = flag(row[h1], row["direction"], sd1)
        f4h  = flag(row[h4], row["direction"], sd4)
        evt  = str(row.get("event_name", ""))[:35]
        print(f"  {pub:<17} {row['direction']:<5} {p1h:>8}  {f1h:<6}  "
              f"{p4h:>8}  {f4h:<6}  {evt}")
    print()


def print_high_criticality_magnitude(high: pd.DataFrame, medium: pd.DataFrame,
                                      irrelevant: pd.DataFrame, sd_map: dict,
                                      session_label: str):
    """
    Evaluate whether high criticality correctly predicts large moves,
    independent of direction. Useful when direction is uncertain but
    knowing a big move is coming still has risk-management value.
    """
    sep()
    print(f"HIGH CRITICALITY — MAGNITUDE ACCURACY{session_label}")
    print("(Does tagging an article 'high' reliably predict a large |move|, regardless of direction?)")
    sep()

    def magnitude_stats(subset: pd.DataFrame, horizon: str, sd: float):
        col = subset[horizon].dropna().abs()
        if len(col) == 0:
            return float("nan"), float("nan"), float("nan"), 0
        pct_1sd = (col > sd).mean() * 100
        pct_2sd = (col > sd * 2).mean() * 100
        return col.mean(), pct_1sd, pct_2sd, len(col)

    print(f"  {'Horizon':<8}  {'group':<10}  {'n':>4}  {'mean|move|':>11}  "
          f"{'%>1SD':>7}  {'%>2SD':>7}")
    print("  " + "-" * 58)
    for h in HORIZONS:
        sd = sd_map[h]
        for label, subset in [("high", high), ("medium", medium), ("irrelevant", irrelevant)]:
            mu, p1, p2, n = magnitude_stats(subset, h, sd)
            if n == 0:
                print(f"  {h:<8}  {label:<10}  {'0':>4}  {'n/a':>11}  {'n/a':>7}  {'n/a':>7}")
            else:
                print(f"  {h:<8}  {label:<10}  {n:>4}  {mu:>10.4f}%  {p1:>6.1f}%  {p2:>6.1f}%")
        print()


def print_summary(critical_ny: pd.DataFrame, sd_map: dict):
    sep()
    print("SUMMARY TABLE")
    sep()
    print(f"  {'Horizon':<10} {'n':>4}  {'Raw acc':>8}  "
          f"{'@1SD acc':>9}  {'@2SD acc':>9}  {'SD(1SD)':>8}")
    print("  " + "-" * 58)
    for h in HORIZONS:
        raw_hits, raw_n      = directional_accuracy(critical_ny, h)
        hits1, sig1, total1  = sd_gated_accuracy(critical_ny, h, sd_map[h])
        hits2, sig2, total2  = sd_gated_accuracy(critical_ny, h, sd_map[h] * 2)
        raw_acc = f"{raw_hits/raw_n*100:.1f}%" if raw_n else "n/a"
        acc1    = f"{hits1/sig1*100:.1f}% ({hits1}/{sig1})" if sig1 else "n/a (0 sig)"
        acc2    = f"{hits2/sig2*100:.1f}% ({hits2}/{sig2})" if sig2 else "n/a (0 sig)"
        print(f"  {h:<10} {raw_n:>4}  {raw_acc:>8}  {acc1:>9}  {acc2:>9}  "
              f"{sd_map[h]:>7.4f}%")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM DXY signal")
    parser.add_argument("--input",      default=DEFAULT_INPUT,
                        help="Path to merged output CSV")
    parser.add_argument("--no-dedup",   action="store_true",
                        help="Skip event deduplication")
    parser.add_argument("--ny-session", action="store_true",
                        help="Restrict to articles published 8am–5pm ET only")
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    df = load_and_prepare(args.input)

    has_pct = df["pct_1h"].notna().sum()
    print(f"  Total rows:              {len(df)}")
    print(f"  With matched pct_1h:     {has_pct}")
    print(f"  is_critical:             {df['is_critical'].astype(bool).sum()}")
    print(f"  Critical + has pct_1h:   "
          f"{(df['is_critical'].astype(bool) & df['pct_1h'].notna()).sum()}")
    if args.ny_session:
        print(f"  NY session (8am–5pm ET): {df['ny_session'].sum()}")
    print()

    # ── Null distribution from all matched rows ───────────────────────────────
    sd_map = null_distribution(df)

    # ── Session label for print headers ──────────────────────────────────────
    session_label = ", NY session" if args.ny_session else ""

    # ── Critical subset ───────────────────────────────────────────────────────
    critical = df[
        df["is_critical"].astype(bool) &
        df["direction"].isin(["up", "down"])
    ].copy()
    if args.ny_session:
        critical = critical[critical["ny_session"]].copy()

    if not args.no_dedup:
        print("Deduplicating critical articles by (event_name, date_et)...")
        critical = dedup_by_event_date(critical)

    # ── Irrelevant subset ─────────────────────────────────────────────────────
    irrelevant = df[~df["is_relevant"].astype(bool)].copy()
    if args.ny_session:
        irrelevant = irrelevant[irrelevant["ny_session"]].copy()

    # ── High / medium subsets (direction-agnostic) ────────────────────────────
    high   = df[df["criticality_level"] == "high"].copy()
    medium = df[df["criticality_level"] == "medium"].copy()
    if args.ny_session:
        high   = high[high["ny_session"]].copy()
        medium = medium[medium["ny_session"]].copy()

    print(f"  Critical (after dedup):  {len(critical)}")
    print(f"  High criticality:        {len(high)}")
    print(f"  Medium criticality:      {len(medium)}")
    print(f"  Irrelevant:              {len(irrelevant)}")
    print()

    # ── Reports ───────────────────────────────────────────────────────────────
    print_null_distribution(sd_map, df)
    print_raw_accuracy(critical, session_label)
    print_sd_accuracy(critical, sd_map, session_label)
    print_signal_rate_comparison(critical, irrelevant, sd_map, session_label)
    print_magnitude_comparison(critical, irrelevant, session_label)
    print_high_criticality_magnitude(high, medium, irrelevant, sd_map, session_label)
    print_per_article_detail(critical, sd_map, session_label)
    print_summary(critical, sd_map)


if __name__ == "__main__":
    main()
