"""Build a multi-region, multi-season FIRMS corpus for training.

Recent data alone yields 5 of the 7 label classes: agricultural_burn needs the
Oct-Nov stubble window and known_false_positive needs a solar park in bbox.

Writes two files:
    data/firms_corpus.csv            every unique detection pulled
    data/firms_corpus_balanced.csv   capped per pull; what tools/run_training.py reads

    python tools/build_corpus.py                 # fetch from FIRMS, then balance
    python tools/build_corpus.py --balance-only  # re-balance an existing firms_corpus.csv
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))   # data/ paths are repo-root relative
import argparse
from dotenv import load_dotenv; load_dotenv(".env")
import pandas as pd

PULLS = [
    # (label,                region,                   days, start_date)
    ("industrial-gujarat",   "gujarat",                  30, None),
    ("industrial-odisha",    "odisha",                   30, None),
    ("industrial-cgjh",      "chhattisgarh_jharkhand",   30, None),
    ("solar-bhadla",         "rajasthan_solar",          30, None),
    ("agri-punjab-2025",     "punjab_haryana",           15, "2025-10-28"),
    ("wildfire-odisha",      "odisha",                   20, "2026-03-01"),
    ("wildfire-cgjh",        "chhattisgarh_jharkhand",   20, "2026-03-01"),
]

# The March fire-season pulls return ~8-13k detections each and would swamp every other
# class. Capping them also bounds the clustering step, whose distance matrix is n^2 in
# memory: ~9k rows is ~340 MB, the full 22k would be ~2 GB.
CAPS = {"wildfire-odisha": 2500, "wildfire-cgjh": 2500, "agri-punjab-2025": 3000}
SEED = 42

RAW = "data/firms_corpus.csv"
BALANCED = "data/firms_corpus_balanced.csv"


def balance(corpus: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-pull cap. Same input + same SEED -> identical rows, same order."""
    parts = [g.sample(min(len(g), CAPS.get(pull, len(g))), random_state=SEED)
             for pull, g in corpus.groupby("pull")]
    return pd.concat(parts, ignore_index=True)


def fetch() -> pd.DataFrame:
    from data_ingestion.fetch_firms import fetch_firms_hotspots
    frames = []
    for label, region, days, start in PULLS:
        df = fetch_firms_hotspots(region=region, total_days=days, start_date=start)
        print(f"  {label:22s} {region:24s} {days:3d}d  start={start or 'recent':10s} -> {len(df):6,} rows")
        if len(df):
            df = df.copy(); df["pull"] = label
            frames.append(df)
    if not frames:
        sys.exit("No detections returned for any pull. Check FIRMS_MAP_KEY in .env.")
    corpus = pd.concat(frames, ignore_index=True)
    corpus.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"], inplace=True)
    corpus.reset_index(drop=True, inplace=True)
    corpus.to_csv(RAW, index=False)
    print(f"\nCORPUS: {len(corpus):,} unique rows -> {RAW}")
    return corpus


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--balance-only", action="store_true",
                    help=f"skip the FIRMS fetch and re-balance an existing {RAW}")
    args = ap.parse_args()

    if args.balance_only:
        if not os.path.exists(RAW):
            sys.exit(f"{RAW} not found - run without --balance-only first.")
        corpus = pd.read_csv(RAW)
    else:
        os.makedirs("data", exist_ok=True)
        corpus = fetch()

    bal = balance(corpus)
    bal.to_csv(BALANCED, index=False)
    print(f"BALANCED: {len(bal):,} rows -> {BALANCED}   "
          f"(clustering matrix ~{len(bal) ** 2 * 4 / 1e6:.0f} MB)")
    print(bal["pull"].value_counts().to_string())
