"""
NASA FIRMS Active Fire Ingestion Module
=======================================

HOW TO RUN:
    Run directly to fetch recent thermal hotspots for a bounding box:
        python -m data_ingestion.fetch_firms --days 7 --output data/firms_raw.csv
    Or with custom bounding box:
        python -m data_ingestion.fetch_firms --west 72.5 --south 21.0 --east 73.5 --north 22.0 --days 7

INPUTS & ENVIRONMENT:
    - FIRMS_MAP_KEY: NASA FIRMS Map Key in environment or .env file.
    - Bounding box coordinates (west, south, east, north) and day range (1-10 days per batch).

OUTPUT:
    - Pandas DataFrame / CSV containing VIIRS NRT columns:
      [latitude, longitude, bright_ti4, scan, track, acq_date, acq_time, satellite, confidence, version, bright_ti5, frp, daynight]

STRICT RULES OBSERVED:
    - Uses Area API endpoint pattern: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/{day_range}
    - Batches queries into weekly increments (<= 7-10 days) to prevent hitting the 5,000 transaction rate limit.
    - Gracefully handles missing API keys with realistic fallback generation for development/testing environments.
"""

import os
import sys
import io
import time
import itertools
import argparse
from datetime import datetime, timedelta
from typing import Optional, List
import requests
import pandas as pd
import numpy as np

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    FIRMS_MAP_KEY,
    FIRMS_MAP_KEYS,
    FIRMS_SOURCE_ARCHIVE,
    FIRMS_NRT_START_DATE,
    FIRMS_BASE_URL,
    FIRMS_SOURCE,
    DEFAULT_BBOX,
    DEFAULT_REGION,
    REGIONS,
    get_region_bbox,
    DATA_DIR,
)

# Required schema columns from VIIRS NRT
REQUIRED_FIRMS_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
]


_KEY_CURSOR = itertools.count()


def next_map_key() -> str:
    """Round-robin across the configured MAP_KEY pool. Each key has its own
    5000/10-min budget; rotation also survives one key being throttled or revoked."""
    pool = FIRMS_MAP_KEYS or [FIRMS_MAP_KEY]
    return pool[next(_KEY_CURSOR) % len(pool)]


def select_sensor(date_str: Optional[str] = None) -> str:
    """NRT only serves recent dates and SP only serves older ones; a request outside a
    sensor's window returns an empty CSV rather than an error. Pick by date so historical
    pulls do not silently come back empty."""
    if not date_str:
        return FIRMS_SOURCE
    try:
        return FIRMS_SOURCE if date_str >= FIRMS_NRT_START_DATE else FIRMS_SOURCE_ARCHIVE
    except TypeError:
        return FIRMS_SOURCE


def normalise_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """VIIRS reports confidence as a letter, MODIS as 0-100. Downstream code assumes a
    number: persistence_log does float(row["confidence"]), api/schemas types it as float,
    and OUTPUT_SCHEMA documents "0 - 100". On real VIIRS data that raises
    `ValueError: could not convert string to float: 'n'`.

    It never surfaced because the synthetic generator emits uniform(75, 100), so every
    run before this one exercised numbers that real FIRMS does not return.

    Mapped to the midpoints of the confidence bands FIRMS documents for VIIRS.
    """
    if "confidence" not in df.columns:
        return df
    out = df.copy()
    letters = {"l": 30.0, "n": 70.0, "h": 95.0}
    out["confidence"] = (
        pd.to_numeric(
            out["confidence"].astype(str).str.strip().str.lower().map(letters).fillna(
                pd.to_numeric(out["confidence"], errors="coerce")),
            errors="coerce")
        .fillna(70.0)
        .astype(float)
    )
    return out


def fetch_firms_batch(
    map_key: str,
    west: float,
    south: float,
    east: float,
    north: float,
    day_range: int = 7,
    date_str: Optional[str] = None,
    sensor: str = FIRMS_SOURCE,
) -> pd.DataFrame:
    """
    Pulls a single batch of FIRMS hotspot CSV data for a bounding box.

    Args:
        map_key: NASA FIRMS 32-character API key.
        west, south, east, north: Bounding box coordinates in degrees.
        day_range: Number of days to pull (1 to 10 max per FIRMS Area API call).
        date_str: Optional reference date in 'YYYY-MM-DD' format.
        sensor: FIRMS sensor identifier (default: VIIRS_SNPP_NRT).

    Returns:
        DataFrame of active fire hotspot detections.
    """
    # The Area API rejects anything above 5: 'Invalid day range. Expects [1..5].'
    day_range = min(max(int(day_range), 1), 5)
    bbox_str = f"{west},{south},{east},{north}"

    if date_str:
        url = f"{FIRMS_BASE_URL}/{map_key}/{sensor}/{bbox_str}/{day_range}/{date_str}"
    else:
        url = f"{FIRMS_BASE_URL}/{map_key}/{sensor}/{bbox_str}/{day_range}"

    print(f"[FIRMS] Fetching {day_range} days for bbox [{bbox_str}] (Sensor: {sensor})...")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            content = response.text.strip()
            # Handle FIRMS textual error responses
            if "Invalid MAP_KEY" in content or "No fire detected" in content or len(content) == 0:
                print(f"[FIRMS] Response message: {content[:120]}")
                return pd.DataFrame(columns=REQUIRED_FIRMS_COLUMNS)
            
            df = pd.read_csv(io.StringIO(content))
            # Standardize column names
            df.columns = [c.strip().lower() for c in df.columns]
            for col in REQUIRED_FIRMS_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return normalise_confidence(df[REQUIRED_FIRMS_COLUMNS])
        else:
            print(f"[FIRMS] HTTP Error {response.status_code}: {response.text[:200]}")
            return pd.DataFrame(columns=REQUIRED_FIRMS_COLUMNS)
    except Exception as e:
        print(f"[FIRMS] Request failed: {e}")
        return pd.DataFrame(columns=REQUIRED_FIRMS_COLUMNS)


def generate_synthetic_firms_sample(
    west: float, south: float, east: float, north: float, days: int = 30, n_points: int = 250
) -> pd.DataFrame:
    """
    Generates synthetic FIRMS hotspot data with realistic industrial flare, wildfire,
    and agricultural burn patterns for offline testing when no FIRMS_MAP_KEY is supplied.
    """
    print("[FIRMS] Generating synthetic development hotspot dataset (realistic industrial distributions)...")
    np.random.seed(42)
    records = []
    base_date = datetime.now() - timedelta(days=days)

    # 1. Refinery / Petrochemical Flare anchors (recurrent, stable coordinates)
    flare_anchors = [
        {"lat": south + 0.25 * (north - south), "lon": west + 0.3 * (east - west), "frp_mean": 45.0, "frp_std": 6.0, "type": "flare"},
        {"lat": south + 0.30 * (north - south), "lon": west + 0.35 * (east - west), "frp_mean": 38.0, "frp_std": 5.0, "type": "flare"},
        {"lat": south + 0.70 * (north - south), "lon": west + 0.8 * (east - west), "frp_mean": 65.0, "frp_std": 8.0, "type": "flare"},
        # Solar Farm Anchor (Daytime only, stable, near solar facility)
        {"lat": south + 0.62 * (north - south), "lon": west + 0.62 * (east - west), "frp_mean": 7.5, "frp_std": 0.8, "type": "solar"},
        # Landfill / Waste Stockpile Anchor (Moderate sustained FRP)
        {"lat": south + 0.41 * (north - south), "lon": west + 0.16 * (east - west), "frp_mean": 22.0, "frp_std": 3.0, "type": "stockpile"},
    ]

    # Generate daily/nightly observations for steady facilities
    for day in range(days):
        curr_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        for anchor in flare_anchors:
            anc_type = anchor.get("type", "flare")
            # Recurrence based on type
            recurrence_prob = 0.85 if anc_type in ["flare", "solar"] else 0.45
            if np.random.rand() < recurrence_prob:
                lat_jitter = anchor["lat"] + np.random.normal(0, 0.0008)
                lon_jitter = anchor["lon"] + np.random.normal(0, 0.0008)
                
                if anc_type == "solar":
                    daynight = "D"
                    acq_time = "1330"
                elif anc_type == "stockpile":
                    daynight = "D" if np.random.rand() < 0.6 else "N"
                    acq_time = "1330" if daynight == "D" else "0130"
                else:
                    daynight = "N" if np.random.rand() < 0.55 else "D"
                    acq_time = "0130" if daynight == "N" else "1330"

                frp = max(3.0, np.random.normal(anchor["frp_mean"], anchor["frp_std"]))
                
                records.append({
                    "latitude": round(lat_jitter, 5),
                    "longitude": round(lon_jitter, 5),
                    "bright_ti4": round(320.0 + frp * 1.2, 2),
                    "scan": 0.4,
                    "track": 0.4,
                    "acq_date": curr_date,
                    "acq_time": acq_time,
                    "satellite": "N",
                    "confidence": round(float(np.random.uniform(75, 100)), 1),
                    "version": "2.0NRT",
                    "bright_ti5": round(295.0 + frp * 0.4, 2),
                    "frp": round(frp, 2),
                    "daynight": daynight,
                })

    # Single-instance industrial accident surge
    accident_date = (base_date + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    records.append({
        "latitude": round(south + 0.24 * (north - south), 5),
        "longitude": round(west + 0.29 * (east - west), 5),
        "bright_ti4": 395.0,
        "scan": 0.4,
        "track": 0.4,
        "acq_date": accident_date,
        "acq_time": "0215",
        "satellite": "N",
        "confidence": 98.0,
        "version": "2.0NRT",
        "bright_ti5": 340.0,
        "frp": 125.0,
        "daynight": "N",
    })

    # 2. Random Wildfire & Agricultural burns (sporadic, growing, day-skewed)
    for _ in range(n_points):
        d_offset = np.random.randint(0, days)
        curr_date = (base_date + timedelta(days=d_offset)).strftime("%Y-%m-%d")
        lat = np.random.uniform(south, north)
        lon = np.random.uniform(west, east)
        daynight = "D" if np.random.rand() < 0.85 else "N"
        acq_time = "1345" if daynight == "D" else "0145"
        frp = float(np.random.exponential(scale=18.0) + 3.0)
        
        records.append({
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "bright_ti4": round(310.0 + frp * 1.1, 2),
            "scan": 0.45,
            "track": 0.42,
            "acq_date": curr_date,
            "acq_time": acq_time,
            "satellite": "N",
            "confidence": round(float(np.random.uniform(50, 95)), 1),
            "version": "2.0NRT",
            "bright_ti5": round(290.0 + frp * 0.3, 2),
            "frp": round(frp, 2),
            "daynight": daynight,
        })

    df = pd.DataFrame(records)
    print(f"[FIRMS] Generated {len(df)} synthetic hotspot records.")
    return df


def fetch_firms_hotspots(
    region: Optional[str] = None,
    west: Optional[float] = None,
    south: Optional[float] = None,
    east: Optional[float] = None,
    north: Optional[float] = None,
    total_days: int = 30,
    map_key: Optional[str] = None,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches FIRMS active fire hotspots over a multi-week period, batching by 7-day
    windows to strictly respect the FIRMS Area API transaction rate limits.

    Args:
        region: Named Indian region key (e.g. 'gujarat', 'maharashtra', 'odisha', 'all_india').
        west, south, east, north: Optional explicit BBox coordinate overrides in degrees.
        total_days: Total lookback window in days.
        map_key: NASA FIRMS Map Key (falls back to FIRMS_MAP_KEY environment variable).

    Returns:
        Consolidated DataFrame of all hotspot records.
    """
    # Resolve bounding box coordinates: explicit overrides take precedence, else look up region
    if west is None or south is None or east is None or north is None:
        target_region = region or DEFAULT_REGION
        bbox = get_region_bbox(target_region)
        w = float(west if west is not None else bbox["west"])
        s = float(south if south is not None else bbox["south"])
        e = float(east if east is not None else bbox["east"])
        n = float(north if north is not None else bbox["north"])
    else:
        w, s, e, n = float(west), float(south), float(east), float(north)

    key = map_key or FIRMS_MAP_KEY
    if not key or key.strip() == "" or key.startswith("your_"):
        print(f"[FIRMS] Notice: No valid FIRMS_MAP_KEY found. Utilizing offline synthetic simulation generator for bbox [{w}, {s}, {e}, {n}].")
        return generate_synthetic_firms_sample(w, s, e, n, days=total_days)

    all_dfs: List[pd.DataFrame] = []
    
    # Batch requests in 7-day increments
    # DATE is the START of the window (verified: /5/2026-08-01 returns 08-01..08-05),
    # so batches must walk FORWARD from the oldest date. The previous loop passed
    # datetime.now() for batch 0, which requested five days into the future.
    batch_size = 5
    num_batches = (total_days + batch_size - 1) // batch_size
    # start_date pins an explicit historical window; without it we walk back from today.
    if start_date:
        window_start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        window_start = datetime.now() - timedelta(days=total_days - 1)

    for i in range(num_batches):
        days_in_batch = min(batch_size, total_days - (i * batch_size))
        batch_start = window_start + timedelta(days=i * batch_size)
        date_str = batch_start.strftime("%Y-%m-%d")

        df_batch = fetch_firms_batch(
            map_key=next_map_key(),
            west=w,
            south=s,
            east=e,
            north=n,
            day_range=days_in_batch,
            date_str=date_str,
            sensor=select_sensor(date_str),
        )

        if not df_batch.empty:
            all_dfs.append(df_batch)

        # Rate-limiting courtesy pause
        if i < num_batches - 1:
            time.sleep(0.5)

    if not all_dfs:
        # A configured key that returns nothing is real information ("no fires in this
        # bbox/window"). Fabricating synthetic rows here is how a pipeline ends up
        # trained on invented data with nothing on screen saying so.
        print("[FIRMS] WARNING: key configured but zero detections returned for "
              f"[{w},{s},{e},{n}] over {total_days}d. Returning EMPTY frame, not synthetic.")
        return pd.DataFrame(columns=REQUIRED_FIRMS_COLUMNS)

    consolidated = pd.concat(all_dfs, ignore_index=True)
    consolidated.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"], inplace=True)
    consolidated.reset_index(drop=True, inplace=True)
    print(f"[FIRMS] Successfully retrieved {len(consolidated)} total hotspot observations.")
    return consolidated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NASA FIRMS Thermal Hotspots")
    parser.add_argument("--region", type=str, default=DEFAULT_REGION, help=f"Named Indian industrial region ({', '.join(REGIONS.keys())})")
    parser.add_argument("--west", type=float, default=None, help="Optional raw west bbox override")
    parser.add_argument("--south", type=float, default=None, help="Optional raw south bbox override")
    parser.add_argument("--east", type=float, default=None, help="Optional raw east bbox override")
    parser.add_argument("--north", type=float, default=None, help="Optional raw north bbox override")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-date", dest="start_date", type=str, default=None,
                        help="Window START in YYYY-MM-DD. Omit to walk back from today.")
    parser.add_argument("--output", type=str, default="data/firms_raw.csv")
    args = parser.parse_args()

    df_hotspots = fetch_firms_hotspots(
        region=args.region,
        west=args.west,
        south=args.south,
        east=args.east,
        north=args.north,
        total_days=args.days,
            start_date=args.start_date,
    )
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_hotspots.to_csv(out_path, index=False)
    print(f"[FIRMS] Saved {len(df_hotspots)} rows to {out_path}")
