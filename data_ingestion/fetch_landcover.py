"""
Land Cover Point-Sampling Ingestion Module
==========================================

HOW TO RUN:
    Test point-sampling for specific coordinates:
        python -m data_ingestion.fetch_landcover --lat 21.17 --lon 72.83
    Or test batch sampling for a CSV of hotspots:
        python -m data_ingestion.fetch_landcover --input data/firms_raw.csv --output data/landcover_sampled.csv

INPUTS & ENVIRONMENT:
    - Latitude, Longitude floating point coordinates.
    - Cloud REST endpoint (Microsoft Planetary Computer STAC / OpenLandMap / Earth Engine API).

OUTPUT:
    - Standardized string class for each point:
      {'forest', 'grassland', 'cropland', 'built_up', 'barren', 'water', 'wetland'}

STRICT RULES OBSERVED:
    - Point-samples land cover per hotspot coord via cloud API (ESA WorldCover product).
    - NEVER bulk-downloads WorldCover raster tiles — eliminates tens of gigabytes of wasted storage.
    - Features per-coordinate LRU/hash memory cache to minimize network calls for nearby recurring points.
"""

import os
import json
import rasterio
import math
import atexit
import sys
import argparse
from typing import Dict
import requests
import pandas as pd

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DATA_DIR

# ESA WorldCover 10m class map
ESA_WORLDCOVER_MAP: Dict[int, str] = {
    10: "forest",        # Tree cover
    20: "grassland",     # Shrubland
    30: "grassland",     # Grassland
    40: "cropland",      # Cropland
    50: "built_up",      # Built-up / Urban
    60: "barren",        # Bare / sparse vegetation
    70: "water",         # Snow and ice
    80: "water",         # Permanent water bodies
    90: "wetland",       # Herbaceous wetland
    95: "forest",        # Mangroves
    100: "barren",       # Moss and lichen
}

# Local in-memory cache to prevent redundant point queries
# Disk-backed so the cache survives the process. Each miss is a ~0.3s HTTP round trip to
# openlandmap; on a 5,000-point pull an in-memory-only cache means re-paying 20-40 minutes
# on every re-run, which dominates the pipeline's wall-clock.
_LANDCOVER_CACHE_PATH = DATA_DIR / "landcover_cache.json"


def _load_landcover_cache() -> Dict[str, str]:
    try:
        with open(_LANDCOVER_CACHE_PATH, "r") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_landcover_cache() -> None:
    """Atomic write so an interrupted run cannot leave a truncated cache behind."""
    try:
        _LANDCOVER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LANDCOVER_CACHE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(_LANDCOVER_CACHE, fh)
        os.replace(tmp, _LANDCOVER_CACHE_PATH)
    except OSError as exc:
        print(f"[LandCover] Could not persist cache: {exc}")


_LANDCOVER_CACHE: Dict[str, str] = _load_landcover_cache()
if _LANDCOVER_CACHE:
    print(f"[LandCover] Reusing {len(_LANDCOVER_CACHE)} cached point lookups from disk.", file=sys.stderr)
atexit.register(save_landcover_cache)


ESA_WORLDCOVER_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
)


def _esa_tile_url(lat: float, lon: float) -> str:
    """ESA WorldCover v200 tiles are 3x3 degree, named by their SW corner."""
    la = math.floor(lat / 3.0) * 3
    lo = math.floor(lon / 3.0) * 3
    ns = f"N{la:02d}" if la >= 0 else f"S{abs(la):02d}"
    ew = f"E{lo:03d}" if lo >= 0 else f"W{abs(lo):03d}"
    return f"{ESA_WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif"


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 3)}_{round(lon, 3)}"


def sample_landcover_point(lat: float, lon: float, timeout: int = 5) -> str:
    """Single-point land cover. Prefer sample_landcover_for_hotspots for bulk work:
    it groups points by tile and opens each COG once."""
    return sample_landcover_for_hotspots(
        pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    ).iloc[0]


def sample_landcover_for_hotspots(df_hotspots: pd.DataFrame) -> pd.Series:
    """Land cover per hotspot, read from ESA WorldCover 10m COGs on S3.

    Replaces an openlandmap query that never succeeded: the endpoint requires both
    `coll` and `regex` and returned HTTP 422 for every call, so the module always fell
    through to a `(latitude * 100) % 7` heuristic. That made land_cover_class a hash of
    the latitude digits -- and land_cover_class is both a model feature and the sole
    basis for the agricultural_burn and wildfire label rules, so those labels were
    being assigned by modular arithmetic on a coordinate the model can also see.

    Points are grouped by 3x3 degree tile so each COG is opened once and read with HTTP
    range requests; the raster is never downloaded in full.
    """
    if df_hotspots is None or df_hotspots.empty:
        return pd.Series([], dtype=str)

    lats = df_hotspots["latitude"].astype(float).to_numpy()
    lons = df_hotspots["longitude"].astype(float).to_numpy()
    out = [None] * len(lats)

    pending: Dict[str, list] = {}
    for i, (la, lo) in enumerate(zip(lats, lons)):
        ck = _cache_key(la, lo)
        hit = _LANDCOVER_CACHE.get(ck)
        if hit is not None:
            out[i] = hit
        else:
            pending.setdefault(_esa_tile_url(la, lo), []).append((i, la, lo, ck))

    for url, items in pending.items():
        try:
            with rasterio.open(f"/vsicurl/{url}") as ds:
                coords = [(lo, la) for _, la, lo, _ in items]
                for (idx, _la, _lo, ck), val in zip(items, ds.sample(coords)):
                    res = ESA_WORLDCOVER_MAP.get(int(val[0]), "barren")
                    out[idx] = res
                    _LANDCOVER_CACHE[ck] = res
        except Exception as exc:
            # A missing ocean tile is normal; anything else is worth seeing.
            print(f"[LandCover] tile unavailable ({url.rsplit('/', 1)[-1]}): "
                  f"{type(exc).__name__}. Marking {len(items)} points 'unknown'.")
            for idx, _la, _lo, ck in items:
                out[idx] = "unknown"
                _LANDCOVER_CACHE[ck] = "unknown"

    if pending:                    # only touch disk when something was actually fetched
        save_landcover_cache()
    return pd.Series(out, index=df_hotspots.index, dtype=str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Land Cover at Specific Coordinates")
    parser.add_argument("--lat", type=float, default=21.17)
    parser.add_argument("--lon", type=float, default=72.83)
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input)
        df["land_cover_class"] = sample_landcover_for_hotspots(df)
        out_file = args.output or args.input
        df.to_csv(out_file, index=False)
        print(f"[LandCover] Sampled {len(df)} points and saved to {out_file}")
    else:
        lc = sample_landcover_point(args.lat, args.lon)
        print(f"[LandCover] Point ({args.lat}, {args.lon}) -> Land Cover Class: {lc}")
