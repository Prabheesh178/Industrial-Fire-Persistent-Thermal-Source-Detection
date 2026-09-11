"""
Configuration and Settings Module
=================================

HOW TO RUN:
    This module is imported by all pipeline and API components.
    To test configuration loading independently:
        python -m config.settings
    Or from the project root:
        python config/settings.py

ENVIRONMENT VARIABLES / INPUTS:
    - FIRMS_MAP_KEY: NASA FIRMS Map Key (32-character string from NASA EOSDIS).
    - DB_PATH: Path to SQLite database file (default: "data/hotspots.db").
    - DEMO_BBOX_WEST, DEMO_BBOX_SOUTH, DEMO_BBOX_EAST, DEMO_BBOX_NORTH: Default bounding box coordinates.
    - OVERPASS_URL: Overpass API interpreter URL.
    - OSM_CACHE_PATH: Path to cached OSM GeoJSON/JSON extract.

OUTPUT:
    Exposes typed configuration constants, paths, hazard weights, and model hyperparameters.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Base Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_DIR = BASE_DIR / "db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# NASA FIRMS API Settings
FIRMS_MAP_KEY: str = os.getenv("FIRMS_MAP_KEY", "")

# Optional comma-separated pool of MAP_KEYs. Requests round-robin across them and fail
# over on error. Each key carries its own 5000/10-min budget.
FIRMS_MAP_KEYS: List[str] = [
    k.strip() for k in os.getenv("FIRMS_MAP_KEYS", FIRMS_MAP_KEY).split(",") if k.strip()
]
FIRMS_BASE_URL: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE: str = "VIIRS_SNPP_NRT"

# Verified against GET /api/data_availability/csv/{KEY}/ALL on 2026-09-09:
#   VIIRS_SNPP_SP   2012-01-20 .. 2026-04-27
#   VIIRS_SNPP_NRT  2026-04-28 .. present   (seamless handover, no gap)
# Requesting a date outside a sensor's window returns an empty CSV, not an error,
# so historical pulls silently yielded nothing while the source was pinned to NRT.
FIRMS_SOURCE_ARCHIVE: str = "VIIRS_SNPP_SP"
FIRMS_NRT_START_DATE: str = "2026-04-28"

# =====================================================================
# India Regional Registry (Single Source of Truth for Bounding Boxes)
# =====================================================================

REGIONS: Dict[str, Dict[str, Any]] = {
    "gujarat": {
        "name": "gujarat",
        "label": "Gujarat Industrial Belt (Hazira, Dahej, Jamnagar, Surat)",
        "bbox": {"west": 69.5, "south": 20.5, "east": 73.8, "north": 23.0},
        "description": "Petrochemical refineries, LNG terminals, ports, and heavy chemical clusters in Gujarat.",
    },
    "maharashtra": {
        "name": "maharashtra",
        "label": "Maharashtra Industrial Belt (Mumbai, MMR, Pune, Raigad, Tarapur)",
        "bbox": {"west": 72.6, "south": 18.3, "east": 74.5, "north": 20.0},
        "description": "Chemical corridors, manufacturing MIDCs, and energy installations in Maharashtra.",
    },
    "odisha": {
        "name": "odisha",
        "label": "Odisha Industrial Belt (Angul, Jharsuguda, Paradip, Kalinganagar)",
        "bbox": {"west": 83.5, "south": 19.8, "east": 87.0, "north": 22.2},
        "description": "Steel plants, aluminum smelters, coal mining complexes, and deep-water ports in Odisha.",
    },
    "chhattisgarh_jharkhand": {
        "name": "chhattisgarh_jharkhand",
        "label": "East-Central Mining & Steel Belt (Bhilai, Korba, Dhanbad, Jamshedpur)",
        "bbox": {"west": 81.0, "south": 21.0, "east": 86.8, "north": 24.2},
        "description": "Coal fields, thermal power plants, and integrated steelworks in CG & JH.",
    },
    "all_india": {
        "name": "all_india",
        "label": "All India Coverage",
        "bbox": {"west": 68.0, "south": 6.5, "east": 97.5, "north": 37.5},
        "description": "Nationwide active fire and thermal anomaly coverage across the Indian subcontinent.",
    },
}

# Alias for backwards compatibility or alternative naming
REGIONS["punjab_haryana"] = {
    "name": "punjab_haryana",
    "label": "Punjab & Haryana Agricultural Belt (Ludhiana, Patiala, Karnal, Hisar)",
    "bbox": {"west": 73.8, "south": 29.3, "east": 77.6, "north": 32.6},
    "description": "Paddy and wheat stubble burning belt. Required for agricultural_burn labels.",
}
REGIONS["rajasthan_solar"] = {
    "name": "rajasthan_solar",
    "label": "Bhadla Solar Park & Thar Renewable Zone",
    "bbox": {"west": 71.4, "south": 27.1, "east": 72.4, "north": 28.0},
    "description": "Large solar installations. Required for known_false_positive labels.",
}
REGIONS["gujarat_jamnagar"] = REGIONS["gujarat"]


def get_region_bbox(region_name: Optional[str] = None) -> Dict[str, float]:
    """
    Resolves a region key against the REGIONS registry.
    Returns a copy of the bounding box dict {west, south, east, north}.
    Falls back to the default region ("gujarat") if name is invalid or omitted.
    """
    if not region_name:
        return dict(REGIONS[DEFAULT_REGION]["bbox"])
    
    key = str(region_name).strip().lower()
    if key in REGIONS:
        return dict(REGIONS[key]["bbox"])
    
    # Check if key matches without underscores or hyphens
    normalized_key = key.replace("-", "_").replace(" ", "_")
    if normalized_key in REGIONS:
        return dict(REGIONS[normalized_key]["bbox"])

    print(f"[Settings] Warning: Unknown region '{region_name}'. Falling back to '{DEFAULT_REGION}'.")
    return dict(REGIONS[DEFAULT_REGION]["bbox"])


def get_regions_list() -> List[Dict[str, Any]]:
    """
    Returns list of standard regions for API serialization.
    Filters out duplicate aliases.
    """
    unique_keys = ["gujarat", "maharashtra", "odisha", "chhattisgarh_jharkhand", "all_india"]
    return [REGIONS[k] for k in unique_keys if k in REGIONS]


# Default Active Region
DEFAULT_REGION: str = os.getenv("DEFAULT_REGION", "gujarat")

# Default Bounding Box derived directly from REGIONS registry (Single Source of Truth)
DEFAULT_BBOX: Dict[str, float] = get_region_bbox(DEFAULT_REGION)

# OpenStreetMap / Overpass Settings
OVERPASS_URL: str = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OSM_CACHE_PATH: Path = BASE_DIR / os.getenv("OSM_CACHE_PATH", "data/osm_cache.json")

# Database Path
DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "data/hotspots.db")
SCHEMA_PATH: Path = DB_DIR / "schema.sql"

# Open-Meteo Weather Endpoints
OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

# 7 Standardized Event Type Classes (Exact spec requirement)
EVENT_TYPES: List[str] = [
    "industrial_flare",
    "industrial_accident",
    "stockpile_combustion",
    "known_false_positive",
    "wildfire",
    "agricultural_burn",
    "unknown_source",
]

# Hazard Weight Lookup Table (Exact spec requirement)
HAZARD_WEIGHTS: Dict[str, float] = {
    "wildfire": 1.0,
    "unknown_source": 1.0,
    "industrial_accident": 0.9,
    "stockpile_combustion": 0.6,
    "agricultural_burn": 0.3,
    "industrial_flare": 0.1,
    "known_false_positive": 0.0,
}

# India Agricultural Stubble Burning Season Months (Apr-May, Oct-Nov)
AGRI_BURN_MONTHS: List[int] = [4, 5, 10, 11]

# Spatial & Clustering Hyperparameters
DBSCAN_EPS_METERS: float = 750.0
DBSCAN_TIME_HOURS: float = 12.0
FACILITY_PROXIMITY_THRESHOLD_M: float = 500.0
H3_RESOLUTION: int = 8

# Model A (Classifier) Hyperparameters
# n_jobs=1 is deliberate. Measured on an M4 at 2,000 rows x 19 features:
#   n_jobs=1 -> 1.27s | n_jobs=2 -> 2.21s | n_jobs=4 -> 2.65s | n_jobs=8 -> 6.11s
# Thread sync dominates at this data size; parallelism only pays above ~200k rows.
MODEL_A_PARAMS: Dict = {
    "n_jobs": 1,
    "objective": "multiclass",
    "num_class": len(EVENT_TYPES),
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "random_state": 42,
    "verbose": -1,
}

# Model C (24h Risk Escalation) Hyperparameters
# n_jobs=1 is deliberate. Measured on an M4 at 2,000 rows x 19 features:
#   n_jobs=1 -> 1.27s | n_jobs=2 -> 2.21s | n_jobs=4 -> 2.65s | n_jobs=8 -> 6.11s
# Thread sync dominates at this data size; parallelism only pays above ~200k rows.
MODEL_C_PARAMS: Dict = {
    "n_jobs": 1,
    "objective": "binary",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "random_state": 42,
    "verbose": -1,
}

# API Server Configuration
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", 8000))

if __name__ == "__main__":
    print("=== Configuration Loaded Successfully ===")
    print(f"Base Directory:     {BASE_DIR}")
    print(f"FIRMS Key Defined:  {bool(FIRMS_MAP_KEY)}")
    print(f"Database Path:      {DB_PATH}")
    print(f"OSM Cache Path:     {OSM_CACHE_PATH}")
    print(f"Default BBox:       {DEFAULT_BBOX}")
    print(f"Event Types:        {EVENT_TYPES}")
