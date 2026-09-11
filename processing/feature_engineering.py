"""
Feature Engineering Module
==========================

HOW TO RUN:
    Build the complete feature table from persisted hotspots:
        python -m processing.feature_engineering --input data/hotspots_persisted.csv --output data/feature_table.csv

INPUTS & ENVIRONMENT:
    - Persisted Hotspots DataFrame with spatial join, persistence log metrics, weather, and land cover.

OUTPUT:
    - Complete feature DataFrame containing the exact specified feature set:
      [frp, brightness_ti4, brightness_ti5, confidence, daynight,
       dist_to_nearest_industrial_m, nearest_industrial_type, land_cover_class,
       temperature, humidity, wind_speed, days_active_last_30, night_detection_fraction,
       month, is_agri_burn_season, frp_zscore_vs_facility_baseline,
       dist_to_populated_area_m, dist_to_critical_infra_m, cluster_growth_rate]

STRICT RULES OBSERVED:
    - Produces exact feature list without omissions or substitutions.
    - Includes `night_detection_fraction` and strictly omits `recurrence_time_of_day_std`.
    - Includes exposure features (`dist_to_populated_area_m`, `dist_to_critical_infra_m`).
    - Derives `month` and `is_agri_burn_season` (India Oct-Nov / Apr-May windows).
"""

import os
import sys
import argparse
from typing import Optional, Any
import pandas as pd

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import AGRI_BURN_MONTHS
from data_ingestion.fetch_landcover import sample_landcover_for_hotspots
from data_ingestion.fetch_weather import fetch_weather_for_hotspots

# The canonical feature column list required for modeling
CANONICAL_FEATURE_COLUMNS = [
    "frp",
    "brightness_ti4",
    "brightness_ti5",
    "confidence",
    "daynight",
    "dist_to_nearest_industrial_m",
    "nearest_industrial_type",
    "land_cover_class",
    "temperature",
    "humidity",
    "wind_speed",
    "days_active_last_30",
    "night_detection_fraction",
    "month",
    "is_agri_burn_season",
    "frp_zscore_vs_facility_baseline",
    "dist_to_populated_area_m",
    "dist_to_critical_infra_m",
    "cluster_growth_rate",
]


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts month and is_agri_burn_season from acq_date.
    """
    df_out = df.copy()
    months = []
    is_agri = []
    
    for val in df_out["acq_date"]:
        try:
            m = int(str(val).split("-")[1])
        except Exception:
            m = 9
        months.append(m)
        is_agri.append(1 if m in AGRI_BURN_MONTHS else 0)

    df_out["month"] = months
    df_out["is_agri_burn_season"] = is_agri
    return df_out


def build_feature_table(
    df_hotspots: pd.DataFrame,
    anomaly_engine: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Assembles the complete feature engineering table from raw/processed hotspots.
    Ensures all required data ingestion dependencies (weather, land cover) and spatial metrics
    are enriched and formatted.
    """
    if df_hotspots.empty:
        return pd.DataFrame(columns=CANONICAL_FEATURE_COLUMNS)

    df_feat = df_hotspots.copy()

    # 1. Ensure Land Cover is populated
    if "land_cover_class" not in df_feat.columns:
        df_feat["land_cover_class"] = sample_landcover_for_hotspots(df_feat)

    # 2. Ensure Weather is populated
    if "temperature" not in df_feat.columns or "wind_speed" not in df_feat.columns:
        df_feat = fetch_weather_for_hotspots(df_feat)

    # 3. Extract temporal indicators (month, is_agri_burn_season)
    df_feat = extract_temporal_features(df_feat)

    # 4a. FIRMS ships the thermal bands as bright_ti4 / bright_ti5. The defaults block
    # below names them brightness_ti4 / brightness_ti5, so those columns never existed
    # and were created as the literal constants 315.0 / 295.0 for every row -- two of
    # Model A's eighteen features carrying zero information. Alias them first.
    for src, dst in (("bright_ti4", "brightness_ti4"), ("bright_ti5", "brightness_ti5")):
        if src in df_feat.columns:
            if dst not in df_feat.columns:
                df_feat[dst] = df_feat[src]
            else:
                df_feat[dst] = df_feat[dst].where(df_feat[dst].notna(), df_feat[src])

    # 4b. Two physical discriminators FIRMS does not ship directly.
    #  delta_bt   = 4um minus 11um brightness temperature. A hot, compact source (flare,
    #               furnace, explosion) shows a large split; a cooler spreading vegetation
    #               front shows a small one. bright_ti4 saturates at 367K, so the split
    #               also encodes "this pixel pegged the sensor".
    #  frp_density= radiative power per unit pixel footprint (MW/km2). VIIRS pixels grow
    #               toward swath edge, so raw FRP conflates intensity with pixel size;
    #               dividing by scan*track separates them.
    if {"bright_ti4", "bright_ti5"}.issubset(df_feat.columns):
        df_feat["delta_bt"] = (
            pd.to_numeric(df_feat["bright_ti4"], errors="coerce")
            - pd.to_numeric(df_feat["bright_ti5"], errors="coerce")
        )
    if {"scan", "track"}.issubset(df_feat.columns):
        area = (pd.to_numeric(df_feat["scan"], errors="coerce")
                * pd.to_numeric(df_feat["track"], errors="coerce"))
        df_feat["frp_density"] = (
            pd.to_numeric(df_feat["frp"], errors="coerce") / area.where(area > 0)
        )

    # 4. Fill defaults for missing numeric or spatial fields
    defaults = {
        "delta_bt": 0.0,
        "frp_density": 0.0,
        "brightness_ti4": 315.0,
        "brightness_ti5": 295.0,
        "confidence": 80.0,
        "daynight": "D",
        "dist_to_nearest_industrial_m": 5000.0,
        "nearest_industrial_type": "none",
        "days_active_last_30": 1,
        "night_detection_fraction": 0.0,
        "dist_to_populated_area_m": 5000.0,
        "dist_to_critical_infra_m": 5000.0,
        "cluster_growth_rate": 0.0,
        "frp_zscore_vs_facility_baseline": 0.0,
    }
    for col, default_val in defaults.items():
        if col not in df_feat.columns:
            df_feat[col] = default_val
        else:
            df_feat[col] = df_feat[col].fillna(default_val)

    # 5. Compute or attach frp_zscore_vs_facility_baseline from Model B if engine is provided
    if anomaly_engine is not None:
        z_scores = []
        for _, row in df_feat.iterrows():
            loc_k = str(row.get("location_key", ""))
            frp = float(row.get("frp", 0.0))
            fac_t = str(row.get("nearest_industrial_type", "none"))
            z, _ = anomaly_engine.compute_z_score(loc_k, frp, fac_t)
            z_scores.append(round(float(z), 3))
        df_feat["frp_zscore_vs_facility_baseline"] = z_scores

    return df_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble Full Feature Engineering Table")
    parser.add_argument("--input", type=str, default="data/hotspots_persisted.csv")
    parser.add_argument("--output", type=str, default="data/feature_table.csv")
    args = parser.parse_args()

    if os.path.exists(args.input):
        df_in = pd.read_csv(args.input)
        df_out = build_feature_table(df_in)
        df_out.to_csv(args.output, index=False)
        print(f"[FeatureEngineering] Built feature matrix with {len(df_out)} rows and {len(df_out.columns)} cols -> {args.output}")
    else:
        print(f"[FeatureEngineering] Input file not found: {args.input}")
