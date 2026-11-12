"""
End-to-End Training & Pipeline Orchestrator
===========================================

HOW TO RUN:
    Run the full end-to-end training and data pipeline from the repository root:
        python -m models.train_all
    Or with a specific Indian region:
        python -m models.train_all --region gujarat --days 30
    Or with custom bounding box coordinates:
        python -m models.train_all --days 30 --west 72.5 --south 21.0 --east 73.5 --north 22.0

INPUTS & ENVIRONMENT:
    - FIRMS_MAP_KEY: NASA FIRMS Map Key in .env (falls back to synthetic generator if omitted).
    - DEFAULT_REGION / REGIONS or CLI arguments for target region coordinates.

OUTPUT:
    - Populates `data/hotspots.db` SQLite database with persistence history.
    - Saves engineered feature snapshot to `data/feature_table.csv` and `data/labeled_dataset.csv`.
    - Saves trained model artifacts in `artifacts/`:
      * `artifacts/model_a.joblib` (LightGBM 7-class classifier + SHAP explainer)
      * `artifacts/facility_baselines.json` (Statistical Median + MAD baseline table)
      * `artifacts/model_c.joblib` (LightGBM 24h risk escalation model)

STRICT PIPELINE EXECUTION SEQUENCE OBSERVED:
    Step 1: fetch_firms.py + fetch_osm.py -> spatial_join.py (boundary distance, cached OSM extract)
    Step 2: clustering.py (assign event_id BEFORE feature engineering)
    Step 3: persistence_log.py (build/update hotspot_history table)
    Step 4: feature_engineering.py (full canonical feature table)
    Step 5: labeling_heuristic.py (generate 7-class pseudo-labels)
    Step 6: Train Model A, fit Model B baselines, train Model C (with duty-cycle filter)
    Step 7: Validate Score Combiner on test events
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    BASE_DIR,
    DEFAULT_BBOX,
    DEFAULT_REGION,
    REGIONS,
    get_region_bbox,
    FIRMS_MAP_KEY,
    FIRMS_MAP_KEYS,
    DB_PATH,
    ARTIFACTS_DIR,
    DATA_DIR,
    DB_DIR,
    OSM_CACHE_PATH,
)
from data_ingestion.fetch_firms import fetch_firms_hotspots
from data_ingestion.fetch_osm import fetch_osm_industrial_and_exposure
from processing.spatial_join import perform_spatial_join
from processing.clustering import cluster_hotspots_spatiotemporal
from processing.persistence_log import update_persistence_log
from processing.feature_engineering import build_feature_table
from processing.labeling_heuristic import apply_labeling_heuristic
from models.model_a_classifier import ModelAClassifier
from models.model_b_anomaly_engine import ModelBAnomalyEngine
from models.model_c_risk_model import ModelCRiskModel
from models.score_combiner import ScoreCombiner


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KB"
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def preflight_check() -> None:
    """
    Verifies required directories, environment variables, and cache files before
    starting the training pipeline, printing a clear status message for anything missing.
    """
    print("=" * 70)
    print("  PRE-FLIGHT ENVIRONMENT & CACHE CHECK")
    print("=" * 70)

    # 1. Required directories
    for label, folder in [
        ("Data directory", DATA_DIR),
        ("Artifacts directory", ARTIFACTS_DIR),
        ("Database schema directory", DB_DIR),
    ]:
        if folder.exists():
            print(f"  [OK]      Folder exists: {label} ({folder.resolve()})")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED] Missing folder created: {label} ({folder.resolve()})")

    # 2. Environment variables & .env file
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        print(f"  [OK]      .env configuration file found ({env_file.resolve()})")
    else:
        print(f"  [MISSING] .env file not found at {env_file.resolve()} (using process env / defaults)")

    valid_key = bool(FIRMS_MAP_KEY and FIRMS_MAP_KEY.strip() and not FIRMS_MAP_KEY.startswith("your_"))
    if valid_key:
        print(f"  [OK]      FIRMS_MAP_KEY configured ({len(FIRMS_MAP_KEYS)} key(s) in rotation pool)")
    else:
        print(
            "  [WARNING] FIRMS_MAP_KEY is missing or placeholder in .env!\n"
            "            -> Pipeline will use the offline synthetic simulation generator.\n"
            "            -> Set FIRMS_MAP_KEY in .env to ingest real NASA VIIRS detections."
        )

    print(f"  [OK]      DEFAULT_REGION={DEFAULT_REGION}")

    # 3. Cache & database files
    cache_files = [
        ("OSM facility cache", Path(OSM_CACHE_PATH), "Will be fetched via Overpass API in Step 1"),
        ("SQLite persistence DB", Path(DB_PATH), "Will be initialized and populated in Step 3"),
        ("ESA WorldCover cache", DATA_DIR / "landcover_cache.json", "Will be populated via S3 COG reads in Step 4"),
        ("Open-Meteo weather cache", DATA_DIR / "weather_cache.json", "Will be populated via Open-Meteo in Step 4"),
    ]
    for label, path, missing_note in cache_files:
        if path.exists():
            size_str = _format_size(path.stat().st_size)
            print(f"  [OK]      {label}: {path.resolve()} ({size_str})")
        else:
            print(f"  [MISSING] {label} not found at {path.resolve()} -> {missing_note}")
    print("=" * 70 + "\n")


def print_completion_summary(
    elapsed: float,
    n_detections: int,
    n_events: int,
    anomaly_engine: ModelBAnomalyEngine,
) -> None:
    """
    Prints the full path and size of every artifact saved, plus row counts for events,
    facilities, and baselines.
    """
    # Count OSM facilities
    n_osm_elements = 0
    if Path(OSM_CACHE_PATH).exists():
        try:
            with open(OSM_CACHE_PATH, "r", encoding="utf-8") as fh:
                osm_payload = json.load(fh)
                n_osm_elements = len(osm_payload.get("elements", []))
        except Exception:
            n_osm_elements = 0

    # Count SQLite history rows
    n_db_rows = 0
    if Path(DB_PATH).exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM hotspot_history")
                n_db_rows = int(cur.fetchone()[0] or 0)
        except Exception:
            n_db_rows = 0

    n_fac_baselines = len(anomaly_engine.facility_baselines)
    n_reg_baselines = len(anomaly_engine.regional_baselines)

    print("\n" + "=" * 70)
    print(f"  TRAINING PIPELINE SUMMARY (Completed in {elapsed:.2f}s)")
    print("=" * 70)
    print("  ROW & ENTITY COUNTS:")
    print(f"    * Raw hotspot detections processed : {n_detections:,}")
    print(f"    * Clustered thermal events         : {n_events:,}")
    print(f"    * SQLite hotspot_history rows      : {n_db_rows:,}")
    print(f"    * Cached OSM elements              : {n_osm_elements:,}")
    print(f"    * Model B facility baselines       : {n_fac_baselines:,}")
    print(f"    * Model B regional type baselines  : {n_reg_baselines:,}")

    print("\n  SAVED ARTIFACTS & DATA FILES (Full Path & Size):")
    expected_files = [
        ARTIFACTS_DIR / "model_a.joblib",
        ARTIFACTS_DIR / "facility_baselines.json",
        ARTIFACTS_DIR / "model_c.joblib",
         Path(DB_PATH),
        DATA_DIR / "feature_table.csv",
        DATA_DIR / "labeled_dataset.csv",
        Path(OSM_CACHE_PATH),
        DATA_DIR / "landcover_cache.json",
        DATA_DIR / "weather_cache.json",
    ]
    for fp in expected_files:
        abs_p = fp.resolve()
        if abs_p.exists():
            sz = _format_size(abs_p.stat().st_size)
            print(f"    [SAVED]   {str(abs_p):<62} ({sz})")
        else:
            print(f"    [MISSING] {str(abs_p)}")
    print("=" * 70)


def run_training_pipeline(
    region: Optional[str] = None,
    west: Optional[float] = None,
    south: Optional[float] = None,
    east: Optional[float] = None,
    north: Optional[float] = None,
    days: int = 30,
) -> None:
    """
    Executes the 7-step data ingestion, processing, baseline fitting, and model training sequence.
    """
    preflight_check()

    # Resolve bounding box coordinates
    if west is None or south is None or east is None or north is None:
        target_region = region or DEFAULT_REGION
        bbox = get_region_bbox(target_region)
        w = float(west if west is not None else bbox["west"])
        s = float(south if south is not None else bbox["south"])
        e = float(east if east is not None else bbox["east"])
        n = float(north if north is not None else bbox["north"])
    else:
        target_region = region or "custom"
        w, s, e, n = float(west), float(south), float(east), float(north)

    start_time = time.time()
    print("=" * 70)
    print("  INDUSTRIAL FIRE & PERSISTENT THERMAL SOURCE DETECTION PIPELINE")
    print("=" * 70)
    print(f"Target Region:       {target_region}")
    print(f"Target Bounding Box: West={w}, South={s}, East={e}, North={n}")
    print(f"Historical Window:   {days} days")
    print(f"Artifacts Path:      {ARTIFACTS_DIR.resolve()}")
    print(f"Database Path:       {Path(DB_PATH).resolve()}\n")

    # -------------------------------------------------------------------------
    # STEP 1: Ingestion & Spatial Join (Boundary Distances)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 1/7] Ingesting NASA FIRMS Hotspots & OSM Infrastructure Geometries...")
    raw_cache_path = DATA_DIR / "firms_raw.csv"
    import pandas as pd
    if raw_cache_path.exists() and (time.time() - raw_cache_path.stat().st_mtime < 3600):
        print(f"[FIRMS] Reusing recent raw FIRMS cache from {raw_cache_path}...")
        df_raw = pd.read_csv(raw_cache_path)
    else:
        df_raw = fetch_firms_hotspots(west=w, south=s, east=e, north=n, total_days=days)
        if not df_raw.empty:
            df_raw.to_csv(raw_cache_path, index=False)
            print(f"[FIRMS] Saved {len(df_raw):,} raw observations to {raw_cache_path}")

    if df_raw.empty:
        raise RuntimeError("No hotspot data available to train or process pipeline.")

    # Cache OSM infrastructure (refresh if missing or only containing the 6-element synthetic fallback)
    need_osm_refresh = True
    if Path(OSM_CACHE_PATH).exists():
        try:
            with open(OSM_CACHE_PATH, "r", encoding="utf-8") as fh:
                existing_osm = json.load(fh)
            if len(existing_osm.get("elements", [])) >= 50:
                need_osm_refresh = False
        except Exception:
            need_osm_refresh = True

    if need_osm_refresh:
        if target_region == "all_india":
            print("[OSM] Fetching and merging OSM industrial facilities across Indian industrial belts...")
            merged_elements = {}
            seen_bboxes = set()
            for sub_reg in REGIONS.keys():
                if sub_reg == "all_india":
                    continue
                sub_box = get_region_bbox(sub_reg)
                bbox_tuple = (sub_box["west"], sub_box["south"], sub_box["east"], sub_box["north"])
                if bbox_tuple in seen_bboxes:
                    continue
                seen_bboxes.add(bbox_tuple)
                print(f"  -> Querying Overpass for region '{sub_reg}'...")
                sub_data = fetch_osm_industrial_and_exposure(
                    west=sub_box["west"],
                    south=sub_box["south"],
                    east=sub_box["east"],
                    north=sub_box["north"],
                    force_refresh=True,
                )
                for el in sub_data.get("elements", []):
                    key = (el.get("type"), el.get("id"))
                    merged_elements[key] = el
            merged_payload = {
                "version": 0.6,
                "generator": "Overpass API Multi-Region India Extract",
                "elements": list(merged_elements.values()),
            }
            with open(OSM_CACHE_PATH, "w", encoding="utf-8") as fh:
                json.dump(merged_payload, fh)
            print(f"[OSM] Saved {len(merged_elements):,} unique OSM elements to {OSM_CACHE_PATH}")
        else:
            fetch_osm_industrial_and_exposure(west=w, south=s, east=e, north=n, force_refresh=True)
    else:
        fetch_osm_industrial_and_exposure(west=w, south=s, east=e, north=n)

    # Boundary distance spatial join
    df_joined = perform_spatial_join(df_raw, str(OSM_CACHE_PATH))
    print(f" -> Spatial Join complete: {len(df_joined)} points processed with boundary distances.")

    # Bound N for O(N^2) DBSCAN distance matrix while preserving 100% of near-industrial detections
    max_cluster_rows = 8500
    if len(df_joined) > max_cluster_rows:
        near_ind = df_joined[df_joined["dist_to_nearest_industrial_m"] <= 3000.0]
        far_ind = df_joined[df_joined["dist_to_nearest_industrial_m"] > 3000.0]
        remaining_budget = max(1000, max_cluster_rows - len(near_ind))
        if len(far_ind) > remaining_budget:
            far_ind = far_ind.sample(n=remaining_budget, random_state=42)
        import pandas as pd
        df_joined = pd.concat([near_ind, far_ind], ignore_index=True)
        print(
            f" -> Balanced nationwide sample to {len(df_joined):,} rows "
            f"(kept all {len(near_ind):,} near-facility points + {len(far_ind):,} background points) "
            f"to bound O(N^2) DBSCAN memory."
        )

    # -------------------------------------------------------------------------
    # STEP 2: Spatiotemporal Clustering (DBSCAN) - BEFORE Feature Engineering
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 2/7] Running Spatiotemporal DBSCAN Clustering (Assign event_ids)...")
    df_clustered = cluster_hotspots_spatiotemporal(df_joined)
    n_events = int(df_clustered["event_id"].nunique())
    print(f" -> Clustering complete: Formed {n_events} distinct thermal event clusters.")

    # -------------------------------------------------------------------------
    # STEP 3: Build/Update Persistence History Log (hotspot_history Table)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 3/7] Updating Hotspot Persistence History Database...")
    df_persisted = update_persistence_log(df_clustered, str(DB_PATH))
    print(" -> Persistence log updated. Computed days_active_last_30 and night_detection_fraction.")

    # -------------------------------------------------------------------------
    # STEP 4: Build Canonical Feature Engineering Table
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 4/7] Constructing Full Feature Engineering Matrix...")
    anomaly_engine = ModelBAnomalyEngine(ARTIFACTS_DIR, DB_PATH)
    anomaly_engine.fit_from_db()

    df_features = build_feature_table(df_persisted, anomaly_engine=anomaly_engine)
    feature_csv_path = DATA_DIR / "feature_table.csv"
    df_features.to_csv(feature_csv_path, index=False)
    print(f" -> Feature engineering complete: Saved {len(df_features)} rows to {feature_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 5: Apply Heuristic Pseudo-Labeling (7 Classes)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 5/7] Bootstrapping Ground Truth via 7-Class Labeling Heuristic...")
    df_labeled = apply_labeling_heuristic(df_features)
    labeled_csv_path = DATA_DIR / "labeled_dataset.csv"
    df_labeled.to_csv(labeled_csv_path, index=False)
    print(" -> Pseudo-labeling complete. Class distribution:")
    for cls_name, count in df_labeled["event_type"].value_counts().items():
        print(f"    * {cls_name:<25}: {count:>4} samples")

    # -------------------------------------------------------------------------
    # STEP 6: Train Models A, B, and C
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 6/7] Training Models A and C (Model B baselines fit in Step 4)...")

    # Model A: LightGBM Multiclass Classifier + SHAP TreeExplainer
    print("\n--- Training Model A (LightGBM Multi-Class Event Classifier) ---")
    model_a = ModelAClassifier(ARTIFACTS_DIR)
    metrics_a = model_a.train(df_labeled)
    print(
        f" -> Model A trained (train={metrics_a.get('n_train', 0)}, val={metrics_a.get('n_val', 0)} samples)."
    )

    # Model C: LightGBM 24h Escalation Risk Model (Filtered duty_cycle < 0.2)
    print("\n--- Training Model C (LightGBM 24h Risk Escalation Model) ---")
    model_c = ModelCRiskModel(ARTIFACTS_DIR)
    metrics_c = model_c.train(df_labeled)
    print(
        f" -> Model C trained (train={metrics_c.get('n_train', 0)}, val={metrics_c.get('n_val', 0)} samples)."
    )

    # -------------------------------------------------------------------------
    # STEP 7: Score Combiner Verification
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 7/7] Verifying Score Combiner on Test Samples...")
    combiner = ScoreCombiner()
    sample_row = df_features.iloc[0]
    pred_a = model_a.predict_point(sample_row)
    z_score, _ = anomaly_engine.compute_z_score(
        sample_row.get("location_key", ""),
        float(sample_row.get("frp", 0.0)),
        str(sample_row.get("nearest_industrial_type", "none")),
    )
    pred_b = {"anomaly_score_normalized": anomaly_engine.compute_anomaly_score_normalized(z_score)}
    pred_c = model_c.predict_point(sample_row)

    combined_score = combiner.combine(
        model_a_output=pred_a,
        model_b_output=pred_b,
        model_c_output=pred_c,
        dist_to_populated_m=float(sample_row.get("dist_to_populated_area_m", 5000.0)),
        dist_to_infra_m=float(sample_row.get("dist_to_critical_infra_m", 5000.0)),
    )

    print(" -> Sample End-to-End Scored Point:")
    print(json.dumps(combined_score, indent=2))

    elapsed = round(time.time() - start_time, 2)
    print_completion_summary(
        elapsed=elapsed,
        n_detections=len(df_raw),
        n_events=n_events,
        anomaly_engine=anomaly_engine,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Industrial Fire Pipeline & Model Trainer")
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        help=f"Named Indian industrial region ({', '.join(REGIONS.keys())})",
    )
    parser.add_argument("--west", type=float, default=None, help="Optional raw west bbox override")
    parser.add_argument("--south", type=float, default=None, help="Optional raw south bbox override")
    parser.add_argument("--east", type=float, default=None, help="Optional raw east bbox override")
    parser.add_argument("--north", type=float, default=None, help="Optional raw north bbox override")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    run_training_pipeline(
        region=args.region,
        west=args.west,
        south=args.south,
        east=args.east,
        north=args.north,
        days=args.days,
    )
