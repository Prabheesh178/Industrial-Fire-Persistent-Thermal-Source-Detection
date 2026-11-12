"""
FastAPI REST API Service
========================

HOW TO RUN:
    Start the API server:
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    Or from the project root:
        python -m api.main

API ENDPOINTS:
    - GET /health
      Returns service status, data_as_of currency timestamp, active event count,
      and dynamically inspected model artifact file metadata (size and modified time).
    - GET /regions
      Returns predefined Indian industrial regions and bounding boxes.
    - GET /facilities
      Returns GeoJSON FeatureCollection of industrial facilities parsed from cached OSM data.
    - GET /hotspots?region=&bbox=&since_hours=
      Returns aggregated thermal hotspot events (one record per unique event_id)
      with strict observed vs inferred separation and signed SHAP attributions.
    - GET /hotspot/{event_id}
      Returns full feature vector, constituent cluster pixels, class probabilities,
      and signed SHAP feature attributions.
    - GET /facility/{location_key}/history
      Returns FRP time-series with median, MAD, and 3-sigma anomaly threshold bands for charting.

OUTPUT:
    - JSON responses matching the exact schemas defined in api/schemas.py.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    DEFAULT_BBOX,
    DEFAULT_REGION,
    REGIONS,
    get_region_bbox,
    get_regions_list,
    DATA_DIR,
    DB_PATH,
    ARTIFACTS_DIR,
    OSM_CACHE_PATH,
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
)
from api.schemas import (
    HotspotResponse,
    HotspotPixel,
    ShapDriver,
    ObservedData,
    InferredData,
    HotspotDetailResponse,
    FacilityHistoryResponse,
    RegionInfo,
    ArtifactInfo,
    HealthResponse,
)
from models.model_a_classifier import ModelAClassifier
from models.model_b_anomaly_engine import ModelBAnomalyEngine
from models.model_c_risk_model import ModelCRiskModel
from models.score_combiner import ScoreCombiner

# Initialize FastAPI App
app = FastAPI(
    title="Industrial Fire & Persistent Thermal Source Detection API",
    description="Operational API for classifying industrial flares, accidents, wildfires, and unauthorized thermal sources across India.",
    version="1.0.0",
)

# Enable universal CORS covering all Vercel domains, localhost, and custom frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
model_a = ModelAClassifier(ARTIFACTS_DIR)
model_b = ModelBAnomalyEngine(ARTIFACTS_DIR, DB_PATH)
model_c = ModelCRiskModel(ARTIFACTS_DIR)
score_combiner = ScoreCombiner()

# In-memory cache of scored hotspot events (keyed by unique event_id)
_SCORED_CACHE: Dict[str, Dict[str, Any]] = {}
_FACILITY_NAME_CACHE: Dict[str, str] = {}
_FACILITIES_GEOJSON_CACHE: Optional[Dict[str, Any]] = None
_SNAPSHOT_DF_CACHE: Optional[pd.DataFrame] = None
_SNAPSHOT_MTIME: Optional[float] = None

FEATURE_TABLE_PATH = Path(DATA_DIR) / "feature_table.csv"
LABELED_DATASET_PATH = Path(DATA_DIR) / "labeled_dataset.csv"
SLIM_FACILITIES_PATH = Path(DATA_DIR) / "facilities_slim.json"


def _classify_osm_element(tags: Dict[str, Any]) -> Optional[str]:
    """
    Maps OSM element tags to canonical facility category without requiring GeoPandas.
    """
    if (
        (tags.get("power") == "generator" and tags.get("generator:source") == "solar")
        or tags.get("plant:source") == "solar"
    ):
        return "solar_farm"
    if (
        tags.get("landuse") == "landfill"
        or tags.get("amenity") == "waste_disposal"
        or "waste" in tags
    ):
        return "waste_landfill"
    if (
        tags.get("landuse") == "quarry"
        or tags.get("product") == "coal"
        or "mining" in tags.values()
    ):
        return "quarry_mining"
    if (
        tags.get("product") in ["oil", "petroleum", "gas"]
        or tags.get("man_made") == "petroleum_well"
        or "refinery" in str(tags.get("name", "")).lower()
    ):
        return "refinery"
    if tags.get("power") == "plant":
        return "power_plant"
    if tags.get("landuse") == "industrial" or tags.get("man_made") == "works":
        return "general_industrial"
    return None


def _extract_element_coords(el: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Extracts representative (lon, lat) point from an Overpass JSON element.
    """
    el_type = el.get("type")
    if el_type == "node" and "lat" in el and "lon" in el:
        return float(el["lon"]), float(el["lat"])
    if el_type == "way" and "geometry" in el and el["geometry"]:
        lats = [float(pt["lat"]) for pt in el["geometry"] if "lat" in pt]
        lons = [float(pt["lon"]) for pt in el["geometry"] if "lon" in pt]
        if lats and lons:
            return float(np.mean(lons)), float(np.mean(lats))
    if el_type == "relation" and "members" in el:
        lats, lons = [], []
        for member in el.get("members", []):
            for pt in member.get("geometry", []):
                if "lat" in pt and "lon" in pt:
                    lats.append(float(pt["lat"]))
                    lons.append(float(pt["lon"]))
        if lats and lons:
            return float(np.mean(lons)), float(np.mean(lats))
    return None


def _load_osm_facilities() -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Parses cached facility extract into a facility_id -> name map and a GeoJSON FeatureCollection.
    Prioritizes data/facilities_slim.json (~5 MB) for low RAM footprint, falling back to data/osm_cache.json.
    """
    global _FACILITY_NAME_CACHE, _FACILITIES_GEOJSON_CACHE
    if _FACILITIES_GEOJSON_CACHE is not None:
        return _FACILITY_NAME_CACHE, _FACILITIES_GEOJSON_CACHE

    names: Dict[str, str] = {}
    features: List[Dict[str, Any]] = []

    # 1. Prefer slim runtime facilities file
    if SLIM_FACILITIES_PATH.exists():
        try:
            with open(SLIM_FACILITIES_PATH, "r", encoding="utf-8") as f:
                slim_data = json.load(f)
            _FACILITY_NAME_CACHE = slim_data.get("names", {})
            _FACILITIES_GEOJSON_CACHE = slim_data.get("facilities", {"type": "FeatureCollection", "features": []})
            return _FACILITY_NAME_CACHE, _FACILITIES_GEOJSON_CACHE
        except Exception as e:
            print(f"[API] Warning: Failed to parse slim facilities cache ({e}). Falling back...")

    # 2. Fall back to raw OSM cache if slim file is missing
    if Path(OSM_CACHE_PATH).exists():
        try:
            with open(OSM_CACHE_PATH, "r", encoding="utf-8") as f:
                osm_data = json.load(f)
            for el in osm_data.get("elements", []):
                el_type = el.get("type")
                el_id = el.get("id")
                tags = el.get("tags", {}) or {}
                fac_id = f"osm_{el_type}_{el_id}"
                name = tags.get("name")
                if name:
                    names[fac_id] = str(name)

                category = _classify_osm_element(tags)
                if not category:
                    continue
                coords = _extract_element_coords(el)
                if not coords:
                    continue
                lon, lat = coords
                features.append(
                    {
                        "type": "Feature",
                        "id": fac_id,
                        "geometry": {
                            "type": "Point",
                            "coordinates": [round(lon, 5), round(lat, 5)],
                        },
                        "properties": {
                            "facility_id": fac_id,
                            "name": name or f"{category.replace('_', ' ').title()} ({fac_id})",
                            "category": category,
                            "osm_type": el_type,
                            "osm_id": el_id,
                        },
                    }
                )
        except Exception as e:
            print(f"[API] Warning: Failed to parse OSM cache ({e}).")

    _FACILITY_NAME_CACHE = names
    _FACILITIES_GEOJSON_CACHE = {
        "type": "FeatureCollection",
        "features": features,
    }
    return _FACILITY_NAME_CACHE, _FACILITIES_GEOJSON_CACHE


def ensure_models_loaded() -> Tuple[bool, List[str], Optional[str]]:
    """
    Ensures Model A, Model B, and Model C artifacts are loaded in memory.
    Returns (all_ok, missing_files, error_message).
    """
    model_a_path = Path(ARTIFACTS_DIR) / "model_a.joblib"
    model_b_path = Path(ARTIFACTS_DIR) / "facility_baselines.json"
    model_c_path = Path(ARTIFACTS_DIR) / "model_c.joblib"

    missing: List[str] = []

    if not model_a_path.exists():
        missing.append("artifacts/model_a.joblib")
    elif not model_a.is_trained or model_a.model is None:
        try:
            model_a.load()
        except Exception as e:
            missing.append(f"artifacts/model_a.joblib (load failed: {e})")

    if not model_b_path.exists():
        missing.append("artifacts/facility_baselines.json")
    elif not model_b.facility_baselines and not model_b.regional_baselines:
        try:
            model_b.load()
        except Exception as e:
            missing.append(f"artifacts/facility_baselines.json (load failed: {e})")

    if not model_c_path.exists():
        missing.append("artifacts/model_c.joblib")
    elif not model_c.is_trained or model_c.model is None:
        try:
            model_c.load()
        except Exception as e:
            missing.append(f"artifacts/model_c.joblib (load failed: {e})")

    if missing:
        msg = (
            f"Missing required trained model artifacts: {', '.join(missing)}. "
            "Run 'python -m models.train_all' and commit the generated files before serving."
        )
        return False, missing, msg

    return True, [], None


def _inspect_artifact_file(path: Path, is_loaded: bool) -> ArtifactInfo:
    """
    Reads filesystem metadata (size and UTC modified time) for an artifact file.
    """
    if not path.exists():
        return ArtifactInfo(
            filename=path.name,
            exists=False,
            loaded=False,
            size_bytes=None,
            modified_at=None,
        )
    stat = path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ArtifactInfo(
        filename=path.name,
        exists=True,
        loaded=is_loaded,
        size_bytes=int(stat.st_size),
        modified_at=mtime_iso,
    )


def _format_iso_utc(acq_date: Any, acq_time: Any) -> str:
    """
    Formats FIRMS acq_date ('YYYY-MM-DD') and acq_time ('HHMM' or int) into ISO-8601 UTC string.
    """
    d_str = str(acq_date or "2026-09-04").strip()
    t_raw = str(acq_time or "0000").strip().split(".")[0]
    t_padded = t_raw.zfill(4)
    hh = t_padded[:2] if len(t_padded) >= 2 else "00"
    mm = t_padded[2:4] if len(t_padded) >= 4 else "00"
    if not hh.isdigit() or int(hh) > 23:
        hh = "00"
    if not mm.isdigit() or int(mm) > 59:
        mm = "00"
    return f"{d_str}T{hh}:{mm}:00Z"


def _get_dataset_summary() -> Tuple[Optional[str], int]:
    """
    Determines max acquisition timestamp (data_as_of) and distinct event count from
    cached snapshot or SQLite persistence log.
    """
    if _SNAPSHOT_DF_CACHE is not None and not _SNAPSHOT_DF_CACHE.empty:
        data_as_of = str(_SNAPSHOT_DF_CACHE["_iso_ts"].max()) if "_iso_ts" in _SNAPSHOT_DF_CACHE.columns else None
        n_events = (
            int(_SNAPSHOT_DF_CACHE["event_id"].nunique())
            if "event_id" in _SNAPSHOT_DF_CACHE.columns
            else len(_SNAPSHOT_DF_CACHE)
        )
        return data_as_of, n_events

    for csv_path in (FEATURE_TABLE_PATH, LABELED_DATASET_PATH):
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, usecols=lambda c: c in {"event_id", "acq_date", "acq_time"})
                if not df.empty:
                    iso_times = [
                        _format_iso_utc(d, t)
                        for d, t in zip(
                            df["acq_date"],
                            df["acq_time"] if "acq_time" in df.columns else ["0000"] * len(df),
                        )
                    ]
                    data_as_of = max(iso_times) if iso_times else None
                    n_events = int(df["event_id"].nunique()) if "event_id" in df.columns else len(df)
                    return data_as_of, n_events
            except Exception:
                pass

    if _SCORED_CACHE:
        last_seen_vals = [
            v["response"].last_seen
            for v in _SCORED_CACHE.values()
            if hasattr(v.get("response"), "last_seen") and v["response"].last_seen
        ]
        data_as_of = max(last_seen_vals) if last_seen_vals else None
        return data_as_of, len(_SCORED_CACHE)

    if Path(DB_PATH).exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute("SELECT MAX(acq_date), COUNT(DISTINCT event_id) FROM hotspot_history")
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                return f"{row[0]}T23:59:00Z", int(row[1] or 0)
        except Exception:
            pass

    return None, 0


def _compute_signed_shap_factors(feature_row: pd.Series, event_type: str, top_n: int = 5) -> List[ShapDriver]:
    """
    Extracts signed SHAP drivers ({feature, direction, weight}) from Model A's TreeExplainer.
    Never fabricates weights: returns [] if explainer is unavailable.
    """
    if not model_a.is_trained or model_a.explainer is None or model_a.label_encoder is None:
        return []

    try:
        df_single = pd.DataFrame([feature_row]).reset_index(drop=True)
        X = model_a._preprocess_features(df_single)
        if event_type in model_a.label_encoder.classes_:
            class_idx = int(model_a.label_encoder.transform([event_type])[0])
        else:
            class_idx = 0

        shap_values = model_a.explainer.shap_values(X)
        if isinstance(shap_values, list):
            signed_vals = np.asarray(shap_values[class_idx][0], dtype=float)
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            signed_vals = np.asarray(shap_values[0, :, class_idx], dtype=float)
        else:
            signed_vals = np.asarray(shap_values[0], dtype=float)

        abs_vals = np.abs(signed_vals)
        total_abs = float(np.sum(abs_vals))
        if total_abs <= 1e-12:
            return []

        top_indices = np.argsort(abs_vals)[::-1][:top_n]
        drivers: List[ShapDriver] = []
        feature_names = model_a.feature_names
        for idx in top_indices:
            weight = round(float(abs_vals[idx] / total_abs), 3)
            if weight <= 0:
                continue
            direction = "+" if float(signed_vals[idx]) >= 0.0 else "-"
            drivers.append(
                ShapDriver(
                    feature=str(feature_names[idx]),
                    direction=direction,
                    weight=weight,
                )
            )
        return drivers
    except Exception:
        return []


def _load_or_build_feature_dataframe(
    w: float, s: float, e: float, n: float, since_hours: int
) -> pd.DataFrame:
    """
    Loads pre-engineered feature table from disk when available (Render serving mode),
    or lazily runs the ingestion & spatial pipeline if live polling is requested or no snapshot exists.
    Filters rows by bounding box [w, s, e, n] and since_hours relative to max timestamp in the data.
    """
    global _SNAPSHOT_DF_CACHE, _SNAPSHOT_MTIME
    live_poll = os.getenv("LIVE_FIRMS_POLL", "false").lower() == "true"
    df_features: Optional[pd.DataFrame] = None

    if not live_poll:
        for candidate in (FEATURE_TABLE_PATH, LABELED_DATASET_PATH):
            if candidate.exists():
                try:
                    mtime = candidate.stat().st_mtime
                    if _SNAPSHOT_DF_CACHE is not None and _SNAPSHOT_MTIME == mtime:
                        df_features = _SNAPSHOT_DF_CACHE
                        break
                    df_loaded = pd.read_csv(candidate)
                    if not df_loaded.empty and "latitude" in df_loaded.columns and "longitude" in df_loaded.columns:
                        if "acq_date" in df_loaded.columns:
                            iso_list = [
                                _format_iso_utc(d, t)
                                for d, t in zip(
                                    df_loaded["acq_date"],
                                    df_loaded["acq_time"] if "acq_time" in df_loaded.columns else ["0000"] * len(df_loaded),
                                )
                            ]
                            df_loaded["_iso_ts"] = iso_list
                            df_loaded["_ts_dt"] = pd.to_datetime(iso_list, utc=True, errors="coerce")
                        _SNAPSHOT_DF_CACHE = df_loaded
                        _SNAPSHOT_MTIME = mtime
                        df_features = df_loaded
                        break
                except Exception as exc:
                    print(f"[API] Warning: Could not read {candidate}: {exc}")

    if df_features is not None:
        # Filter relative to global max timestamp in dataset if since_hours is specified
        df_sub = df_features
        if since_hours > 0 and "_ts_dt" in df_sub.columns:
            max_ts = df_features["_ts_dt"].max()
            if pd.notna(max_ts):
                cutoff = max_ts - pd.Timedelta(hours=since_hours)
                df_sub = df_sub[df_sub["_ts_dt"] >= cutoff]

        # Filter by bounding box
        mask_bbox = (
            (df_sub["longitude"] >= w)
            & (df_sub["longitude"] <= e)
            & (df_sub["latitude"] >= s)
            & (df_sub["latitude"] <= n)
        )
        return df_sub[mask_bbox].copy()

    # Lazy import heavy geospatial modules only when running live ingestion pipeline
    from data_ingestion.fetch_firms import fetch_firms_hotspots
    from processing.spatial_join import perform_spatial_join
    from processing.clustering import cluster_hotspots_spatiotemporal
    from processing.persistence_log import update_persistence_log
    from processing.feature_engineering import build_feature_table

    days_lookback = max(1, (since_hours + 23) // 24)
    df_raw = fetch_firms_hotspots(west=w, south=s, east=e, north=n, total_days=days_lookback)
    if df_raw.empty:
        return pd.DataFrame()

    df_joined = perform_spatial_join(df_raw, str(OSM_CACHE_PATH))
    df_clustered = cluster_hotspots_spatiotemporal(df_joined)
    df_persisted = update_persistence_log(df_clustered, str(DB_PATH))
    return build_feature_table(df_persisted, anomaly_engine=model_b)


def _extract_shap_from_matrix(
    shap_values: Any, sample_idx: int, class_idx: int, top_n: int = 5
) -> Tuple[List[str], List[ShapDriver]]:
    """
    Extracts top feature names and signed ShapDriver objects for row `sample_idx` and `class_idx`
    from a batch TreeExplainer output.
    """
    try:
        if isinstance(shap_values, list):
            signed_vals = np.asarray(shap_values[class_idx][sample_idx], dtype=float)
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            signed_vals = np.asarray(shap_values[sample_idx, :, class_idx], dtype=float)
        else:
            signed_vals = np.asarray(shap_values[sample_idx], dtype=float)

        abs_vals = np.abs(signed_vals)
        total_abs = float(np.sum(abs_vals))
        feature_names = model_a.feature_names
        top_indices = np.argsort(abs_vals)[::-1]
        top_names = [str(feature_names[i]) for i in top_indices[:3]]

        if total_abs <= 1e-12:
            return top_names, []

        drivers: List[ShapDriver] = []
        for idx in top_indices[:top_n]:
            weight = round(float(abs_vals[idx] / total_abs), 3)
            if weight <= 0:
                continue
            direction = "+" if float(signed_vals[idx]) >= 0.0 else "-"
            drivers.append(
                ShapDriver(
                    feature=str(feature_names[idx]),
                    direction=direction,
                    weight=weight,
                )
            )
        return top_names, drivers
    except Exception:
        return ["dist_to_nearest_industrial_m", "night_detection_fraction", "daynight"], []


def _score_and_aggregate_events(df_features: pd.DataFrame) -> List[HotspotResponse]:
    """
    Scores each pixel row through Model A, Model B, Model C, and ScoreCombiner,
    then aggregates rows by event_id into a single HotspotResponse per event_id.
    Uses the highest-severity pixel (highest final_risk_score, tie-broken by FRP)
    for the event's inferred scores and SHAP attributions.
    """
    if df_features.empty:
        return []

    facility_names, _ = _load_osm_facilities()

    # Ensure event_id column exists
    if "event_id" not in df_features.columns:
        df_features = df_features.copy()
        df_features["event_id"] = [f"evt_{i:04d}" for i in range(len(df_features))]

    results: List[HotspotResponse] = []
    uncached_groups: List[Tuple[str, pd.DataFrame]] = []

    for event_id_raw, group in df_features.groupby("event_id", sort=False):
        event_id = str(event_id_raw)
        cached = _SCORED_CACHE.get(event_id)
        if cached is not None and cached["response"].pixel_count == len(group):
            results.append(cached["response"])
        else:
            uncached_groups.append((event_id, group))

    if uncached_groups:
        df_uncached = pd.concat([g for _, g in uncached_groups], axis=0).reset_index(drop=True)
        X_a_all = model_a._preprocess_features(df_uncached)
        probs_a_all = model_a.model.predict_proba(X_a_all)
        classes_a = list(model_a.label_encoder.classes_)

        X_c_all = model_c._preprocess_features(df_uncached)
        probs_c_all = model_c.model.predict_proba(X_c_all)[:, 1]

        pending_events: List[Dict[str, Any]] = []
        row_offset = 0

        for event_id, group in uncached_groups:
            n_rows = len(group)
            pixels_list: List[HotspotPixel] = []
            scored_candidates: List[Dict[str, Any]] = []

            for local_i, (_, row) in enumerate(group.iterrows()):
                global_i = row_offset + local_i
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                frp = float(row.get("frp", 0.0))
                ti4 = float(row.get("bright_ti4", row.get("brightness_ti4", 320.0)))
                conf = float(row.get("confidence", 80.0))
                acq_date = str(row.get("acq_date", "2026-09-04"))
                acq_time = str(row.get("acq_time", "0000")).split(".")[0].zfill(4)
                iso_ts = str(row.get("_iso_ts") or _format_iso_utc(acq_date, acq_time))
                ind_type = str(row.get("nearest_industrial_type", "none"))
                dist_ind = float(row.get("dist_to_nearest_industrial_m", 5000.0))
                loc_key = str(row.get("location_key", f"h3_{round(lat, 3)}_{round(lon, 3)}"))

                pixels_list.append(
                    HotspotPixel(
                        lat=round(lat, 5),
                        lon=round(lon, 5),
                        frp=round(frp, 2),
                        brightness_ti4=round(ti4, 1),
                        acq_date=acq_date,
                        acq_time=acq_time,
                    )
                )

                # Model A inference from vectorized batch
                probs_row = probs_a_all[global_i]
                top_idx = int(np.argmax(probs_row))
                pred_class = str(classes_a[top_idx])
                confidence_a = float(probs_row[top_idx])
                pred_a = {
                    "event_type": pred_class,
                    "event_type_confidence": round(confidence_a, 3),
                    "probabilities": {cls: round(float(p), 3) for cls, p in zip(classes_a, probs_row)},
                    "top_shap_features": [],
                    "_class_idx": top_idx,
                }

                # Model B anomaly inference
                z_score, is_anom = model_b.compute_z_score(loc_key, frp, ind_type)
                anomaly_norm = model_b.compute_anomaly_score_normalized(z_score)
                pred_b = {
                    "anomaly_score_normalized": anomaly_norm,
                    "z_score": z_score,
                    "is_anomalous": is_anom,
                }

                # Model C 24h risk inference from vectorized batch
                risk_prob = float(probs_c_all[global_i])
                pred_c = {
                    "risk_24h": round(risk_prob, 3),
                    "escalating_24h": bool(risk_prob >= 0.50),
                }

                # Score Combiner
                dist_pop = float(row.get("dist_to_populated_area_m", 5000.0))
                dist_infra = float(row.get("dist_to_critical_infra_m", 5000.0))
                inferred = score_combiner.combine(
                    model_a_output=pred_a,
                    model_b_output=pred_b,
                    model_c_output=pred_c,
                    dist_to_populated_m=dist_pop,
                    dist_to_infra_m=dist_infra,
                )

                scored_candidates.append(
                    {
                        "row": row,
                        "lat": lat,
                        "lon": lon,
                        "frp": frp,
                        "ti4": ti4,
                        "conf": conf,
                        "acq_date": acq_date,
                        "acq_time": acq_time,
                        "iso_ts": iso_ts,
                        "ind_type": ind_type,
                        "dist_ind": dist_ind,
                        "loc_key": loc_key,
                        "pred_a": pred_a,
                        "pred_b": pred_b,
                        "pred_c": pred_c,
                        "inferred": inferred,
                    }
                )

            row_offset += n_rows
            best = max(
                scored_candidates,
                key=lambda c: (float(c["inferred"]["final_risk_score"]), float(c["frp"])),
            )
            pending_events.append(
                {
                    "event_id": event_id,
                    "pixels_list": pixels_list,
                    "scored_candidates": scored_candidates,
                    "best": best,
                }
            )

        # Vectorized TreeExplainer SHAP computation across representative event rows
        df_rep = pd.DataFrame([pe["best"]["row"] for pe in pending_events]).reset_index(drop=True)
        X_rep = model_a._preprocess_features(df_rep)
        try:
            shap_batch = model_a.explainer.shap_values(X_rep) if model_a.explainer is not None else None
        except Exception:
            shap_batch = None

        for pe_idx, pe in enumerate(pending_events):
            event_id = pe["event_id"]
            pixels_list = pe["pixels_list"]
            scored_candidates = pe["scored_candidates"]
            best = pe["best"]
            rep_row: pd.Series = best["row"]
            rep_inferred = best["inferred"]
            rep_pred_a = best["pred_a"]
            rep_pred_b = best["pred_b"]

            if shap_batch is not None:
                top_shap_names, shap_drivers = _extract_shap_from_matrix(
                    shap_batch, pe_idx, int(rep_pred_a["_class_idx"]), top_n=5
                )
            else:
                shap_drivers = _compute_signed_shap_factors(rep_row, rep_inferred["event_type"], top_n=5)
                top_shap_names = [d.feature for d in shap_drivers[:3]]

            all_lats = [c["lat"] for c in scored_candidates]
            all_lons = [c["lon"] for c in scored_candidates]
            all_frps = [c["frp"] for c in scored_candidates]
            all_ti4s = [c["ti4"] for c in scored_candidates]
            all_confs = [c["conf"] for c in scored_candidates]
            all_ts = [c["iso_ts"] for c in scored_candidates]

            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
            if abs(max_lon - min_lon) < 0.002:
                min_lon -= 0.001
                max_lon += 0.001
            if abs(max_lat - min_lat) < 0.002:
                min_lat -= 0.001
                max_lat += 0.001

            frp_total = round(float(sum(all_frps)), 1)
            frp_max = round(float(max(all_frps)), 1)
            ti4_max = round(float(max(all_ti4s)), 1)
            conf_max = round(float(max(all_confs)), 1)
            first_seen = min(all_ts)
            last_seen = max(all_ts)

            fac_id = str(rep_row.get("facility_id", best["loc_key"]))
            fac_name = facility_names.get(fac_id) or facility_names.get(best["loc_key"])

            observed_obj = ObservedData(
                frp=frp_max,
                frp_total=frp_total,
                frp_max=frp_max,
                brightness_ti4=ti4_max,
                confidence=conf_max,
                acq_date=best["acq_date"],
                acq_time=best["acq_time"],
                daynight=str(rep_row.get("daynight", "D")),
                satellite=str(rep_row.get("satellite", "SNPP")),
                land_cover=str(rep_row.get("land_cover", "unknown")),
                nearest_industrial_type=best["ind_type"],
                nearest_industrial_name=fac_name,
                dist_to_industrial_m=round(best["dist_ind"], 1),
            )

            inferred_obj = InferredData(
                event_type=rep_inferred["event_type"],
                event_type_confidence=rep_inferred["event_type_confidence"],
                z_score=round(float(rep_pred_b["z_score"]), 2),
                is_anomalous=bool(rep_pred_b["is_anomalous"]),
                anomaly_score_normalized=rep_inferred["anomaly_score_normalized"],
                hazard_weight=rep_inferred["hazard_weight"],
                exposure_factor=rep_inferred["exposure_factor"],
                final_risk_score=rep_inferred["final_risk_score"],
                escalating_24h=rep_inferred["escalating_24h"],
                risk_24h=rep_inferred["risk_24h"],
                needs_manual_review=rep_inferred["needs_manual_review"],
                top_shap_features=top_shap_names,
                top_shap_factors=shap_drivers,
            )

            centroid_lat = round(float(np.mean(all_lats)), 4)
            centroid_lon = round(float(np.mean(all_lons)), 4)

            item = HotspotResponse(
                event_id=event_id,
                location_key=best["loc_key"],
                lat=centroid_lat,
                lon=centroid_lon,
                bbox=[
                    round(min_lon, 5),
                    round(min_lat, 5),
                    round(max_lon, 5),
                    round(max_lat, 5),
                ],
                pixel_count=len(scored_candidates),
                first_seen=first_seen,
                last_seen=last_seen,
                observed=observed_obj,
                inferred=inferred_obj,
            )
            results.append(item)

            # Clean feature vector dictionary for JSON serialization
            clean_fv: Dict[str, Any] = {}
            for k, v in rep_row.to_dict().items():
                if str(k).startswith("_"):
                    continue
                if pd.isna(v):
                    clean_fv[str(k)] = None
                elif isinstance(v, (np.integer, int)):
                    clean_fv[str(k)] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_fv[str(k)] = round(float(v), 4)
                elif isinstance(v, (np.bool_, bool)):
                    clean_fv[str(k)] = bool(v)
                else:
                    clean_fv[str(k)] = str(v)

            _SCORED_CACHE[event_id] = {
                "response": item,
                "location_key": best["loc_key"],
                "pixels": pixels_list,
                "feature_vector": clean_fv,
                "class_probabilities": rep_pred_a.get("probabilities", {}),
            }

    # Sort descending by final_risk_score then frp_max
    results.sort(key=lambda r: (r.inferred.final_risk_score, r.observed.frp_max), reverse=True)
    return results


@app.on_event("startup")
def load_models_on_startup():
    """
    Loads trained ML models, statistical baselines, and OSM facility cache into memory on server start.
    Logs clear diagnostics if any model artifacts are missing.
    Spawns background near-real-time FIRMS refresh daemon if REFRESH_ENABLED is set.
    """
    print("[API] Initializing models and baselines on startup...")
    all_ok, missing, err_msg = ensure_models_loaded()
    if all_ok:
        print("[API] All models (Model A, Model B, Model C) loaded successfully.")
    else:
        print(f"[API] WARNING: {err_msg}")
    _load_osm_facilities()

    # Launch background FIRMS refresh worker if enabled
    try:
        from api.refresh_worker import start_refresh_daemon_if_enabled
        start_refresh_daemon_if_enabled()
    except Exception as exc:
        print(f"[API] Warning: Could not initialize refresh worker daemon: {exc}")


@app.get("/")
def root():
    return {
        "service": "Industrial Fire & Persistent Thermal Source Detection API",
        "status": "online",
        "endpoints": [
            "/health",
            "/regions",
            "/facilities",
            "/hotspots?region={region}&since_hours={hours}",
            "/hotspots?bbox={west},{south},{east},{north}&since_hours={hours}",
            "/hotspot/{event_id}",
            "/facility/{location_key}/history",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def get_health():
    """
    Returns operational health, dataset currency timestamp (data_as_of), active event count,
    refresh mode (live vs snapshot), last refresh timestamp, refresh status,
    and dynamically inspected model artifact metadata.
    If model artifacts are missing, returns status='error' with a clear message.
    """
    all_ok, missing, err_msg = ensure_models_loaded()
    data_as_of, active_count = _get_dataset_summary()

    from api.refresh_worker import get_refresh_state
    refresh_info = get_refresh_state()

    artifacts_meta = {
        "model_a": _inspect_artifact_file(
            Path(ARTIFACTS_DIR) / "model_a.joblib",
            is_loaded=bool(model_a.is_trained and model_a.model is not None),
        ),
        "facility_baselines": _inspect_artifact_file(
            Path(ARTIFACTS_DIR) / "facility_baselines.json",
            is_loaded=bool(model_b.facility_baselines or model_b.regional_baselines),
        ),
        "model_c": _inspect_artifact_file(
            Path(ARTIFACTS_DIR) / "model_c.joblib",
            is_loaded=bool(model_c.is_trained and model_c.model is not None),
        ),
        "hotspots_db": _inspect_artifact_file(
            Path(DB_PATH),
            is_loaded=Path(DB_PATH).exists(),
        ),
        "osm_cache": _inspect_artifact_file(
            SLIM_FACILITIES_PATH if SLIM_FACILITIES_PATH.exists() else Path(OSM_CACHE_PATH),
            is_loaded=SLIM_FACILITIES_PATH.exists() or Path(OSM_CACHE_PATH).exists(),
        ),
    }

    return HealthResponse(
        status="ok" if all_ok else "error",
        mode=refresh_info["mode"],
        data_as_of=data_as_of,
        last_refresh_at=refresh_info["last_refresh_at"],
        refresh_status=refresh_info["refresh_status"],
        active_events_count=active_count,
        missing_artifacts=missing,
        model_artifacts=artifacts_meta,
        error=err_msg,
    )


@app.get("/regions", response_model=List[RegionInfo])
def get_regions():
    """
    Returns the list of available predefined Indian industrial regions (name, label, bbox, description)
    so frontend dashboards can build a dynamic region-picker dropdown.
    """
    return get_regions_list()


@app.get("/facilities")
def get_facilities():
    """
    Returns GeoJSON FeatureCollection of industrial, solar, waste, and power facilities
    built from the cached OpenStreetMap extract (data/osm_cache.json).
    """
    _, geojson = _load_osm_facilities()
    return geojson


@app.get("/hotspots", response_model=List[HotspotResponse])
def get_hotspots(
    region: Optional[str] = Query(
        None,
        description="Named Indian region (e.g. 'gujarat', 'maharashtra', 'odisha', 'all_india'). Takes precedence over bbox.",
    ),
    bbox: Optional[str] = Query(
        None,
        description="Bounding box in 'west,south,east,north' format. Example: '72.5,21.0,73.5,22.0'. Used if region is omitted.",
    ),
    since_hours: int = Query(24, description="Lookback window in hours relative to dataset currency (default: 24)"),
):
    """
    Returns aggregated satellite thermal hotspot events for a region or bounding box,
    with satellite evidence strictly separated from model inferences.
    Raises HTTP 503 if required model artifacts have not been trained yet.
    """
    # 1. Resolve bounding box coordinates
    if region:
        reg_clean = region.strip().lower()
        if reg_clean not in REGIONS and reg_clean.replace("-", "_") not in REGIONS:
            valid_keys = [r["name"] for r in get_regions_list()]
            raise HTTPException(
                status_code=400,
                detail=f"Unknown region '{region}'. Available regions: {valid_keys}",
            )
        box = get_region_bbox(reg_clean)
        w, s, e, n = box["west"], box["south"], box["east"], box["north"]
    elif bbox and "," in bbox:
        try:
            w, s, e, n = [float(x.strip()) for x in bbox.split(",")]
        except Exception:
            box = get_region_bbox(DEFAULT_REGION)
            w, s, e, n = box["west"], box["south"], box["east"], box["north"]
    else:
        box = get_region_bbox(DEFAULT_REGION)
        w, s, e, n = box["west"], box["south"], box["east"], box["north"]

    # 2. Verify trained model artifacts are present before serving
    all_ok, _, err_msg = ensure_models_loaded()
    if not all_ok:
        raise HTTPException(status_code=503, detail=err_msg)

    # 3. Load feature table snapshot or execute pipeline
    df_features = _load_or_build_feature_dataframe(w=w, s=s, e=e, n=n, since_hours=since_hours)
    if df_features.empty:
        return []

    # 4. Score and aggregate to one record per unique event_id
    return _score_and_aggregate_events(df_features)


@app.get("/hotspot/{event_id}", response_model=HotspotDetailResponse)
def get_hotspot_detail(event_id: str):
    """
    Returns the comprehensive feature vector, constituent cluster pixels,
    class probabilities, and signed SHAP attributions for a specific event_id.
    """
    if event_id not in _SCORED_CACHE:
        # Warm cache from default snapshot if cold start
        all_ok, _, err_msg = ensure_models_loaded()
        if not all_ok:
            raise HTTPException(status_code=503, detail=err_msg)
        box = get_region_bbox("all_india")
        df_all = _load_or_build_feature_dataframe(
            w=box["west"], s=box["south"], e=box["east"], n=box["north"], since_hours=24 * 365
        )
        if not df_all.empty:
            _score_and_aggregate_events(df_all)

    if event_id in _SCORED_CACHE:
        cached = _SCORED_CACHE[event_id]
        item: HotspotResponse = cached["response"]
        return HotspotDetailResponse(
            event_id=item.event_id,
            location_key=cached["location_key"],
            lat=item.lat,
            lon=item.lon,
            bbox=item.bbox,
            pixel_count=item.pixel_count,
            first_seen=item.first_seen,
            last_seen=item.last_seen,
            observed=item.observed,
            inferred=item.inferred,
            pixels=cached.get("pixels", []),
            full_feature_vector=cached["feature_vector"],
            class_probabilities=cached["class_probabilities"],
        )

    raise HTTPException(status_code=404, detail=f"Hotspot event '{event_id}' not found in active cache.")


@app.get("/facility/{location_key}/history", response_model=FacilityHistoryResponse)
def get_facility_history(location_key: str):
    """
    Returns the historical FRP time series and robust median/MAD baseline bands for a facility.
    """
    ensure_models_loaded()
    history_data = model_b.get_facility_history_band(location_key)
    return FacilityHistoryResponse(
        location_key=history_data["location_key"],
        median_frp=history_data["median_frp"],
        mad_frp=history_data["mad_frp"],
        upper_3sigma_threshold=history_data["upper_3sigma_threshold"],
        lower_3sigma_threshold=history_data["lower_3sigma_threshold"],
        sample_count=history_data["sample_count"],
        history=history_data["history"],
    )


if __name__ == "__main__":
    print(f"[API] Starting server on http://{API_HOST}:{API_PORT}")
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
