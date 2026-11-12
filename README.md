# Industrial Fire & Persistent Thermal Source Detection System

An end-to-end operational data pipeline, statistical anomaly detection engine, LightGBM machine learning classifier, and FastAPI service for satellite-based industrial fire, refinery flare, and unauthorized thermal source monitoring using NASA FIRMS (VIIRS NRT), OpenStreetMap (Overpass), ESA WorldCover land cover point sampling, and Open-Meteo weather data.

---

## 1. Quick Start Guide

### Step 1: Environment Setup
Clone or copy the project to your training machine and install Python dependencies:
```bash
cd industrial-fire-detection
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and set your NASA FIRMS Map Key (free signup at [NASA EOSDIS FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/)):
```env
FIRMS_MAP_KEY=your_32_character_nasa_firms_map_key
DB_PATH=data/hotspots.db
DEFAULT_REGION=gujarat
```
*(Note: If `FIRMS_MAP_KEY` is omitted, the pipeline automatically provides a realistic synthetic industrial flare/wildfire data generator so that testing is immediately functional offline).*

### Step 3: Run the Complete Training Pipeline
Execute the 7-step data ingestion, clustering, persistence logging, feature engineering, and model training sequence for an Indian industrial belt:
```bash
# Default: Gujarat Industrial Belt
python -m models.train_all

# Or select any predefined Indian industrial belt (gujarat, maharashtra, odisha, chhattisgarh_jharkhand, all_india)
python -m models.train_all --region maharashtra --days 30
python -m models.train_all --region odisha --days 30
python -m models.train_all --region all_india --days 14
```

### Step 5: Export Runtime Serving Bundle for Render
To export a clean, minimal runtime bundle containing only what the FastAPI service and automated FIRMS refresh engine need (without training data or uncompressed OSM caches):
```bash
python export_deploy_bundle.py
```
This generates `../thermalwatch-api-deploy` containing all model artifacts, slim facilities, and serving code (~13.6 MB), ready for deployment to Render.

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
`http://localhost:8000/docs`

---

## 2. Technical Architecture & Strict Design Rules

### 1. Ingestion Layer
- **NASA FIRMS**: Uses the Area API with `VIIRS_SNPP_NRT`. Requests are batched in weekly increments (<= 7 days) to strictly avoid hitting the 5,000 transaction per 10-minute rate limit.
- **OpenStreetMap Overpass API**: Query strictly enforces `out geom;` (never `out center;`) and includes `relation[...]` clauses alongside `way[...]` to extract true multipolygon facility boundaries. Results are fetched once and cached in `data/osm_cache.json`.
- **Land Cover**: Point-samples ESA WorldCover via lightweight cloud REST queries per coordinate. **Never bulk-downloads raster tiles** (eliminating tens of gigabytes of wasted storage).
- **Weather**: Open-Meteo REST API pulls `temperature`, `humidity`, and `wind_speed` per hotspot coordinate and acquisition date.

### 2. Spatial Processing & Persistence
- **Boundary Distance (Not Centroid)**: Distance from each hotspot to industrial facilities is calculated to the polygon boundary via Shapely in projected metric space (`EPSG:3857`). Centroids are never used because large industrial complexes (e.g. Jamnagar/Hazira) span kilometers.
- **Spatiotemporal Clustering (DBSCAN)**: Executed **before** feature extraction. Groups adjacent detections using a combined metric ($\approx 750\text{m}$ spatial radius, 12-hour temporal window) to assign stable `event_id` identifiers and calculate `cluster_growth_rate`.
- **Persistence History Log (`db/schema.sql`)**: Keys on `location_key` (`facility_id` if within 500m of a known OSM facility, else an **H3 resolution-8 cell ID**). Naive degree-rounding is strictly forbidden to prevent coordinate jitter from scattering a single facility's history across cells. Computes `days_active_last_30`, `duty_cycle`, and `night_detection_fraction` (fraction of detections with `daynight == 'N'`).

### 3. Machine Learning & Statistical Models
- **Model A (LightGBM Multi-Class Event Classifier)**:
  - 7 exact classes: `industrial_flare`, `industrial_accident`, `stockpile_combustion`, `known_false_positive`, `wildfire`, `agricultural_burn`, `unknown_source`.
  - Feature set: All features except `frp_zscore_vs_facility_baseline`.
  - Explainability: Uses `shap.TreeExplainer` for per-prediction feature attribution (`top_shap_features`).
- **Model B (Statistical Facility Anomaly Engine)**:
  - Robust statistical baseline: Tracks rolling Median and MAD (Median Absolute Deviation) per facility.
  - Strict denominator floor formula:
    $$z = \frac{\text{frp} - \text{median}}{\max(1.4826 \times \text{MAD},\; 0.1 \times \text{median},\; 1.0)}$$
  - Minimum sample gate: Flags as anomalous only if $|z| > 3$ and facility has $\ge 10$ history points; otherwise falls back to regional baseline of the same OSM type.
- **Model C (LightGBM 24h Risk Escalation Model)**:
  - Training scope is filtered to `duty_cycle < 0.2` (excludes steady flaring facilities to prevent target leakage).
  - Target = 1 if max FRP within next 24h exceeds $2\times$ current reading OR cluster footprint grows (`cluster_growth_rate > 0`).
  - Feature set explicitly drops `days_active_last_30` to prevent target leakage.
- **Score Combiner**:
  $$\text{final\_risk\_score} = \text{hazard\_weight}[\text{event\_type}] \times \text{anomaly\_score\_normalized} \times \text{exposure\_factor}$$
  - Model A confidence and Model C risk_24h are **never multiplied or summed** into `final_risk_score`. They are surfaced as distinct triage flags: `needs_manual_review` and `escalating_24h`.

---

## 3. Directory Structure & Per-File "How to Run" Guide

```
industrial-fire-detection/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.py
├── data_ingestion/
│   ├── __init__.py
│   ├── fetch_firms.py
│   ├── fetch_osm.py
│   ├── fetch_landcover.py
│   └── fetch_weather.py
├── processing/
│   ├── __init__.py
│   ├── spatial_join.py
│   ├── clustering.py
│   ├── persistence_log.py
│   ├── feature_engineering.py
│   └── labeling_heuristic.py
├── models/
│   ├── __init__.py
│   ├── model_a_classifier.py
│   ├── model_b_anomaly_engine.py
│   ├── model_c_risk_model.py
│   ├── score_combiner.py
│   └── train_all.py
├── api/
│   ├── __init__.py
│   ├── schemas.py
│   └── main.py
├── db/
│   └── schema.sql
└── artifacts/
    └── .gitkeep
```

---

### Per-File Execution Reference

#### 1. Configuration & Database
- **`config/settings.py`**
  - **Command**: `python -m config.settings`
  - **Inputs**: `.env` file variables (`FIRMS_MAP_KEY`, `DB_PATH`, `DEMO_BBOX_*`).
  - **Output**: Validates and prints loaded configuration, bounding boxes, and model constants.
- **`db/schema.sql`**
  - **Command**: `sqlite3 data/hotspots.db < db/schema.sql` (or auto-initialized by Python).
  - **Inputs**: Database path.
  - **Output**: Creates the `hotspot_history` table with composite primary key `(location_key, acq_date, acq_time)`.

#### 2. Data Ingestion (`data_ingestion/`)
- **`data_ingestion/fetch_firms.py`**
  - **Command**: `python -m data_ingestion.fetch_firms --region gujarat --days 14 --output data/firms_raw.csv`
  - **Inputs**: `region` (e.g. `gujarat`, `maharashtra`, `odisha`, `all_india`) or raw coordinates, `FIRMS_MAP_KEY`, day range.
  - **Output**: CSV containing NASA VIIRS active fire observations.
- **`data_ingestion/fetch_osm.py`**
  - **Command**: `python -m data_ingestion.fetch_osm --region gujarat --output data/osm_cache.json`
  - **Inputs**: `region` key, Overpass API URL, cached GeoJSON output path.
  - **Output**: Cached GeoJSON/JSON containing full OSM geometries (`out geom;`).
- **`data_ingestion/fetch_landcover.py`**
  - **Command**: `python -m data_ingestion.fetch_landcover --lat 21.17 --lon 72.83`
  - **Inputs**: Latitude / Longitude coordinates.
  - **Output**: Standardized land cover class string (`forest`, `cropland`, `built_up`, etc.).
- **`data_ingestion/fetch_weather.py`**
  - **Command**: `python -m data_ingestion.fetch_weather --lat 21.17 --lon 72.83 --date 2026-09-04`
  - **Inputs**: Coordinate and date string (YYYY-MM-DD).
  - **Output**: Temperature (°C), relative humidity (%), and wind speed (km/h).

#### 3. Processing & Persistence (`processing/`)
- **`processing/spatial_join.py`**
  - **Command**: `python -m processing.spatial_join --hotspots data/firms_raw.csv --osm data/osm_cache.json --output data/hotspots_joined.csv`
  - **Inputs**: Hotspots CSV and cached OSM geometries.
  - **Output**: Enriched CSV with exact boundary distances (`dist_to_nearest_industrial_m`, `dist_to_populated_area_m`, `dist_to_critical_infra_m`).
- **`processing/clustering.py`**
  - **Command**: `python -m processing.clustering --input data/hotspots_joined.csv --output data/hotspots_clustered.csv`
  - **Inputs**: Joined hotspots CSV.
  - **Output**: CSV with assigned `event_id` and `cluster_growth_rate`.
- **`processing/persistence_log.py`**
  - **Command**: `python -m processing.persistence_log --input data/hotspots_clustered.csv --db data/hotspots.db`
  - **Inputs**: Clustered hotspots and SQLite database path.
  - **Output**: Populates `hotspot_history` table and computes `days_active_last_30`, `duty_cycle`, and `night_detection_fraction`.
- **`processing/feature_engineering.py`**
  - **Command**: `python -m processing.feature_engineering --input data/hotspots_persisted.csv --output data/feature_table.csv`
  - **Inputs**: Persisted hotspots.
  - **Output**: Full canonical feature table ready for modeling.
- **`processing/labeling_heuristic.py`**
  - **Command**: `python -m processing.labeling_heuristic --input data/feature_table.csv --output data/labeled_dataset.csv`
  - **Inputs**: Feature table CSV.
  - **Output**: Dataset labeled with the 7 ground-truth pseudo-label classes.

#### 4. Models & Orchestration (`models/`)
- **`models/model_a_classifier.py`**
  - **Command**: `python -m models.model_a_classifier --train --input data/labeled_dataset.csv`
  - **Inputs**: Labeled dataset CSV.
  - **Output**: Trained LightGBM multi-class model and SHAP TreeExplainer saved to `artifacts/model_a.joblib`.
- **`models/model_b_anomaly_engine.py`**
  - **Command**: `python -m models.model_b_anomaly_engine --db data/hotspots.db`
  - **Inputs**: `hotspot_history` table in SQLite.
  - **Output**: Robust Median + MAD baselines saved to `artifacts/facility_baselines.json`.
- **`models/model_c_risk_model.py`**
  - **Command**: `python -m models.model_c_risk_model --train --input data/feature_table.csv`
  - **Inputs**: Feature table CSV (filters to duty_cycle < 0.2 and drops `days_active_last_30`).
  - **Output**: Trained LightGBM 24h risk model saved to `artifacts/model_c.joblib`.
- **`models/score_combiner.py`**
  - **Command**: `python -m models.score_combiner --event_type industrial_flare --z_score 1.2 --dist_pop 500 --dist_infra 200`
  - **Inputs**: Model A, B, C outputs and spatial exposure distances.
  - **Output**: Synthesized `final_risk_score` and routing flags (`needs_manual_review`, `escalating_24h`).
- **`models/train_all.py`**
  - **Command**: `python -m models.train_all --region gujarat --days 30`
  - **Inputs**: Indian region key (`--region`) or manual bounding box coordinates and lookback window.
  - **Output**: Runs the full 7-step pipeline and writes all trained artifacts to `artifacts/`.

#### 5. API Layer (`api/`)
- **`api/schemas.py`**
  - **Command**: `python -m api.schemas`
  - **Output**: Validates Pydantic response schemas.
- **`api/main.py`**
  - **Command**: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
  - **Output**: Serves REST endpoints with top-level `observed` vs `inferred` separation:
    - `GET /regions` (list available Indian industrial belts)
    - `GET /hotspots?region=gujarat&since_hours=24` (or `?bbox=...`)
    - `GET /hotspot/{event_id}`
    - `GET /facility/{location_key}/history`

---

## 4. API Response Schema Example

```json
{
  "event_id": "evt_20260904_0032",
  "lat": 21.17,
  "lon": 72.83,
  "observed": {
    "frp": 42.3,
    "confidence": 87.0,
    "acq_date": "2026-09-04",
    "nearest_industrial_type": "refinery",
    "dist_to_industrial_m": 210.0
  },
  "inferred": {
    "event_type": "industrial_flare",
    "event_type_confidence": 0.91,
    "anomaly_score_normalized": 0.12,
    "hazard_weight": 0.1,
    "exposure_factor": 0.4,
    "final_risk_score": 0.048,
    "escalating_24h": false,
    "risk_24h": 0.08,
    "needs_manual_review": false,
    "top_shap_features": [
      "dist_to_nearest_industrial_m",
      "night_detection_fraction",
      "daynight"
    ]
  }
}
```
