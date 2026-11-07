# Model Training & Artifact Portability Guide (`TRAINING_GUIDE.md`)

## Quick-Start Training Commands (`python -m models.train_all`)

Run from the root of `Industrial-Fire-Persistent-Thermal-Source-Detection` after activating your virtual environment:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
# Default demo region (Gujarat, 30 days):
python -m models.train_all --region gujarat --days 30

# Or specific region (maharashtra / odisha / chhattisgarh_jharkhand / all_india):
python -m models.train_all --region all_india --days 30
```

**Windows Command Prompt (CMD):**
```cmd
.venv\Scripts\activate.bat
REM Default demo region (Gujarat, 30 days):
python -m models.train_all --region gujarat --days 30

REM Or specific region (maharashtra / odisha / chhattisgarh_jharkhand / all_india):
python -m models.train_all --region all_india --days 30
```

**Linux / macOS (Bash / Zsh):**
```bash
source .venv/bin/activate
python -m models.train_all --region gujarat --days 30
```

> Note: `train_all.py` automatically runs a pre-flight check (`preflight_check()`) verifying folders, `.env`, `FIRMS_MAP_KEY`, and cache files before starting, and prints a completion summary (`print_completion_summary()`) with exact row counts and full absolute paths/sizes of all generated artifacts at the end.

---

This guide is written for an engineer or operator setting up and training the **Industrial Fire & Persistent Thermal Source Detection** pipeline from scratch on a clean machine, verifying the trained artifacts, and copying them to the machine or Render web service that serves the FastAPI backend.

---

## 1. Prerequisites & Environment Setup

### 1.1 Python Version
- **Python 3.10, 3.11, or 3.12** (64-bit).
- Verify your installation:
  ```bash
  python --version
  ```

### 1.2 Clone the Repository & Create a Virtual Environment

**Linux / macOS:**
```bash
git clone https://github.com/Prabheesh178/Industrial-Fire-Persistent-Thermal-Source-Detection.git
cd Industrial-Fire-Persistent-Thermal-Source-Detection
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/Prabheesh178/Industrial-Fire-Persistent-Thermal-Source-Detection.git
cd Industrial-Fire-Persistent-Thermal-Source-Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 1.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Optional (for offline Geofabrik `.osm.pbf` extraction via `tools/pbf_to_osm_cache.py`)**:
> ```bash
> pip install pyrosm
> ```

### 1.4 Environment Variables (`.env`)
Create a `.env` file in the project root (`Industrial-Fire-Persistent-Thermal-Source-Detection/.env`).

Below is the **complete list of environment variables** actually read by [`config/settings.py`](file:///d:/SIH/industrial-fire-detection/config/settings.py) (verified against all `os.getenv` calls across the codebase):

| Variable | Required? | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `FIRMS_MAP_KEY` | **Required for live FIRMS data** | `""` (empty) | Your 32-character NASA EOSDIS FIRMS Map Key. Request a free key at `https://firms.modaps.eosdis.nasa.gov/api/map_key/`. *(If omitted or left empty, `fetch_firms.py` falls back to generating a synthetic dataset).* |
| `FIRMS_MAP_KEYS` | Optional | Value of `FIRMS_MAP_KEY` | Comma-separated pool of multiple FIRMS Map Keys (e.g., `key1,key2,key3`). `fetch_firms.py` round-robins across them so each key gets its own 5,000-transactions / 10-minute budget. |
| `DEFAULT_REGION` | Optional | `gujarat` | Default named Indian region when `--region` is omitted (`gujarat`, `maharashtra`, `odisha`, `chhattisgarh_jharkhand`, `punjab_haryana`, `rajasthan_solar`, `all_india`). |
| `OVERPASS_URL` | Optional | `https://overpass-api.de/api/interpreter` | OpenStreetMap Overpass API interpreter URL used by `data_ingestion/fetch_osm.py`. |
| `OSM_CACHE_PATH` | Optional | `data/osm_cache.json` | Relative path (from repo root) where OSM industrial & exposure geometries are cached. |
| `DB_PATH` | Optional | `data/hotspots.db` | Relative path (from repo root) to the SQLite persistence database. |
| `API_HOST` | Optional | `0.0.0.0` | Host interface for the FastAPI server (`api/main.py`). |
| `API_PORT` | Optional | `8000` | Port for the FastAPI server (`api/main.py`). |

**Example `.env` file:**
```ini
FIRMS_MAP_KEY=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
DEFAULT_REGION=gujarat
DB_PATH=data/hotspots.db
OSM_CACHE_PATH=data/osm_cache.json
API_HOST=0.0.0.0
API_PORT=8000
```

> **Note on Weather & Land Cover APIs**: Neither Open-Meteo (`data_ingestion/fetch_weather.py`) nor ESA WorldCover Cloud-Optimized GeoTIFFs on AWS S3 (`data_ingestion/fetch_landcover.py`) require an API key.

---

## 2. Exact Training Command Sequence

You can run the training pipeline either via the **single-command orchestrator** (`models/train_all.py`), **step-by-step per module** (in the exact order `train_all.py` executes them), or via the **multi-region, multi-season corpus builder** (`tools/`).

### Available Named `--region` Values
Defined in [`config/settings.py`](file:///d:/SIH/industrial-fire-detection/config/settings.py):
- `gujarat` *(default demo region)*: Hazira, Dahej, Jamnagar, Surat (`69.5, 20.5, 73.8, 23.0`)
- `maharashtra`: Mumbai, MMR, Pune, Raigad, Tarapur (`72.6, 18.3, 74.5, 20.0`)
- `odisha`: Angul, Jharsuguda, Paradip, Kalinganagar (`83.5, 19.8, 87.0, 22.2`)
- `chhattisgarh_jharkhand`: Bhilai, Korba, Dhanbad, Jamshedpur (`81.0, 21.0, 86.8, 24.2`)
- `punjab_haryana`: Ludhiana, Patiala, Karnal, Hisar (`73.8, 29.3, 77.6, 32.6`)
- `rajasthan_solar`: Bhadla Solar Park & Thar Renewable Zone (`71.4, 27.1, 72.4, 28.0`)
- `all_india`: Nationwide coverage (`68.0, 6.5, 97.5, 37.5`)

---

### Option A: Single-Command Orchestrator (`models/train_all.py`)

Run from the repository root:

```bash
# Train on the default demo region (Gujarat Industrial Belt, 30 days lookback)
python -m models.train_all --region gujarat --days 30

# Or train on another Indian industrial belt
python -m models.train_all --region odisha --days 30

# Or train on All-India coverage
python -m models.train_all --region all_india --days 30
```

---

### Option B: Step-by-Step Execution (Exact Order Inside `train_all.py`)

If you want to run, inspect, or debug each stage individually, run the following commands from the repository root in this exact sequence:

#### Step 1a: Fetch NASA FIRMS Active Fire Detections (`data_ingestion/fetch_firms.py`)
```bash
python -m data_ingestion.fetch_firms --region gujarat --days 30 --output data/firms_raw.csv
```
- **What it does**: Queries NASA FIRMS in 5-day batches (automatically selecting `VIIRS_SNPP_NRT` for dates $\ge$ `2026-04-28` and `VIIRS_SNPP_SP` archive for older dates) and normalizes VIIRS letter confidences (`l`/`n`/`h` $\rightarrow$ `30.0`/`70.0`/`95.0`).
- **Produces**: `data/firms_raw.csv`

#### Step 1b: Fetch & Cache OpenStreetMap Facilities (`data_ingestion/fetch_osm.py`)
```bash
python -m data_ingestion.fetch_osm --region gujarat --output data/osm_cache.json
```
- **What it does**: Queries Overpass API with `out geom;` for industrial polygons (`works`, `industrial`, `quarry`, `petroleum_well`, `power=plant`), `landfill`/`waste_disposal`, `solar` generators/plants, populated areas (`place`, `residential`), and critical infrastructure (`pipeline`, `power=line`, `substation`).
- **Produces**: `data/osm_cache.json`

#### Step 1c: Spatial Join & Geodesic Boundary Distance (`processing/spatial_join.py`)
```bash
python -m processing.spatial_join --hotspots data/firms_raw.csv --osm data/osm_cache.json --output data/hotspots_joined.csv
```
- **What it does**: Indexes OSM polygons in `EPSG:7755` (India NSF LCC) via R-tree `sjoin_nearest` and computes exact WGS84 geodesic distance in meters to the nearest facility **boundary** (never centroid), plus `dist_to_populated_area_m` and `dist_to_critical_infra_m`.
- **Produces**: `data/hotspots_joined.csv`

#### Step 2: Spatiotemporal DBSCAN Clustering (`processing/clustering.py`)
```bash
python -m processing.clustering --input data/hotspots_joined.csv --output data/hotspots_clustered.csv
```
- **What it does**: Groups detections within $750\text{ m}$ and $12\text{ hours}$ into stable `event_id` clusters (`evt_YYYYMMDD_XXXX`) and calculates `cluster_growth_rate` **before** feature engineering.
- **Produces**: `data/hotspots_clustered.csv`

#### Step 3: Persistence Log & Recurrence Tracking (`processing/persistence_log.py`)
```bash
python -m processing.persistence_log --input data/hotspots_clustered.csv --db data/hotspots.db --output data/hotspots_persisted.csv
```
- **What it does**: Assigns `location_key` (`facility_id` if within $500\text{ m}$ of an OSM facility, else H3 resolution-8 cell ID `h3_r8_*`), upserts detections into SQLite table `hotspot_history`, and computes `days_active_last_30`, `duty_cycle`, and `night_detection_fraction`.
- **Produces**: `data/hotspots.db` (SQLite database) and `data/hotspots_persisted.csv`

#### Step 4a: Fit Model B Statistical Facility Baselines (`models/model_b_anomaly_engine.py`)
```bash
python -m models.model_b_anomaly_engine --db data/hotspots.db --artifacts artifacts/
```
- **What it does**: Computes robust Median + MAD (`z = (frp - median) / max(1.4826 * MAD, 0.1 * median, 1.0)`) per `location_key` ($\ge 10$ history points) and regional fallback baselines per `nearest_industrial_type`.
- **Produces**: `artifacts/facility_baselines.json`

#### Step 4b: Assemble Canonical Feature Table (`processing/feature_engineering.py`)
```bash
python -m processing.feature_engineering --input data/hotspots_persisted.csv --output data/feature_table.csv
```
- **What it does**: Samples 10 m ESA WorldCover land cover via HTTP range reads (`data/landcover_cache.json`), prefetches Open-Meteo ERA5/forecast weather by $0.1^\circ$ cell (`data/weather_cache.json`), computes `delta_bt` (`bright_ti4 - bright_ti5`), `frp_density` (`frp / (scan * track)`), `month`, `is_agri_burn_season`, and attaches `frp_zscore_vs_facility_baseline`.
- **Produces**: `data/feature_table.csv`, `data/landcover_cache.json`, `data/weather_cache.json`

#### Step 5: Bootstrap 7-Class Pseudo-Labels (`processing/labeling_heuristic.py`)
```bash
python -m processing.labeling_heuristic --input data/feature_table.csv --output data/labeled_dataset.csv
```
- **What it does**: Applies the 7 mutually exclusive domain rules (`industrial_flare`, `industrial_accident`, `stockpile_combustion`, `known_false_positive`, `wildfire`, `agricultural_burn`, `unknown_source`).
- **Produces**: `data/labeled_dataset.csv`

#### Step 6a: Train Model A — 7-Class LightGBM + SHAP Classifier (`models/model_a_classifier.py`)
```bash
python -m models.model_a_classifier --train --input data/labeled_dataset.csv --artifacts artifacts/
```
- **What it does**: Trains the multiclass LightGBM classifier (`n_jobs=1`, `num_leaves=31`, `lr=0.05`, early stopping on validation `multi_logloss`) and initializes `shap.TreeExplainer`.
- **Produces**: `artifacts/model_a.joblib`

#### Step 6b: Train Model C — 24h Escalation Risk Model (`models/model_c_risk_model.py`)
```bash
python -m models.model_c_risk_model --train --input data/labeled_dataset.csv --artifacts artifacts/
```
- **What it does**: Filters training rows to `duty_cycle < 0.2` (excluding steady flares), drops `days_active_last_30`, and trains the binary LightGBM escalation risk model.
- **Produces**: `artifacts/model_c.joblib`

#### Step 7: Verify Score Combiner (`models/score_combiner.py`)
```bash
python -m models.score_combiner --event_type industrial_flare --confidence 0.91 --anomaly_norm 0.12 --risk_24h 0.08 --dist_pop 1500 --dist_infra 400
```

---

### Option C: Multi-Region, Multi-Season Corpus Training (`tools/`)

A single 30-day pull from one region may not contain all 7 classes (for example, `agricultural_burn` requires the Oct–Nov or Apr–May stubble window in `punjab_haryana`, and `known_false_positive` requires solar parks in `rajasthan_solar`). To train across all regions and seasons simultaneously:

```bash
# 1. Fetch multi-region, multi-season FIRMS corpus and create balanced ~9k row sample
python tools/build_corpus.py

# 2. (Recommended if Overpass rate-limits) Build OSM cache from Geofabrik PBF extracts
python tools/pbf_to_osm_cache.py

# 3. Run full 6-stage pipeline + train Models A, B, and C on the balanced corpus
python tools/run_training.py
```

---

## 3. Summary of Produced Files & Locations

| File Path | Produced By | Purpose | Needed on API Server? |
| :--- | :--- | :--- | :--- |
| `artifacts/model_a.joblib` | `model_a_classifier.py` | Trained 7-class LightGBM model, `LabelEncoder`, and feature list | **YES (Required)** |
| `artifacts/facility_baselines.json` | `model_b_anomaly_engine.py` | Per-facility and regional Median + MAD FRP baselines | **YES (Required)** |
| `artifacts/model_c.joblib` | `model_c_risk_model.py` | Trained binary LightGBM 24h escalation model | **YES (Required)** |
| `data/hotspots.db` | `persistence_log.py` | SQLite database containing `hotspot_history` table (used by `/facility/{location_key}/history` and persistence lookups) | **YES (Required)** |
| `data/osm_cache.json` | `fetch_osm.py` / `pbf_to_osm_cache.py` | Cached OSM industrial polygons, populated places, and pipelines | **YES (Required)** |
| `data/landcover_cache.json` | `fetch_landcover.py` | Disk cache of ESA WorldCover 10 m point lookups | **Recommended** (speeds up `/hotspots`) |
| `data/weather_cache.json` | `fetch_weather.py` | Disk cache of Open-Meteo $0.1^\circ$ cell-day weather lookups | **Recommended** (speeds up `/hotspots`) |
| `data/firms_raw.csv` | `fetch_firms.py` | Raw FIRMS detections for `train_all.py` | No (training intermediate) |
| `data/feature_table.csv` | `feature_engineering.py` | 19+ column engineered feature matrix | No (training intermediate) |
| `data/labeled_dataset.csv` | `labeling_heuristic.py` | Feature matrix + pseudo-label `event_type` | No (training intermediate) |

---

## 4. Approximate Run Time & Disk Usage

| Metric | Demo Region (`--region gujarat --days 30`) | Multi-Region Balanced Corpus (`tools/build_corpus.py`) | Full All-India (`--region all_india --days 30`) |
| :--- | :--- | :--- | :--- |
| **Detections ($N$ rows)** | ~300 – 1,500 rows | ~9,200 rows (capped from ~25k raw) | ~15,000 – 35,000+ rows |
| **First Run Wall-Clock** | **~2 – 5 minutes** (cold weather/landcover caches) | **~8 – 15 minutes** (cold caches) | **~25 – 60 minutes** (cold caches + $O(N^2)$ DBSCAN matrix) |
| **Cached Re-Run Time** | **~15 – 30 seconds** | **~1 – 2 minutes** | **~5 – 12 minutes** |
| **RAM Peak (DBSCAN)** | $< 250\text{ MB}$ | $\approx 350\text{ MB}$ ($9\text{k} \times 9\text{k}$ float32 matrix) | $1.5\text{ GB} - 4.5\text{ GB}$ |
| **Disk Usage (`artifacts/` + `data/`)** | **~5 – 15 MB** | **~80 – 150 MB** (with real OSM cache) | **~150 – 350 MB** (plus ~1 GB if storing raw Geofabrik `.osm.pbf` files) |

> **Why Subsequent Runs Are 10× Faster**: `data/landcover_cache.json`, `data/weather_cache.json`, and `data/osm_cache.json` persist on disk and are automatically loaded on import.

---

## 5. Verification Checklist

### 5.1 File Existence & Size Check
After training completes, verify that all required artifacts exist and are non-empty:

```bash
ls -lh artifacts/model_a.joblib artifacts/facility_baselines.json artifacts/model_c.joblib data/hotspots.db data/osm_cache.json
```
- `artifacts/model_a.joblib` should be $> 100\text{ KB}$.
- `artifacts/model_c.joblib` should be $> 20\text{ KB}$.
- `artifacts/facility_baselines.json` should contain `"facility_baselines"` and `"regional_baselines"`.
- `data/osm_cache.json` should contain $> 100$ elements for real OSM data (`jq '.elements | length' data/osm_cache.json`).

### 5.2 Start the API Server & Verify Model Loading
Start the FastAPI service:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Check the startup console log — you must see all three confirmation lines:
```text
[API] Initializing models and baselines on startup...
[API] Model A (Event Classifier) loaded successfully.
[API] Model B (Anomaly Engine) loaded successfully.
[API] Model C (24h Risk Model) loaded successfully.
```

### 5.3 Smoke-Test API Calls
In a second terminal, run these `curl` checks:

1. **Check Service Status (`GET /`)**:
   ```bash
   curl -s http://localhost:8000/ | python -m json.tool
   ```
2. **Verify Regions Registry (`GET /regions`)**:
   ```bash
   curl -s http://localhost:8000/regions | python -m json.tool
   ```
   *Expected*: JSON array of 5 regions (`gujarat`, `maharashtra`, `odisha`, `chhattisgarh_jharkhand`, `all_india`).
3. **Verify Scored Hotspots (`GET /hotspots`)**:
   ```bash
   curl -s "http://localhost:8000/hotspots?region=gujarat&since_hours=24" | python -m json.tool | head -n 45
   ```
   *Expected*: Array of hotspot objects, each with `event_id`, `lat`, `lon`, `observed` (`frp`, `confidence`, `acq_date`, `nearest_industrial_type`, `dist_to_industrial_m`), and `inferred` (`event_type`, `event_type_confidence`, `anomaly_score_normalized`, `hazard_weight`, `exposure_factor`, `final_risk_score`, `escalating_24h`, `risk_24h`, `needs_manual_review`, `top_shap_features`).
4. **Verify Single Event Detail & SHAP Attribution (`GET /hotspot/{event_id}`)**:
   Copy any `event_id` from step 3 and run:
   ```bash
   curl -s "http://localhost:8000/hotspot/<EVENT_ID>" | python -m json.tool
   ```
   *Expected*: Returns `location_key`, `full_feature_vector` (all 19+ features), and `class_probabilities` (7 classes summing to `1.0`).
5. **Verify Facility History & MAD Baseline Band (`GET /facility/{location_key}/history`)**:
   Copy `location_key` from step 4 and run:
   ```bash
   curl -s "http://localhost:8000/facility/<LOCATION_KEY>/history" | python -m json.tool
   ```
   *Expected*: Returns `median_frp`, `mad_frp`, `upper_3sigma_threshold`, `lower_3sigma_threshold`, `sample_count`, and `history` array.

---

## 6. How to Copy Trained Artifacts to the API Serving Machine

If you train on a workstation/GPU box and deploy the FastAPI server on another machine, you do **not** need to re-train on the serving host.

### 6.1 Package the Artifacts on the Training Machine
From the repository root on the training machine:

```bash
tar -czvf trained_bundle.tar.gz \
  artifacts/model_a.joblib \
  artifacts/facility_baselines.json \
  artifacts/model_c.joblib \
  data/hotspots.db \
  data/osm_cache.json \
  data/landcover_cache.json \
  data/weather_cache.json
```

*(On Windows PowerShell)*:
```powershell
Compress-Archive -Path `
  artifacts\model_a.joblib, `
  artifacts\facility_baselines.json, `
  artifacts\model_c.joblib, `
  data\hotspots.db, `
  data\osm_cache.json, `
  data\landcover_cache.json, `
  data\weather_cache.json `
  -DestinationPath trained_bundle.zip -Force
```

### 6.2 Transfer and Unpack on the Serving Machine
```bash
# Copy to the serving machine
scp trained_bundle.tar.gz user@api-server:/path/to/Industrial-Fire-Persistent-Thermal-Source-Detection/

# On the serving machine, unpack in the repository root
cd /path/to/Industrial-Fire-Persistent-Thermal-Source-Detection
tar -xzvf trained_bundle.tar.gz
```
Then start `uvicorn api.main:app --host 0.0.0.0 --port 8000`. The startup hook will load `artifacts/model_a.joblib`, `artifacts/facility_baselines.json`, and `artifacts/model_c.joblib` immediately.

---

## 7. Common Failures & Fixes

| Symptom / Error | Root Cause | Fix |
| :--- | :--- | :--- |
| **`[FIRMS] Notice: No valid FIRMS_MAP_KEY found. Utilizing offline synthetic simulation generator...`** | `FIRMS_MAP_KEY` is missing in `.env`, empty, or starts with `your_`. | Set a valid 32-character NASA FIRMS key in `.env` (`FIRMS_MAP_KEY=...`). |
| **`[FIRMS] WARNING: key configured but zero detections returned... Returning EMPTY frame`** | Either (a) your `FIRMS_MAP_KEY` exceeded the **5,000 transactions / 10-minute** rate limit, (b) `Invalid MAP_KEY`, or (c) you queried an old date against `VIIRS_SNPP_NRT`. | 1. Wait 10 minutes for the FIRMS quota window to reset, or supply comma-separated keys in `FIRMS_MAP_KEYS=key1,key2` in `.env` (automatic round-robin).<br>2. `fetch_firms.py` automatically switches to `VIIRS_SNPP_SP` for dates before `2026-04-28`. |
| **Overpass API HTTP `429` (Too Many Requests), `504` (Gateway Timeout), or TCP Connection Refused** | `overpass-api.de` aggressively rate-limits large bounding-box queries (`all_india`) or temporarily blocks IPs after repeated retries. | **Fix 1 (Recommended)**: Build `data/osm_cache.json` offline from static Geofabrik India `.osm.pbf` extracts using `python tools/pbf_to_osm_cache.py` (zero rate limits).<br>**Fix 2**: Point `OVERPASS_URL` in `.env` to an alternative mirror or copy an existing `data/osm_cache.json` into `data/` (when `data/osm_cache.json` exists and `--force` is not passed, `fetch_osm.py` reuses the local cache without calling Overpass). |
| **`ABORT: OSM cache has only 6 elements -- that is the synthetic fallback, not real data`** (when running `tools/run_training.py`) | Overpass failed during an earlier run and `fetch_osm.py` wrote the 6-element synthetic fallback into `data/osm_cache.json`. | Delete `data/osm_cache.json` and run `python -m data_ingestion.fetch_osm --region gujarat --force` (or run `python tools/pbf_to_osm_cache.py` with Geofabrik PBFs). |
| **`[LandCover] tile unavailable ... Marking N points 'unknown'`** | Temporary network timeout reading ESA WorldCover Cloud-Optimized GeoTIFFs via GDAL `/vsicurl/` from S3. | Re-run `python -m processing.feature_engineering`. Points already fetched are persisted in `data/landcover_cache.json`, so only missing tiles are retried. |
| **`MemoryError` or slow hang in `Step 2: Spatiotemporal Clustering (DBSCAN)`** | Uncapped `all_india` pull returned $> 20,000$ rows, and `custom_spacetime_distance_matrix` allocates an $N \times N$ `float32` distance matrix in RAM. | Use `python tools/build_corpus.py` (which caps wildfire/stubble pulls to keep the balanced corpus at ~9,200 rows / ~340 MB RAM) or reduce `--days` when running on a single machine with limited RAM. |

