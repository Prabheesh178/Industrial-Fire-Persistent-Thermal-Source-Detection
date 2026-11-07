# Deploying the Model API to Render (`DEPLOY_RENDER.md`)

This guide explains how to prepare the trained model artifacts locally, commit them to your GitHub branch, and deploy the FastAPI service to **Render** so the Thermal Watch dashboard can query live model scores over HTTPS.

> **Operating Mode Note**: By default on Render (`LIVE_FIRMS_POLL=false`), the API serves the **trained snapshot** (`data/feature_table.csv` + `data/hotspots.db` + `data/osm_cache.json`) scored in real time through `ModelAClassifier` (LightGBM + SHAP `TreeExplainer`), `ModelBAnomalyEngine` (Median/MAD baselines), `ModelCRiskModel` (24h escalation), and `ScoreCombiner`. Live NASA FIRMS polling is an optional later toggle (`LIVE_FIRMS_POLL=true`) that requires the full geospatial stack (`requirements.txt`).

---

## 1. Required Trained Artifacts & Cache Files to Commit

Because Render's free tier uses an ephemeral filesystem rooted at your Git repository checkout, the following trained artifacts and data caches **must be committed to your repository after you run `python -m models.train_all`** locally:

| File Path (Relative to Repo Root) | Required? | Expected Size (Demo Region `gujarat`) | Expected Size (Multi-Region / `all_india`) | Purpose at Serving Time |
| :--- | :--- | :--- | :--- | :--- |
| `artifacts/model_a.joblib` | **REQUIRED** | `300 KB – 1.5 MB` | `500 KB – 3 MB` | Trained 7-class LightGBM classifier, `LabelEncoder`, and feature schema (used by `GET /hotspots` & `GET /hotspot/{id}` + `shap.TreeExplainer`). |
| `artifacts/facility_baselines.json` | **REQUIRED** | `5 KB – 50 KB` | `50 KB – 500 KB` | Per-facility and regional Median + MAD FRP baselines (used by `ModelBAnomalyEngine`). |
| `artifacts/model_c.joblib` | **REQUIRED** | `100 KB – 600 KB` | `150 KB – 1.2 MB` | Trained binary 24h escalation LightGBM model. |
| `data/feature_table.csv` | **REQUIRED** | `150 KB – 1.5 MB` | `2 MB – 12 MB` | Pre-engineered 19-feature snapshot table served by `GET /hotspots` without requiring `rasterio` or `geopandas` at runtime. |
| `data/hotspots.db` | **REQUIRED** | `150 KB – 2 MB` | `3 MB – 25 MB` | SQLite persistence log (`hotspot_history` table) powering `GET /facility/{location_key}/history` MAD baseline charts. |
| `data/osm_cache.json` | **REQUIRED** | `50 KB – 5 MB` | `15 MB – 85 MB` | Cached OpenStreetMap facilities powering `GET /facilities` GeoJSON and facility name lookups. |

### Startup & Health Safety Guard (`api/main.py`)
If you deploy to Render **before** training and committing `artifacts/model_a.joblib`, `artifacts/facility_baselines.json`, and `artifacts/model_c.joblib`:
- `GET /health` will return HTTP `200` with `"status": "error"`, `"missing_artifacts": [...]`, and a clear diagnostic message telling you which files are missing.
- `GET /hotspots` will return HTTP `503 Service Unavailable` with that same message instead of crashing or returning fake heuristic numbers.

### Commit Command After Training
Once `python -m models.train_all` finishes and prints its completion summary, force-add the required artifacts if any are matched by `.gitignore`:

```bash
git add -f \
  artifacts/model_a.joblib \
  artifacts/facility_baselines.json \
  artifacts/model_c.joblib \
  data/feature_table.csv \
  data/hotspots.db \
  data/osm_cache.json

git commit -m "chore: add trained model artifacts and snapshot caches for Render serving"
```

---

## 2. Handling GitHub's 50 MB Warning & 100 MB File Limit

GitHub warns on files over **50 MB** and rejects pushes containing any single file over **100 MB**.
For the `gujarat` demo region (or a single industrial belt), all 6 files together total **< 10 MB** and push cleanly with standard Git.

If you train on `all_india` or a large Geofabrik OSM extract where `data/osm_cache.json` or `data/hotspots.db` approaches 50–100 MB, use one of these approaches before pushing:

### Option A: Trim `data/hotspots.db` and `data/osm_cache.json` to the Served Window
1. **Compact `data/hotspots.db`** to keep only the last 30–60 days of history and reclaim SQLite pages:
   ```bash
   python -c "import sqlite3; c = sqlite3.connect('data/hotspots.db'); c.execute(\"DELETE FROM hotspot_history WHERE acq_date < date('now', '-60 days')\"); c.commit(); c.execute('VACUUM'); c.close()"
   ```
2. **Minify `data/osm_cache.json`** (strips indentation whitespace, typically reducing file size by 40–55%):
   ```bash
   python -c "import json; p='data/osm_cache.json'; d=json.load(open(p,encoding='utf-8')); json.dump(d,open(p,'w',encoding='utf-8'),separators=(',',':'))"
   ```

### Option B: Track Large Files with Git LFS
If a file still exceeds 50 MB after minification:
```bash
git lfs install
git lfs track "data/osm_cache.json" "data/hotspots.db"
git add .gitattributes
```

---

## 3. Creating the Render Web Service

This repository includes a ready-to-use [`render.yaml`](./render.yaml) Blueprint using [`requirements-serve.txt`](./requirements-serve.txt) (which installs `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`, and `joblib` in ~45 seconds without compiling `rasterio`/GDAL).

### Step-by-Step Setup on Render
1. Sign in to [https://dashboard.render.com](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **Web Service** (or **Blueprint** to auto-detect `render.yaml`).
3. Connect your GitHub repository: `Prabheesh178/Industrial-Fire-Persistent-Thermal-Source-Detection`.
4. Configure the service settings:
   - **Branch**: `integration` (or `main` once merged)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements-serve.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn api.main:app --host 0.0.0.0 --port $PORT
     ```
5. Set the following **Environment Variables** in the Render dashboard (**Environment** tab):

| Environment Variable | Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Pins Python 3.11 for prebuilt wheels of `lightgbm`, `shap`, and `scikit-learn`. |
| `CORS_ORIGINS` | `https://your-dashboard.vercel.app,http://localhost:5173` | Comma-separated list of allowed frontend origins (no trailing slashes). Use `*` during initial testing if needed. |
| `DEFAULT_REGION` | `gujarat` | Default region when `/hotspots` is called without `?region=`. |
| `LIVE_FIRMS_POLL` | `false` | Serves the committed `data/feature_table.csv` snapshot scored by the trained models. |

---

## 4. Render Free-Tier Sleep & Cold-Start Behavior

- **How Render Free Tier Behaves**: Free Render web services spin down after **15 minutes of inactivity**. The first request after an idle period triggers a cold start that takes **30 to 60 seconds** while Render boots the container and `api/main.py` loads `model_a.joblib`, `facility_baselines.json`, and `model_c.joblib`.
- **How the Dashboard Handles This**:
  - The dashboard's API adapter (`src/lib/api.ts`) uses a **60-second cold-start timeout budget** with automatic retry on initial boot.
  - If `/health` or `/hotspots` takes longer than 2.5 seconds on initial load, the dashboard header displays **`Waking the API...`** so the user knows the Render instance is spinning up, rather than prematurely dropping into offline demo mode.
  - Only if the API remains unreachable after the cold-start budget expires does the dashboard fall back to bundled demo data and show the explicit **`Demo data, API not connected`** banner.

---

## 5. Verifying the Deployed Render API

Once Render reports the deploy is **Live**, copy your service URL (e.g. `https://industrial-fire-detection-api.onrender.com`) and run these three verification checks:

### 1. Health & Artifact Check (`GET /health`)
```bash
curl -s https://industrial-fire-detection-api.onrender.com/health | python -m json.tool
```
- Confirm `"status": "ok"`, `"missing_artifacts": []`, `"data_as_of"` is a valid ISO-8601 timestamp, and `"model_artifacts"` shows `"loaded": true` with real `size_bytes` and `modified_at` timestamps for `model_a`, `facility_baselines`, and `model_c`.

### 2. Regions Registry Check (`GET /regions`)
```bash
curl -s https://industrial-fire-detection-api.onrender.com/regions | python -m json.tool
```
- Confirm it returns the 5 Indian regions (`gujarat`, `maharashtra`, `odisha`, `chhattisgarh_jharkhand`, `all_india`) with their bounding boxes.

### 3. Scored Hotspots Check (`GET /hotspots?region=gujarat`)
```bash
curl -s "https://industrial-fire-detection-api.onrender.com/hotspots?region=gujarat&since_hours=720" | python -m json.tool | head -n 60
```
- Confirm it returns aggregated hotspot events with `event_id`, `location_key`, `bbox`, `pixel_count`, `observed`, and `inferred` (including `top_shap_factors`).

### 4. Point the Dashboard at Render
In your dashboard deployment (e.g. Vercel Environment Variables) or local `.env`:
```ini
VITE_API_URL=https://industrial-fire-detection-api.onrender.com
```
Redeploy or restart the dashboard and verify the top command bar shows **LIVE** and **`Data as of <timestamp>`**.

