# Thermal Watch API Service (Render Deployment)

Production REST API service for the **Industrial Fire & Persistent Thermal Source Detection** platform, developed for the Smart India Hackathon (NTRO problem statement).

This repository is a self-contained runtime deployment package designed specifically for **Render's Free Web Service tier** (512 MB RAM ceiling). It contains the pre-trained LightGBM classifiers, robust MAD anomaly baseline models, the live NASA FIRMS background refresh daemon, and lightweight serving geometries.

---

## 1. System Architecture & Serving Mode

The API operates in one of two modes:
1. **Verified Historical Snapshot Mode (`REFRESH_ENABLED=false`)**: Serves the 7,203 pre-clustered satellite thermal events from the October/November 2026 all-India training baseline.
2. **Near-Real-Time Live Refresh Mode (`REFRESH_ENABLED=true`)**: Automatically spins up a background thread on startup that queries the NASA FIRMS Area API every `REFRESH_MINUTES` (default 30 min) for new VIIRS NRT detections, enriches them with spatial and weather features, runs Model A/B/C inference, and merges them into active memory.

> **Latency Note**: NASA VIIRS NRT data is processed and delivered approximately 3 hours after satellite overpass. "Live" indicates near-real-time automated ingestion rather than instantaneous telemetry.

---

## 2. API Endpoints

Base URL: `http://localhost:8000` (or `https://<YOUR_RENDER_SERVICE>.onrender.com`)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Operational status (`ok`/`error`), active mode (`snapshot`/`live`), `data_as_of` timestamp, `last_refresh_at`, `refresh_status`, event count, and model artifact inspection. |
| `/regions` | `GET` | List of supported Indian industrial belts (`gujarat`, `maharashtra`, `odisha`, `chhattisgarh_jharkhand`, `all_india`) with bounding boxes. |
| `/facilities` | `GET` | GeoJSON FeatureCollection of 19,250 industrial facilities (refineries, power plants, chemical works, landfills, mines, solar farms). |
| `/hotspots` | `GET` | Query parameters: `region` (or `bbox`) and `since_hours`. Returns aggregated thermal events with strictly separated **Observed** (satellite evidence) and **Inferred** (model predictions, risk score, signed SHAP drivers). |
| `/hotspot/{event_id}` | `GET` | Full 29-feature diagnostic vector, constituent cluster pixels, and class probability distribution. |
| `/facility/{location_key}/history` | `GET` | Historical FRP time series with Median and 3-sigma MAD anomaly threshold band. |

---

## 3. Local Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)

### Installation & Startup
```bash
# Clone or enter the deploy directory
cd thermalwatch-api-deploy

# Install lightweight runtime dependencies
pip install -r requirements-serve.txt

# Start the server on port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 4. Environment Variables

| Variable | Required? | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `PYTHON_VERSION` | Recommended | `3.11.9` | Pinned Python runtime version on Render. |
| `CORS_ORIGINS` | Recommended | `*` | Comma-separated allowed frontend origins (e.g. `https://thermal-watch-dashboard.vercel.app`). |
| `FIRMS_MAP_KEY` | For Live Refresh | `""` | 32-character NASA EOSDIS FIRMS Map Key. |
| `REFRESH_ENABLED` | Optional | `false` | Set to `true` to enable automated near-real-time VIIRS ingestion. |
| `REFRESH_MINUTES` | Optional | `30` | Interval in minutes between background polling cycles. |
| `REFRESH_RETAIN_DAYS`| Optional | `30` | Rolling in-memory event retention horizon to protect RAM. |

---

## 5. Deployment Guide

For full Render setup instructions, environment variable values, and Vercel dashboard integration steps, refer to [DEPLOY_RENDER.md](DEPLOY_RENDER.md) and [OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md).
