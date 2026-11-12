# Render & Vercel Deployment Runbook (`RENDER_STEPS.md`)

This guide outlines the exact, step-by-step instructions for deploying the **Industrial Fire & Persistent Thermal Source Detection API** to **Render**, configuring environment variables, verifying health & data mode, and updating the **Vercel** dashboard.

---

## 1. Step-by-Step Render Deployment

### 1.1 Create Web Service
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository:
   - Repository: `https://github.com/Prabheesh178/Thermal-_watch_Model`
   - Branch: `main`
4. Enter the build and runtime parameters:
   - **Name**: `thermal-watch-model-api` (or your chosen service name)
   - **Region**: Singapore or Frankfurt (closest to India / lower latency)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements-serve.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn api.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free` (512 MB RAM)
   - **Health Check Path**: `/health`

---

## 2. Environment Variables Configuration (Render)

Under the **Environment** tab of your Render service, add the following variables:

| Key | Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Pinned Python runtime version |
| `CORS_ORIGINS` | `https://thermal-watch-dashboard.vercel.app` | Vercel production frontend origin (no trailing slash) |
| `FIRMS_MAP_KEY` | `your_32_char_nasa_firms_key` | Enter your NASA FIRMS key directly in Render settings |
| `REFRESH_ENABLED` | `false` *(Stage 1)* / `true` *(Stage 2)* | Toggle automated live NRT refresh |
| `REFRESH_MINUTES` | `30` | Interval in minutes between automated FIRMS refresh cycles |
| `REFRESH_RETAIN_DAYS`| `30` | Rolling memory retention limit (in days) |

---

## 3. Two-Stage Rollout Plan

### Stage 1: Verified Historical Snapshot Launch
1. Leave `REFRESH_ENABLED=false` on initial deployment.
2. Click **Deploy Web Service**.
3. Once deployed, verify:
   ```bash
   curl -s "https://<YOUR_RENDER_SERVICE>.onrender.com/health"
   ```
   **Expected Response:**
   ```json
   {
     "status": "ok",
     "mode": "snapshot",
     "data_as_of": "2026-11-02T08:37:00Z",
     "refresh_status": "ok",
     "active_events_count": 7203
   }
   ```
4. Verify endpoints:
   - `GET /regions`: Returns list of 5 Indian industrial regions.
   - `GET /facilities`: Returns GeoJSON of OSM industrial facilities.
   - `GET /hotspots?region=gujarat`: Returns 185 clustered events with full observed/inferred split and signed SHAP attributions.

### Stage 2: Enable Automated Near-Real-Time (NRT) Refresh
1. In Render Web Service settings, switch `REFRESH_ENABLED` to `true`.
2. Click **Save Changes** (Render will trigger a redeployment).
3. The API will wake up, load the trained base snapshot into memory, and launch the background thread.
4. After ~30 seconds, query `GET /health`:
   - `mode` will show `"live"`
   - `last_refresh_at` will show the UTC completion timestamp
   - `refresh_status` will show `"ok"`
5. **Near-Real-Time Satellite Note**: NASA VIIRS thermal detections arrive ~3 hours after satellite overpass. "Live" means near-real-time automated ingestion, not instantaneous fire alarm.

---

## 4. Vercel Dashboard Configuration

1. Log in to [vercel.com](https://vercel.com) and navigate to your **thermal-watch-dashboard** project.
2. Go to **Settings** -> **Environment Variables**.
3. Add / Edit the variable:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://<YOUR_RENDER_SERVICE>.onrender.com` *(Do NOT include a trailing slash)*
   - **Target**: Check `Production` and `Preview`.
4. Go to **Deployments** tab, click **Redeploy** on the latest deployment (because Vite reads `VITE_API_URL` at build time).

---

## 5. End-to-End Smoke Test Checklist

- [ ] `GET /health` on Render returns HTTP 200 with `"status": "ok"`.
- [ ] `GET /regions` returns Indian region definitions.
- [ ] `GET /hotspots?region=gujarat` returns sorted events.
- [ ] Visiting `https://thermal-watch-dashboard.vercel.app/`:
  - [ ] Top bar displays `MODEL API CONNECTED`.
  - [ ] Sub-banner displays `National Technical Research Organisation · Smart India Hackathon 26162 · Live, updated <time>` (or `Snapshot, data as of <date>`).
  - [ ] Map renders event centroids with severity halos and industrial facility footprints.
  - [ ] Clicking any triage row opens the detail panel showing signed SHAP drivers, cluster pixel counts, and Model B MAD baseline time-series.

---

## 6. Common Failures & Fixes

| Symptom | Probable Cause | Fix |
| :--- | :--- | :--- |
| **CORS error in browser console** | `CORS_ORIGINS` on Render is missing your Vercel URL or has a trailing slash. | In Render Environment settings, set `CORS_ORIGINS=https://thermal-watch-dashboard.vercel.app` (no trailing slash). |
| **Dashboard shows "Demo data · API not connected"** | `VITE_API_URL` was not set on Vercel or app was not redeployed. | In Vercel Project Settings, add `VITE_API_URL=https://...onrender.com` and trigger a **Redeploy**. |
| **Render Web Service Out of Memory (OOM)** | Free tier 512 MB exceeded due to uncompressed OSM cache or long lookback. | `facilities_slim.json` is committed (`5.4 MB`), keeping peak RAM around ~258 MB. Ensure `REFRESH_RETAIN_DAYS` is set to `30`. |
| **Cold-start delay (~50 seconds)** | Render free tier spins down web services after 15 minutes of inactivity. | The dashboard displays an amber `WAKING THE API...` badge and waits up to 60 seconds before falling back to demo data. |
| **`refresh_status: failed: FIRMS_MAP_KEY missing`** | Key was not set under Render Environment tab. | Enter your 32-character key under `FIRMS_MAP_KEY` in Render dashboard and save. |
