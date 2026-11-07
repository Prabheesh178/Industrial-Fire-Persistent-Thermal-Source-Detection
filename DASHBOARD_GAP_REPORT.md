# Dashboard ↔ Model API Integration Gap Report (`DASHBOARD_GAP_REPORT.md`)

**Model Repository**: `Industrial-Fire-Persistent-Thermal-Source-Detection` (`d:\SIH\industrial-fire-detection`)  
**Dashboard Repository**: `Thermal-Watch-SIH-2026` (`d:\SIH\Thermal-Watch-SIH-2026`)  
**Branch**: `integration`

---

## 1. Dashboard Stack & Current Data Loading Architecture

### 1.1 Tech Stack (`package.json`)
- **Framework & Build**: React `18.3.1`, TypeScript `5.7.3`, Vite `6.1.0`
- **Styling & Motion**: Tailwind CSS `3.4.17`, `motion` (`12.23.12`), `gsap` (`3.13.0`), `lucide-react` (`0.544.0`), React Bits components (`CountUp`, `SplitFlapText`, `RotatingText`, `StrokeText`, `CursorGrid`)
- **Geospatial Engine**: `maplibre-gl` (`5.9.0`)
  - [`MapView.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/MapView.tsx): Renders national/regional surveillance map with 6 basemaps (Esri World Imagery, CARTO Dark Matter, Voyager, Positron, Topo) and automatic vector-to-raster tile fallback.
  - [`FootprintMap.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/FootprintMap.tsx): Renders 375 m VIIRS pixel footprints and downwind dispersion cone inside the detail drawer.
- **Time-Series Charting**: `recharts` (`2.15.4`)
  - [`FrpChart.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/FrpChart.tsx): Plots event FRP history against the facility's Median + MAD baseline band inside [`DetailDrawer.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/DetailDrawer.tsx).
  - [`FacilityView.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/FacilityView.tsx): Full-screen modal plotting 30d/90d facility FRP history, baseline band, and 30-day persistence duty cycle.

### 1.2 How the Dashboard Loads Data Today
1. **Base URL Configuration** ([`src/lib/api.ts`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/lib/api.ts)):
   - Reads `import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'`.
   - Defines `DEMO_MODE = import.meta.env.PROD && !import.meta.env.VITE_API_BASE`.
2. **Primary Fetch Path** ([`src/lib/useConsole.ts`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/lib/useConsole.ts)):
   - Calls `Promise.all([fetchHealth(), fetchEvents({ limit: 300 }), fetchRisk(), fetchFacilities()])` against `server/mock_api.py`.
   - When `link === 'live'`, immediately opens a Server-Sent Events connection to `new EventSource(`${API_BASE}/stream`)`.
   - **Critical Bug When Pointed at FastAPI**: Because the real FastAPI server does not have a `/stream` SSE endpoint, `EventSource` receives HTTP `404`, fires `es.onerror`, and **immediately flips `link` from `'live'` back to `'cached'`**.
3. **Offline / Synthetic Fallback**:
   - Bundled in [`src/lib/snapshot.json`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/lib/snapshot.json) (generated from [`server/scenarios.py`](file:///d:/SIH/Thermal-Watch-SIH-2026/server/scenarios.py) with 208 synthetic events across 6 hardcoded scenarios).
   - If `fetchHealth()` or `fetchEvents()` fails, `useConsole.ts` silently catches the error, populates state from `snapshot.json`, and sets `link = 'cached'`.

---

## 2. Endpoint & Route Comparison

| Capability | Dashboard Current Call (`src/lib/api.ts`) | Model API Endpoint (`api/main.py` & `OUTPUT_SCHEMA.md`) | Gap Status |
| :--- | :--- | :--- | :--- |
| **Region Picker** | *Not called (no region picker in UI)* | `GET /regions` | **Missing in Dashboard**: Need to fetch `/regions` and add a Region Picker dropdown that drives `/hotspots?region=...` and flies the map to the selected region's `bbox`. |
| **Active Hotspots List** | `GET /events?priority=&type=&limit=300` | `GET /hotspots?region={region}&bbox={bbox}&since_hours={hours}` | **Route & Query Mismatch**: Dashboard calls `/events`; Model API serves `/hotspots` with `region`, `bbox`, and `since_hours`. |
| **Single Event Detail** | `GET /events/{event_id}` | `GET /hotspot/{event_id}` | **Route Mismatch**: Dashboard calls plural `/events/{id}`; Model API serves singular `/hotspot/{event_id}`. |
| **Facility FRP History** | `GET /facility/{id}/history?days=45` | `GET /facility/{location_key}/history` | **Route Match**: Path matches (`/facility/{location_key}/history`). Model API returns all stored history points without needing `?days=`. |
| **Service Health** | `GET /health` | `GET /` | **Route & Payload Mismatch**: Dashboard calls `/health` expecting `{ status, mode, engine, server_time, active_events_count, model_versions, firms_latency }`. Model API serves `GET /` returning `{ service, status, endpoints }`. |
| **OSM Facilities GeoJSON** | `GET /facilities` | *None (stored on disk in `data/osm_cache.json`)* | **Missing Endpoint**: Dashboard map layer `'facilities'` expects a GeoJSON `FeatureCollection` of industrial sites. `fetchFacilities()` falls back to `cached.facilities()` (4 hardcoded points) if 404. |
| **24h Risk Grid GeoJSON** | `GET /risk` | *None (per-event `risk_24h` is on each hotspot)* | **Missing Endpoint**: Dashboard map layer `'risk'` (off by default) expects a GeoJSON `FeatureCollection` of risk polygons with `p_hotspot_24h`. |
| **SSE Live Stream & Replay** | `GET /stream`, `POST /replay/start`, `POST /replay/stop` | *None (REST polling architecture)* | **Architectural Mismatch**: `EventSource('/stream')` 404s against FastAPI and forces `link` to `'cached'`. Must disable or guard SSE when connected to the REST Model API. |

---

## 3. Granularity Gap: Detections vs. Clustered Events

- **Dashboard Contract**: *"Events, not pixels. One `event_id` is one marker and one queue row."* (`TriageRail.tsx` uses `key={e.event_id}` and `MapView.tsx` uses `id: e.event_id`).
- **Model API Current Behavior** ([`api/main.py` L202–L288](file:///d:/SIH/industrial-fire-detection/api/main.py#L202-L288)): `df_features` contains one row per satellite detection pixel. Multiple pixels in the same space-time DBSCAN cluster share the same `event_id` (e.g., `evt_20260904_0032`), so `/hotspots` currently returns **duplicate `event_id` entries** in the array, and `_SCORED_CACHE[event_id]` overwrites earlier pixels with the last pixel in the loop.
- **Resolution for Phase 4**: Group or deduplicate by `event_id` (aggregating `pixel_count`, `frp_total_mw`, `frp_max_mw`, `first_seen`, `last_seen`, `bbox`, and `pixels` list) in the adapter (and/or `api/main.py`) so every `event_id` is unique.

---

## 4. Field-by-Field Schema Comparison

### 4.1 Endpoint 1: `GET /regions`
- **Model API (`RegionInfo` in `api/schemas.py`)**:
  - `name: str` (`"gujarat"`, `"maharashtra"`, `"odisha"`, `"chhattisgarh_jharkhand"`, `"all_india"`)
  - `label: str`
  - `bbox: { west: float, south: float, east: float, north: float }`
  - `description: Optional[str]`
- **Dashboard (`src/lib/types.ts`)**:
  - Currently has **no `RegionInfo` type** and **no region selector UI**.
  - Instead, `App.tsx` hardcodes 6 synthetic scenario buttons (`SCENARIOS`).

---

### 4.2 Endpoint 2: `GET /hotspots` (`HotspotResponse` vs. Dashboard `EventSummary`)

#### A. Top-Level Event Fields
| Field | Dashboard `EventSummary` (`src/lib/types.ts`) | Model API `HotspotResponse` (`api/schemas.py`) | Status | Notes / Adapter Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **Event ID** | `event_id: string` | `event_id: str` | **MATCH** | Direct 1:1 match. |
| **Coordinates** | `centroid: { lat: number; lon: number }` | `lat: float`, `lon: float` | **MISMATCH (Shape)** | Wrap flat `lat`, `lon` into `centroid: { lat, lon }`. |
| **Bounding Box** | `bbox: [number, number, number, number]` | *Not returned* | **MISSING in API** | Compute `[minLon, minLat, maxLon, maxLat]` across cluster detections (or $\pm 0.0035^\circ$ around `lat, lon`). |
| **Cluster Size** | `pixel_count: number` | *Not returned* | **MISSING in API** | Count of FIRMS detections sharing `event_id` (or `1` if single detection). |
| **First Seen** | `first_seen: string` (ISO-8601 UTC) | `observed.acq_date: str` (`"YYYY-MM-DD"`) | **MISMATCH (Name & Format)** | API `HotspotResponse` only includes `observed.acq_date` (`YYYY-MM-DD`), omitting `acq_time`. Note: `useConsole.ts` L181 filters out events where `(Date.now() - new Date(e.last_seen)) / 3600_000 > filters.windowHours`! |
| **Last Seen** | `last_seen: string` (ISO-8601 UTC) | `observed.acq_date: str` (`"YYYY-MM-DD"`) | **MISMATCH (Name & Format)** | Convert `acq_date` (+ `acq_time` if exposed) to ISO-8601 UTC; adjust window filter in `useConsole.ts` relative to dataset max timestamp so historical/archived FIRMS pulls are not hidden by `Date.now()`. |
| **Triage Status** | `status: 'new' \| 'acknowledged' \| 'dismissed'` | *Not returned* | **MISSING in API** | UI-local state; adapter defaults to `'new'`. |
| **Limitations** | `limitations: string[]` | *Not returned* | **MISSING in API** | Derive deterministically from sensor/baseline properties: `bright_ti4 >= 367` $\rightarrow$ `'bright_ti4_saturated'`, `confidence < 50` $\rightarrow$ `'low_confidence'`, `dist_to_industrial_m >= 2000` $\rightarrow$ `'no_facility_match'`, sparse baseline $\rightarrow$ `'sparse_history'`, plus `'firms_nrt_latency'`. |

#### B. `observed` Section (`Observed` vs. `ObservedData`)
| Field | Dashboard `Observed` (`src/lib/types.ts`) | Model API `ObservedData` (`api/schemas.py`) | Status | Notes / Adapter Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **FRP Total** | `frp_total_mw: number` | `frp: float` | **MISMATCH (Name)** | Map `observed.frp` (or cluster sum) $\rightarrow$ `observed.frp_total_mw`. |
| **FRP Max** | `frp_max_mw: number` | `frp: float` | **MISMATCH (Name)** | Map `observed.frp` (or cluster max) $\rightarrow$ `observed.frp_max_mw`. |
| **Confidence** | `confidence: 'low' \| 'nominal' \| 'high' \| string` | `confidence: float` (`0.0 - 100.0`) | **MISMATCH (Type)** | API returns numeric percentage (`30.0`, `70.0`, `95.0`). Dashboard `CONFIDENCE_META` expects `'low'`, `'nominal'`, or `'high'` (and can also display numeric `%`). Map `< 50` $\rightarrow$ `'low'`, `< 85` $\rightarrow$ `'nominal'`, $\ge 85$ $\rightarrow$ `'high'`. |
| **Acquisition Date** | *(At top-level `first_seen`/`last_seen`)* | `acq_date: str` (`"YYYY-MM-DD"`) | **MISMATCH (Location)** | Used by adapter to populate `first_seen` and `last_seen`. |
| **Nearest Facility** | `nearest_industrial: { facility_id: string; name: string; type: string; distance_m: number } \| null` | `nearest_industrial_type: str`<br>`dist_to_industrial_m: float` | **MISMATCH (Shape & Missing ID/Name)** | In `/hotspots`, `ObservedData` has flat `nearest_industrial_type` and `dist_to_industrial_m` (no `facility_id` or `name`; `location_key` is only returned on `/hotspot/{event_id}`). When `nearest_industrial_type === 'none'` or `dist_to_industrial_m >= 2000`, `nearest_industrial` must be `null`. |
| **Brightness TI4** | `bright_ti4_max_k: number` | *Not in `/hotspots` `ObservedData`* (in `/hotspot/{id}` `full_feature_vector.brightness_ti4`) | **MISSING in `/hotspots`** | Present in `df_features["brightness_ti4"]` and `/hotspot/{id}`, but omitted from `ObservedData` in `api/schemas.py`. |
| **TI4 Saturated Flag**| `bright_ti4_saturated: boolean` | *Not returned* | **MISSING in API** | True when `brightness_ti4 >= 367.0` K (VIIRS I4 saturation threshold). |
| **Day/Night Pass** | `daynight: 'D' \| 'N' \| string` | *Not in `/hotspots` `ObservedData`* (in `/hotspot/{id}` `full_feature_vector.daynight`) | **MISSING in `/hotspots`** | Present in `df_features["daynight"]` and `/hotspot/{id}`. |
| **Sensor Name** | `sensor: string` | *Not returned* | **MISSING in API** | Constant `'VIIRS'` (375 m active fire sensor). |
| **Satellite Name** | `satellite: string` | *Not in `ObservedData`* (in `df_features["satellite"]`) | **MISSING in `/hotspots`** | `'SNPP'` / `'N'`. |
| **Detection Count** | `detection_count: number` | *Not returned* | **MISSING in API** | Count of detections in the event cluster. |
| **Land Cover** | `land_cover: string` | *Not in `/hotspots` `ObservedData`* (in `/hotspot/{id}` `full_feature_vector.land_cover_class`) | **MISSING in `/hotspots`** | Named `land_cover_class` in `full_feature_vector`. |
| **Cloud Cover %** | `cloud_cover_pct: number` | *Not in API* | **MISSING in API** | Pipeline does not ingest cloud cover percentage from Open-Meteo. Should be rendered as `n/a` or omitted rather than faked. |

#### C. `inferred` Section (`Inferred` vs. `InferredData`)
| Field | Dashboard `Inferred` (`src/lib/types.ts`) | Model API `InferredData` (`api/schemas.py`) | Status | Notes / Adapter Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **Event Class** | `event_type: EventType \| string`<br>(`industrial_incident`, `persistent_flare`, `wildfire`, `agricultural_burn`, `known_false_positive`, `unknown_source`) | `event_type: str`<br>(`industrial_accident`, `industrial_flare`, `stockpile_combustion`, `wildfire`, `agricultural_burn`, `known_false_positive`, `unknown_source`) | **MISMATCH (Class Names & Count)** | Model A has **7 classes**; Dashboard `TYPE_META` has **6 classes** with 2 renamed:<br>1. `industrial_flare` vs `persistent_flare`<br>2. `industrial_accident` vs `industrial_incident`<br>3. `stockpile_combustion` is missing from Dashboard `TYPE_META`! |
| **Class Confidence** | `event_type_confidence: number` (`0–1`) | `event_type_confidence: float` (`0–1`) | **MATCH** | Direct 1:1 match. |
| **Anomaly Score / Z** | `anomaly_z: number` (signed $\sigma$, e.g. `+4.2`) | `anomaly_score_normalized: float` (`0.0–1.0`) | **MISMATCH (Name & Units)** | API `/hotspots` returns `anomaly_score_normalized = clip(z / 6.0, 0, 1)`. Raw `z_score` (`frp_zscore_vs_facility_baseline`) is computed in `api/main.py` L228 (`z_score, is_anom = model_b.compute_z_score(...)`) and stored in `full_feature_vector`, but not included in `InferredData`. |
| **Anomaly Boolean** | `anomaly_flag: boolean` | *Not in `InferredData`* (computed in `api/main.py` as `is_anom`) | **MISSING in `InferredData`** | `is_anom` is computed by `model_b.compute_z_score()` in `api/main.py` L228 (`abs(z) > 3.0`), or `anomaly_score_normalized >= 0.5`. |
| **Novelty Boolean** | `novelty_flag: boolean` | `needs_manual_review: bool` (partial overlap) | **MISMATCH (Semantics)** | Dashboard `novelty_flag` specifically marks unmapped thermal sources (`event_type === 'unknown_source'` or `dist_to_industrial_m >= 2000`). API returns `needs_manual_review`. |
| **24h Escalation Risk**| `risk_24h: number \| null` (`0–1`) | `risk_24h: float` (`0–1`) | **MATCH** | Direct 1:1 match. |
| **Escalating 24h Flag**| *(Not in Dashboard `Inferred`)* | `escalating_24h: bool` | **IGNORED by Dashboard** | Returned by API (`risk_24h >= 0.50`); can be surfaced in drawer/badges. |
| **Hazard Weight** | *(Not in Dashboard `Inferred`)* | `hazard_weight: float` (`0–1`) | **IGNORED by Dashboard** | Returned by API; part of `final_risk_score` decomposition. |
| **Exposure Factor** | *(Not in Dashboard `Inferred`)* | `exposure_factor: float` (`0–1`) | **IGNORED by Dashboard** | Returned by API; part of `final_risk_score` decomposition. |
| **Needs Manual Review**| *(Not in Dashboard `Inferred`)* | `needs_manual_review: bool` | **IGNORED by Dashboard** | Returned by API; analyst triage routing flag. |
| **Risk / Severity** | `severity: number` (`0–100` integer) | `final_risk_score: float` (`0.0–1.0`) | **MISMATCH (Name & Scale)** | `severity = Math.round(final_risk_score * 100)`. |
| **Triage Priority** | `priority: 'HIGH' \| 'MEDIUM' \| 'LOW'` | *Not returned* | **MISSING in API** | Derive from `final_risk_score` & flags: e.g. `final_risk_score >= 0.36` or `needs_manual_review` with high hazard $\rightarrow$ `'HIGH'`, `final_risk_score >= 0.15` $\rightarrow$ `'MEDIUM'`, else `'LOW'`. |
| **SHAP Drivers** | `top_factors: { feature: string; direction: '+' \| '-'; weight: number }[]` | `top_shap_features: List[str]` | **MISMATCH (Shape)** | API returns top 3 feature names (`string[]`) without signed weights. Dashboard `DetailDrawer` expects `{ feature, direction, weight }[]`. |
| **Text Explanation** | `explanation: string` | *Not returned* | **MISSING in API** | Synthesize deterministically in adapter from API fields (`event_type`, `hazard_weight`, `anomaly_score_normalized`, `exposure_factor`, `final_risk_score`, `risk_24h`). |
| **Model Version** | `model_version: string` | *Not returned* | **MISSING in API** | Can be set to `'Model A (LGBM-7c) · Model B (MAD) · Model C (LGBM-Risk)'`. |

---

### 4.3 Endpoint 3: `GET /hotspot/{event_id}` (`HotspotDetailResponse` vs. Dashboard `EventDetail`)

| Field | Dashboard `EventDetail` (`src/lib/types.ts`) | Model API `HotspotDetailResponse` (`api/schemas.py`) | Status | Notes / Adapter Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **Base Summary** | Extends `EventSummary` | `event_id`, `location_key`, `lat`, `lon`, `observed`, `inferred` | **PARTIAL** | Includes `location_key: str` (facility ID or `h3_r8_*`), which links directly to `/facility/{location_key}/history`. |
| **Full Feature Vector**| *(Not on `EventDetail` directly)* | `full_feature_vector: Dict[str, Any]` | **EXTRA in API (Used by Adapter)** | Contains all 19+ features (`brightness_ti4`, `brightness_ti5`, `delta_bt`, `frp_density`, `daynight`, `land_cover_class`, `temperature`, `humidity`, `wind_speed`, `days_active_last_30`, `night_detection_fraction`, `frp_zscore_vs_facility_baseline`, `dist_to_populated_area_m`, `dist_to_critical_infra_m`, `cluster_growth_rate`). |
| **Class Probabilities**| *(Not on `EventDetail`)* | `class_probabilities: Dict[str, float]` | **IGNORED by Dashboard** | 7-class probability distribution from Model A; can be displayed in `DetailDrawer` inference section. |
| **FRP Time Series** | `frp_series: FrpPoint[]` (`{ ts, frp_mw, satellite, daynight, anomalous }[]`) | *Served on `GET /facility/{location_key}/history`* | **ENDPOINT SPLIT** | Adapter fetches `/hotspot/{event_id}` to get `location_key`, then fetches `/facility/{location_key}/history` to populate `frp_series` and `baseline`. |
| **Baseline Band** | `baseline: Baseline \| null` (`{ median_mw, band_low_mw, band_high_mw, n_history, source }`) | *Served on `GET /facility/{location_key}/history`* | **ENDPOINT SPLIT** | Mapped from `FacilityHistoryResponse` (`median_frp`, `lower_3sigma_threshold`, `upper_3sigma_threshold`, `sample_count`). |
| **Footprint Pixels** | `pixels: { lat: number; lon: number; frp_mw: number; temp_k: number }[]` | Single `lat`, `lon` + `full_feature_vector.frp`, `brightness_ti4` | **MISMATCH (Shape)** | Adapter constructs `pixels` array from event detections. |
| **Wind Vector** | `wind: { speed_kmh: number; bearing_deg: number; downwind_cone?: [number, number][] }` | `full_feature_vector.wind_speed: float` | **PARTIAL** | `speed_kmh` comes from `full_feature_vector.wind_speed` (Open-Meteo 10m wind speed). `bearing_deg` is not fetched by `fetch_weather.py`. |
| **Pop. Exposure** | `population_5km_exposure: number` | `full_feature_vector.dist_to_populated_area_m: float` | **MISMATCH (Metric)** | API computes `dist_to_populated_area_m` (meters to nearest residential/place polygon) and `inferred.exposure_factor` (`0–1`), not population headcount. |
| **Optical Imagery** | `imagery: { before_url, after_url, acquired_at } \| null` | *Not returned* | **MATCH (`null`)** | Dashboard `DetailDrawer.tsx` has a built-in empty state for `imagery: null`. |
| **Processing Trail** | `timeline: { ts: string; event: string }[]` | *Not returned* | **MISSING in API** | Adapter constructs from actual `acq_date`, `location_key`, `event_type`, and `final_risk_score`. |

---

### 4.4 Endpoint 4: `GET /facility/{location_key}/history` (`FacilityHistoryResponse` vs. Dashboard `FacilitySeries`)

| Field | Dashboard `FacilitySeries` (`src/lib/types.ts`) | Model API `FacilityHistoryResponse` (`api/schemas.py`) | Status | Notes / Adapter Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **Facility Key** | `facility_id: string` | `location_key: str` | **MISMATCH (Name)** | Map `location_key` $\rightarrow$ `facility_id`. |
| **Facility Name** | `facility_name: string` | *Not returned* | **MISSING in API** | Use `location_key` or matched facility label. |
| **History Count** | `n_history: number` | `sample_count: int` | **MISMATCH (Name)** | Map `sample_count` $\rightarrow$ `n_history`. |
| **Baseline Source** | `source: string` (`'facility'` \| `'regional'`) | *Not returned* | **MISSING in API** | Per Model B rule (`model_b_anomaly_engine.py` L148): `'facility'` if `sample_count >= 10` else `'regional'`. |
| **Baseline Object** | `baseline: { median_mw: number; band_low_mw: number; band_high_mw: number; iqr_mw?: number } \| null` | Flat fields:<br>• `median_frp: float`<br>• `mad_frp: float`<br>• `lower_3sigma_threshold: float`<br>• `upper_3sigma_threshold: float` | **MISMATCH (Flat vs Nested & Field Names)** | Map:<br>• `median_frp` $\rightarrow$ `baseline.median_mw`<br>• `lower_3sigma_threshold` $\rightarrow$ `baseline.band_low_mw`<br>• `upper_3sigma_threshold` $\rightarrow$ `baseline.band_high_mw`<br>• `mad_frp` $\rightarrow$ `baseline.iqr_mw`<br>(If `sample_count === 0`, `baseline: null`). |
| **Persistence Metrics**| `days_active_last_30?: number`<br>`duty_cycle_30d?: number` | *Not in `FacilityHistoryResponse`* | **MISSING in Endpoint** | Compute directly from `history[].date`: count distinct dates within last 30 days (`days_active_last_30`) and divide by `30.0` (`duty_cycle_30d`). |
| **History Array** | `history: { timestamp: string; frp_mw: number; z_score: number; median_mw?: number; band_low_mw?: number; band_high_mw?: number; satellite?: string }[]` | `history: { date: str; time: str; frp: float; confidence: float; daynight: str }[]` | **MISMATCH (Field Names)** | For each point in `history`:<br>• `timestamp` = `${p.date}T${p.time.padStart(4,'0').slice(0,2)}:${p.time.padStart(4,'0').slice(2,4)}:00Z`<br>• `frp_mw` = `p.frp`<br>• `z_score` = `(p.frp - median_frp) / Math.max(1.4826 * mad_frp, 0.1 * median_frp, 1.0)`<br>• `daynight` = `p.daynight` |

---

## 5. Hardcoded Strings & Values in the Dashboard to Remove/Replace in Phase 4

1. **[`src/App.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/App.tsx#L29-L36)**:
   - Hardcoded `SCENARIOS` array (`Jamnagar 11.3σ`, `NTRO novelty`, `Odisha unmapped`, `Similipal wildfire`, `Routine flare`, `Solar false positive`). Will be replaced/driven by the live `/regions` picker (and only shown in fallback demo mode if applicable).
2. **[`src/App.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/App.tsx#L189-L204) & [`src/components/MobileShell.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/MobileShell.tsx#L86)**:
   - Offline banner currently reads `'Hosted demo — bundled synthetic snapshot, 208 events'` / `'Service unreachable — showing the cached synthetic snapshot'`. Will be updated to explicitly show **"Demo data, API not connected"** whenever falling back to demo/snapshot data.
3. **[`src/components/CommandBar.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/CommandBar.tsx#L39)**:
   - Classification strip hardcodes `'Synthetic demonstration data'` even when connected to the live API. Will be dynamic (`Live API telemetry` when `link === 'live'`, `'Demo data, API not connected'` when fallback).
4. **[`src/components/CommandBar.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/CommandBar.tsx#L171-L174) & [`src/components/MobileShell.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/MobileShell.tsx#L277-L280)**:
   - Hardcoded fallback `'models A 1.0.2 · B 1.0.0 · C 1.0.1'`.
5. **[`src/components/DetailDrawer.tsx`](file:///d:/SIH/Thermal-Watch-SIH-2026/src/components/DetailDrawer.tsx#L123)**:
   - Exported text brief header hardcodes `THERMAL WATCH — EVENT BRIEF (synthetic demonstration data)`.

---

## 6. Recommended Phase 4 Integration Plan (Minimal API Surface Additions + Dashboard Adapter)

To keep all model logic, features, thresholds, class definitions, and `ScoreCombiner` 100% untouched while giving the dashboard every field it needs without fabricating numbers:

1. **Optional Lightweight Additions in `api/` (`api/schemas.py` & `api/main.py`)**:
   - In `/hotspots`, aggregate or deduplicate by `event_id` (or attach `location_key` and `full_feature_vector` / `z_score` / `is_anomalous` / `acq_time` / `daynight` / `brightness_ti4` / `land_cover_class` to `HotspotResponse`) so `/hotspots` has rich observed & inferred context immediately on load without requiring $N$ separate calls to `/hotspot/{event_id}`.
   - Add CORS configuration driven by `CORS_ORIGINS` env var (defaulting to `*` or local Vite dev + production origins).
2. **Single Data Adapter in Dashboard (`src/lib/api.ts`)**:
   - Read `VITE_API_URL` (with fallback to `VITE_API_BASE` and `http://localhost:8000`) and provide `.env.example`.
   - Implement `fetchRegions()` calling `GET /regions`.
   - Implement `fetchEvents(region)` calling `GET /hotspots?region={region}&since_hours={hours}` and mapping `HotspotResponse[]` $\rightarrow$ `EventSummary[]`.
   - Implement `fetchEventDetail(eventId)` calling `GET /hotspot/{event_id}` + `GET /facility/{location_key}/history` and mapping to `EventDetail`.
   - Implement `fetchFacilityHistory(locationKey)` calling `GET /facility/{location_key}/history` and mapping to `FacilitySeries`.
   - Support all 7 canonical model classes in `src/lib/design.ts` (`industrial_flare`, `industrial_accident`, `stockpile_combustion`, `wildfire`, `agricultural_burn`, `known_false_positive`, `unknown_source`), while keeping backwards compatibility with the 6 snapshot class names when in offline demo mode.
   - Surface the exact `observed` (`frp`, `confidence`, `acq_date`, `nearest_industrial_type`, `dist_to_industrial_m`) and `inferred` (`event_type`, `event_type_confidence`, `anomaly_score_normalized`, `hazard_weight`, `exposure_factor`, `final_risk_score`, `escalating_24h`, `risk_24h`, `needs_manual_review`, `top_shap_features`) fields in `DetailDrawer.tsx` with strict visual separation.
