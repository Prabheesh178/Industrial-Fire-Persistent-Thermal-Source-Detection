# Frontend & Dashboard Integration Guide: API Output Schema

This document serves as the complete technical contract for frontend and dashboard developers integrating against the **Industrial Fire & Persistent Thermal Source Detection API**.

---

## 1. Core Design Principle: Observed vs. Inferred

All hotspot payload objects strictly separate **Satellite Evidence (Observed)** from **Model AI Predictions (Inferred)**:
- `observed`: Direct, un-manipulated satellite & GIS sensor data (FRP peak/total/max, I4 brightness temperature, satellite confidence, acquisition date & UTC time, day/night pass, satellite sensor, ESA WorldCover land cover, and nearest OSM facility boundary distance, category, and name).
- `inferred`: Synthesized intelligence produced by the machine learning classifiers and anomaly engine (event classification, signed z-score & boolean anomaly gate, normalized anomaly score, hazard weighting, spatial exposure factor, final risk score, 24h escalation probability, and signed SHAP feature drivers).

> **Frontend Implementation Tip**: Visually delineate these two sections in the UI and never merge observed satellite measurements with model inferences in the same table.

---

## 2. API Endpoints

Base URL: `http://localhost:8000` (or your Render deployment URL)  
CORS: Controlled via `CORS_ORIGINS` environment variable (comma-separated origins)  
Format: `application/json`

---

### Endpoint 1: `GET /health`

Returns operational service status, dataset currency (`data_as_of`), active cluster count, and dynamically inspected model artifact file metadata (size and UTC modified timestamp read from disk — no hardcoded versions).

- If required model artifacts (`artifacts/model_a.joblib`, `artifacts/facility_baselines.json`, `artifacts/model_c.joblib`) are missing, `status` is `"error"`, `missing_artifacts` lists each missing file, and `GET /hotspots` returns HTTP `503` with the same message instead of serving fake or heuristic numbers.

#### Response Schema

```typescript
interface ArtifactInfo {
  filename: string;
  exists: boolean;
  loaded: boolean;
  size_bytes: number | null;
  modified_at: string | null; // ISO-8601 UTC timestamp from filesystem mtime
}

interface HealthResponse {
  status: "ok" | "error";
  error: string | null;
  missing_artifacts: string[];
  data_as_of: string | null;  // Max acquisition timestamp in dataset (ISO-8601 UTC)
  active_events_count: number;
  model_artifacts: Record<string, ArtifactInfo>;
}
```

#### Sample Response Payload (`GET /health`)

```json
{
  "status": "ok",
  "error": null,
  "missing_artifacts": [],
  "data_as_of": "2026-09-04T20:42:00Z",
  "active_events_count": 18,
  "model_artifacts": {
    "model_a": {
      "filename": "model_a.joblib",
      "exists": true,
      "loaded": true,
      "size_bytes": 418920,
      "modified_at": "2026-11-07T12:30:00Z"
    },
    "facility_baselines": {
      "filename": "facility_baselines.json",
      "exists": true,
      "loaded": true,
      "size_bytes": 14820,
      "modified_at": "2026-11-07T12:29:50Z"
    },
    "model_c": {
      "filename": "model_c.joblib",
      "exists": true,
      "loaded": true,
      "size_bytes": 182410,
      "modified_at": "2026-11-07T12:30:05Z"
    },
    "hotspots_db": {
      "filename": "hotspots.db",
      "exists": true,
      "loaded": true,
      "size_bytes": 245760,
      "modified_at": "2026-11-07T12:29:45Z"
    },
    "osm_cache": {
      "filename": "osm_cache.json",
      "exists": true,
      "loaded": true,
      "size_bytes": 84120,
      "modified_at": "2026-11-07T12:29:40Z"
    }
  }
}
```

---

### Endpoint 2: `GET /regions`

Returns the registry of available predefined Indian industrial regions (name, human-readable label, bounding box, and industrial profile description) so that the frontend dashboard can render a dynamic region-picker dropdown.

#### Response Schema (Array of Region Objects)

```typescript
interface RegionInfo {
  name: string;        // Unique identifier key (e.g. "gujarat", "maharashtra", "odisha", "all_india")
  label: string;       // Human-friendly title and coverage cities
  bbox: {
    west: number;      // Western longitude in degrees
    south: number;     // Southern latitude in degrees
    east: number;      // Eastern longitude in degrees
    north: number;     // Northern latitude in degrees
  };
  description: string; // Summary of key industries, refineries, and installations in the region
}
```

#### Sample Response Payload (`GET /regions`)

```json
[
  {
    "name": "gujarat",
    "label": "Gujarat Industrial Belt (Hazira, Dahej, Jamnagar, Surat)",
    "bbox": {
      "west": 69.5,
      "south": 20.5,
      "east": 73.8,
      "north": 23.0
    },
    "description": "Petrochemical refineries, LNG terminals, ports, and heavy chemical clusters in Gujarat."
  },
  {
    "name": "maharashtra",
    "label": "Maharashtra Industrial Belt (Mumbai, MMR, Pune, Raigad, Tarapur)",
    "bbox": {
      "west": 72.6,
      "south": 18.3,
      "east": 74.5,
      "north": 20.0
    },
    "description": "Chemical corridors, manufacturing MIDCs, and energy installations in Maharashtra."
  },
  {
    "name": "odisha",
    "label": "Odisha Industrial Belt (Angul, Jharsuguda, Paradip, Kalinganagar)",
    "bbox": {
      "west": 83.5,
      "south": 19.8,
      "east": 87.0,
      "north": 22.2
    },
    "description": "Steel plants, aluminum smelters, coal mining complexes, and deep-water ports in Odisha."
  },
  {
    "name": "chhattisgarh_jharkhand",
    "label": "East-Central Mining & Steel Belt (Bhilai, Korba, Dhanbad, Jamshedpur)",
    "bbox": {
      "west": 81.0,
      "south": 21.0,
      "east": 86.8,
      "north": 24.2
    },
    "description": "Coal fields, thermal power plants, and integrated steelworks in CG & JH."
  },
  {
    "name": "all_india",
    "label": "All India Coverage",
    "bbox": {
      "west": 68.0,
      "south": 6.5,
      "east": 97.5,
      "north": 37.5
    },
    "description": "Nationwide active fire and thermal anomaly coverage across the Indian subcontinent."
  }
]
```

---

### Endpoint 3: `GET /facilities`

Returns a GeoJSON `FeatureCollection` of industrial, refinery, power, solar, waste, and mining facilities parsed from the cached OpenStreetMap extract (`data/osm_cache.json`).

#### Sample Response Payload (`GET /facilities`)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "osm_way_1001",
      "geometry": {
        "type": "Point",
        "coordinates": [72.805, 21.25]
      },
      "properties": {
        "facility_id": "osm_way_1001",
        "name": "Hazira Petrochemical Refinery",
        "category": "refinery",
        "osm_type": "way",
        "osm_id": 1001
      }
    }
  ]
}
```

---

### Endpoint 4: `GET /hotspots`

Retrieves active thermal hotspot events for a specified Indian region or custom spatial bounding box. Detections are aggregated by `event_id` so each `event_id` appears **once** in the response, with cluster-level pixel count, FRP total/max, first/last seen timestamps, bounding box, and the highest-severity pixel's model scores and signed SHAP attributions.

#### Query Parameters

| Parameter | Type | Required | Default | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `region` | `string` | Optional | `gujarat` | **Primary Interface**: Named Indian region key (`gujarat`, `maharashtra`, `odisha`, `chhattisgarh_jharkhand`, `all_india`). **Takes precedence over `bbox`**. | `maharashtra` |
| `bbox` | `string` | Optional | Derived from region | Advanced manual override: `west,south,east,north` in degrees. Used only if `region` is omitted. | `72.5,21.0,73.5,22.0` |
| `since_hours` | `integer` | Optional | `24` | Historical lookback window in hours relative to dataset `data_as_of`. | `48` |

#### Response Schema (Array of Aggregated Hotspot Objects)

```typescript
interface ShapDriver {
  feature: string;                     // Canonical feature name
  direction: "+" | "-";                // "+" if pushing toward predicted class, "-" if pushing away
  weight: number;                      // Normalized SHAP attribution magnitude (0.0 - 1.0)
}

interface HotspotResponse {
  event_id: string;                    // Unique cluster ID (e.g. "evt_20260904_0032")
  location_key: string;                // Facility OSM ID or H3 resolution-8 cell ID
  lat: number;                         // Cluster centroid latitude in degrees
  lon: number;                         // Cluster centroid longitude in degrees
  bbox: [number, number, number, number]; // [west, south, east, north] of cluster pixels
  pixel_count: number;                 // Number of satellite detections in this event cluster
  first_seen: string;                  // Earliest detection timestamp in cluster (ISO-8601 UTC)
  last_seen: string;                   // Latest detection timestamp in cluster (ISO-8601 UTC)
  observed: {
    frp: number;                       // Peak Fire Radiative Power in MW
    frp_total: number;                 // Sum of Fire Radiative Power across cluster pixels in MW
    frp_max: number;                   // Maximum pixel Fire Radiative Power in MW
    brightness_ti4: number;            // Maximum I4 band brightness temperature in Kelvin
    confidence: number;                // Detection confidence percentage (0 - 100)
    acq_date: string;                  // Acquisition date in "YYYY-MM-DD"
    acq_time: string;                  // Acquisition time in UTC ("HHMM")
    daynight: "D" | "N";               // Day ("D") or Night ("N") satellite pass
    satellite: string;                 // Satellite identifier (e.g. "SNPP", "N20")
    land_cover: string;                // ESA WorldCover 10m land cover class
    nearest_industrial_type: string;   // OSM facility category ("refinery", "power_plant", "waste_landfill", "quarry_mining", "solar_farm", "general_industrial", "none")
    nearest_industrial_name: string | null; // OSM facility name tag if available
    dist_to_industrial_m: number;      // Exact distance in meters to nearest facility boundary (0 = inside boundary)
  };
  inferred: {
    event_type: string;                // Predicted event class (7 canonical classes)
    event_type_confidence: number;     // Classification probability (0.0 to 1.0)
    z_score: number;                   // Model B signed robust z-score in sigma units
    is_anomalous: boolean;             // Model B boolean anomaly gate (|z| > 3 with >= 10 history samples)
    anomaly_score_normalized: number;  // Robust statistical z-score deviation normalized to [0.0, 1.0]
    hazard_weight: number;             // Multiplier for event type severity (0.0 to 1.0)
    exposure_factor: number;           // Human & infrastructure exposure risk (0.0 to 1.0)
    final_risk_score: number;          // Combined score: hazard_weight * anomaly_score_normalized * exposure_factor (0.0 to 1.0)
    escalating_24h: boolean;           // True if 24h risk is critical (risk_24h >= 0.50)
    risk_24h: number;                  // 24h spread/intensification probability (0.0 to 1.0)
    needs_manual_review: boolean;      // True if low confidence, accident, or candidate illegal unknown source
    top_shap_features: string[];       // Top feature driver names explaining the Model A prediction
    top_shap_factors: ShapDriver[];    // Signed SHAP attributions per driver from TreeExplainer
  };
}
```

#### Sample Response Payload (`GET /hotspots?region=gujarat&since_hours=24`)

```json
[
  {
    "event_id": "evt_20260904_0032",
    "location_key": "osm_way_1001",
    "lat": 21.1700,
    "lon": 72.8300,
    "bbox": [72.828, 21.168, 72.832, 21.172],
    "pixel_count": 2,
    "first_seen": "2026-09-04T01:30:00Z",
    "last_seen": "2026-09-04T13:30:00Z",
    "observed": {
      "frp": 42.3,
      "frp_total": 78.4,
      "frp_max": 42.3,
      "brightness_ti4": 365.2,
      "confidence": 87.0,
      "acq_date": "2026-09-04",
      "acq_time": "1330",
      "daynight": "N",
      "satellite": "SNPP",
      "land_cover": "built_up",
      "nearest_industrial_type": "refinery",
      "nearest_industrial_name": "Hazira Petrochemical Refinery",
      "dist_to_industrial_m": 210.0
    },
    "inferred": {
      "event_type": "industrial_flare",
      "event_type_confidence": 0.91,
      "z_score": 0.72,
      "is_anomalous": false,
      "anomaly_score_normalized": 0.12,
      "hazard_weight": 0.1,
      "exposure_factor": 0.40,
      "final_risk_score": 0.048,
      "escalating_24h": false,
      "risk_24h": 0.08,
      "needs_manual_review": false,
      "top_shap_features": [
        "dist_to_nearest_industrial_m",
        "night_detection_fraction",
        "daynight"
      ],
      "top_shap_factors": [
        {
          "feature": "dist_to_nearest_industrial_m",
          "direction": "+",
          "weight": 0.42
        },
        {
          "feature": "night_detection_fraction",
          "direction": "+",
          "weight": 0.31
        },
        {
          "feature": "daynight",
          "direction": "+",
          "weight": 0.18
        }
      ]
    }
  }
]
```

---

### Endpoint 5: `GET /hotspot/{event_id}`

Fetches the complete diagnostic profile, constituent 375 m VIIRS pixel footprints, full feature vector, and 7-class probability breakdown for a specific thermal event.

#### Path Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `event_id` | `string` | The cluster event ID (e.g. `evt_20260904_0032`). |

#### Response Schema

```typescript
interface HotspotPixel {
  lat: number;
  lon: number;
  frp: number;
  brightness_ti4: number;
  acq_date: string;
  acq_time: string;
}

interface HotspotDetailResponse extends HotspotResponse {
  pixels: HotspotPixel[];
  full_feature_vector: Record<string, any>;
  class_probabilities: {
    industrial_flare: number;
    industrial_accident: number;
    stockpile_combustion: number;
    known_false_positive: number;
    wildfire: number;
    agricultural_burn: number;
    unknown_source: number;
  };
}
```

---

### Endpoint 6: `GET /facility/{location_key}/history`

Returns the historical FRP time series along with robust Median + MAD statistical baseline bands for time-series chart rendering.

#### Path Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `location_key` | `string` | Facility identifier (e.g. `osm_way_1001` or `h3_r8_88609a64d1fffff`). |

#### Response Schema

```typescript
interface FacilityHistoryResponse {
  location_key: string;
  median_frp: number;                    // Facility median FRP baseline in MW
  mad_frp: number;                       // Median Absolute Deviation of FRP
  upper_3sigma_threshold: number;        // Anomaly threshold: median + 3 * max(1.4826*MAD, 0.1*median, 1.0)
  lower_3sigma_threshold: number;        // Lower threshold (clamped to >= 0)
  sample_count: number;                  // Total historical observations logged
  history: Array<{
    date: string;                        // "YYYY-MM-DD"
    time: string;                        // "HHMM" (e.g. "0130" or "1330")
    frp: number;                         // FRP reading in MW
    confidence: number;                  // Detection confidence
    daynight: "D" | "N";                 // Day or Night overpass
  }>;
}
```

---

## 3. Reference: 7 Canonical Event Types

| Event Type String | UI Display Name | Default Hazard Weight | Typical Action / Routing |
| :--- | :--- | :--- | :--- |
| `industrial_flare` | Routine Industrial Flare | `0.1` | Low priority; monitor baseline |
| `industrial_accident` | Industrial Fire Accident | `0.9` | High priority emergency alert |
| `stockpile_combustion` | Stockpile / Waste Combustion | `0.6` | Environmental health alert |
| `known_false_positive` | Solar Farm Thermal Reflection | `0.0` | Suppressed / Informational only |
| `wildfire` | Wildfire | `1.0` | Forest service containment routing |
| `agricultural_burn` | Stubble / Agricultural Burn | `0.3` | Seasonal air quality tracking |
| `unknown_source` | Unregistered / Illegal Thermal Source | `1.0` | Route to NTRO / Analyst manual review |
