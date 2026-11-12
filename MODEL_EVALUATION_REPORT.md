# Model Training Verification & Accuracy Evaluation Report

**Generated:** 2026-11-08  
**Region Scope:** `all_india` (`68.1°E–97.4°E, 6.7°N–35.5°N`)  
**Lookback Window:** 30 days (`2026-10-09` to `2026-11-02`)  
**Corpus Size:** 8,500 satellite thermal detections aggregated into **7,203 unique spatiotemporal events**

---

## 1. Training Completion & Artifact Verification

All **three models** (**Model A**, **Model B**, and **Model C**) and all supporting spatial/persistence artifacts have been trained and saved to disk:

| Component | Artifact Path | Status | Size | Last Modified (IST) | Description |
| :--- | :--- | :--- | ---: | :--- | :--- |
| **Model A** | `artifacts/model_a.joblib` | Trained & Verified | 1.56 MB (`1,561,297 B`) | `2026-11-07 21:12:13` | 7-class LightGBM classifier + `LabelEncoder` + `shap.TreeExplainer` |
| **Model B** | `artifacts/facility_baselines.json` | Trained & Verified | 1.06 MB (`1,056,593 B`) | `2026-11-07 20:17:39` | Robust Median/MAD FRP baselines for 7,098 facilities + 7 regional category fallbacks |
| **Model C** | `artifacts/model_c.joblib` | Trained & Verified | 246 KB (`245,922 B`) | `2026-11-07 21:12:14` | Binary LightGBM 24h escalation classifier (`duty_cycle < 0.2`) |
| **Feature Table** | `data/feature_table.csv` | Generated | 2.38 MB (`2,376,521 B`) | `2026-11-07 21:11:32` | 8,500 rows × 29 engineered radiometric, spatial, weather, and persistence columns |
| **Labeled Dataset** | `data/labeled_dataset.csv` | Generated | 2.51 MB (`2,510,668 B`) | `2026-11-07 21:12:11` | Weak-supervision ground-truth labels across all 7 canonical classes |
| **Persistence DB** | `data/hotspots.db` | Generated | 1.78 MB (`1,781,760 B`) | `2026-11-07 19:53:40` | SQLite H3 persistence history (`days_active_last_30`, `night_detection_fraction`) |
| **OSM Cache** | `data/osm_cache.json` | Cached | 63.75 MB (`63,755,155 B`) | `2026-11-07 19:49:10` | All-India OpenStreetMap industrial, power, mining, solar, waste, and infrastructure geometries |
| **Land Cover Cache** | `data/landcover_cache.json` | Cached | 233 KB (`232,640 B`) | `2026-11-07 20:49:29` | ESA WorldCover 10m Cloud-Optimized GeoTIFF lookups |
| **Weather Cache** | `data/weather_cache.json` | Cached | 1.13 MB (`1,134,702 B`) | `2026-11-07 21:11:27` | Open-Meteo ERA5 historical temperature, relative humidity, and wind speed lookups |

---

## 2. Model A — Multi-Class Event Classifier (LightGBM + SHAP)

### 2.1 Architecture & Features
- **Estimator:** `lightgbm.LGBMClassifier(objective="multiclass", num_class=7, num_leaves=31, learning_rate=0.05, n_estimators=300)` with early stopping (`stopping_rounds=25`) on validation `multi_logloss`.
- **Input Features (20):** `frp`, `brightness_ti4`, `brightness_ti5`, `confidence`, `daynight`, `dist_to_nearest_industrial_m`, `nearest_industrial_type`, `land_cover_class`, `temperature`, `humidity`, `wind_speed`, `days_active_last_30`, `night_detection_fraction`, `month`, `is_agri_burn_season`, `dist_to_populated_area_m`, `dist_to_critical_infra_m`, `cluster_growth_rate`, `delta_bt`, `frp_density`.
- **Leakage Guard:** Strictly excludes `frp_zscore_vs_facility_baseline` (which is produced by Model B).

### 2.2 Overall Accuracy Summary

| Evaluation Split | Sample Size ($N$) | Overall Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Stratified 80/20 Holdout Validation** (`random_state=42`) | 1,700 | **99.12%** | 0.9307 | 0.8813 | 0.8762 | **0.9915** |
| **Ordered/Chronological First-20% Validation** | 1,698 | **95.70%** | 0.5334 | 0.6853 | 0.5662 | **0.9621** |
| **Full Corpus Evaluation** | 8,500 | **99.14%** | 0.9289 | 0.9459 | 0.9299 | **0.9919** |

### 2.3 Per-Class Performance (Stratified 20% Holdout Validation, $N = 1,700$)

| Canonical Event Type | Precision | Recall | F1-Score | Validation Support | Full Corpus Support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `agricultural_burn` | 0.9989 | 0.9989 | **0.9989** | 950 | 4,750 |
| `unknown_source` | 1.0000 | 1.0000 | **1.0000** | 418 | 2,090 |
| `wildfire` | 1.0000 | 0.9618 | **0.9805** | 288 | 1,440 |
| `industrial_flare` | 1.0000 | 0.8750 | **0.9333** | 8 | 38 |
| `stockpile_combustion` | 0.8333 | 1.0000 | **0.9091** | 5 | 27 |
| `industrial_accident` | 0.6829 | 1.0000 | **0.8116** | 28 | 142 |
| `known_false_positive` | 1.0000 | 0.3333 | **0.5000** | 3 | 13 |
| **Macro Average** | **0.9307** | **0.8813** | **0.8762** | **1,700** | **8,500** |
| **Weighted Average** | **0.9937** | **0.9912** | **0.9915** | **1,700** | **8,500** |

### 2.4 Per-Class Performance (Full Corpus, $N = 8,500$)

| Canonical Event Type | Precision | Recall | F1-Score | Support |
| :--- | ---: | ---: | ---: | ---: |
| `agricultural_burn` | 0.9983 | 0.9983 | **0.9983** | 4,750 |
| `industrial_accident` | 0.7100 | 1.0000 | **0.8304** | 142 |
| `industrial_flare` | 1.0000 | 0.8158 | **0.8986** | 38 |
| `known_false_positive` | 1.0000 | 0.8462 | **0.9167** | 13 |
| `stockpile_combustion` | 0.7941 | 1.0000 | **0.8852** | 27 |
| `unknown_source` | 1.0000 | 1.0000 | **1.0000** | 2,090 |
| `wildfire` | 1.0000 | 0.9611 | **0.9802** | 1,440 |
| **Macro Average** | **0.9289** | **0.9459** | **0.9299** | **8,500** |
| **Weighted Average** | **0.9936** | **0.9914** | **0.9919** | **8,500** |

### 2.5 Full Corpus Confusion Matrix ($N = 8,500$)
Rows represent ground-truth weak-supervision labels; columns represent Model A predictions:

| True \ Predicted | `agri_burn` | `ind_accident` | `ind_flare` | `false_pos` | `stockpile` | `unknown` | `wildfire` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`agricultural_burn`** | **4,742** | 7 | 0 | 0 | 1 | 0 | 0 |
| **`industrial_accident`** | 0 | **142** | 0 | 0 | 0 | 0 | 0 |
| **`industrial_flare`** | 0 | 5 | **31** | 0 | 2 | 0 | 0 |
| **`known_false_positive`** | 0 | 2 | 0 | **11** | 0 | 0 | 0 |
| **`stockpile_combustion`** | 0 | 0 | 0 | 0 | **27** | 0 | 0 |
| **`unknown_source`** | 0 | 0 | 0 | 0 | 0 | **2,090** | 0 |
| **`wildfire`** | 8 | 44 | 0 | 0 | 4 | 0 | **1,384** |

> **Triage Safety Observation:** Recall on `industrial_accident` is **100.0%** (`142 / 142`) and recall on `stockpile_combustion` is **100.0%** (`27 / 27`). The few misclassifications occur when borderline wildfires or flares near industrial perimeters are conservatively routed into `industrial_accident`, which triggers manual analyst review rather than missing a hazard.

### 2.6 Top 10 LightGBM Feature Importances (Model A Split Counts)

| Rank | Feature Name | Split Importance | Physical Role |
| ---: | :--- | ---: | :--- |
| 1 | `dist_to_nearest_industrial_m` | 2,378 | Separates inside-perimeter industrial activity from rural/forest fires |
| 2 | `dist_to_critical_infra_m` | 2,312 | Identifies proximity to pipelines, substations, and rail corridors |
| 3 | `temperature` | 1,949 | Ambient ERA5 surface air temperature context |
| 4 | `brightness_ti5` | 897 | VIIRS I5 long-wave thermal channel (background temperature contrast) |
| 5 | `dist_to_populated_area_m` | 889 |Separates peri-urban/industrial zones from remote forest interiors |
| 6 | `night_detection_fraction` | 680 | Distinguishes 24/7 industrial flaring/combustors from daytime burns |
| 7 | `days_active_last_30` | 609 | Captures 30-day persistence at the same H3 resolution-8 cell |
| 8 | `humidity` | 590 | Relative humidity control on wildfire vs. industrial combustion |
| 9 | `frp_density` | 575 | Radiative power concentration per cluster pixel (`MW / pixel`) |
| 10 | `wind_speed` | 534 | Surface wind speed driving spread and plume dispersion |

---

## 3. Model B — Statistical Facility Baseline & Robust MAD Anomaly Engine

### 3.1 Baseline Coverage
- **Individual Facility / Location Baselines Fitted:** **7,098** unique H3/facility keys (`artifacts/facility_baselines.json`)
- **Regional Industrial Category Fallbacks Fitted:** **7** (`general_industrial`, `power_plant`, `quarry_mining`, `refinery`, `solar_farm`, `waste_landfill`, `none`)
- **Mathematical Formulation:**
  $$z_{\text{robust}} = \frac{\text{FRP} - \text{median}(\text{FRP})}{1.4826 \times \text{MAD}(\text{FRP})}$$
  Locations with $< 10$ historical observations automatically blend toward the regional category baseline via shrinkage factor $\alpha = n / 10$.

### 3.2 Z-Score Distribution & Anomaly Detection Rate ($N = 8,500$)
- **Flagged Anomalies ($|z| > 3\sigma$ with sufficient history):** **208 detections** (**2.45%** of corpus)
- **Z-Score Quantiles Across Corpus:**
  - `p10`: `-0.896`
  - `p25`: `-0.605`
  - `p50` (Median): `-0.108`
  - `p75`: `+0.690`
  - `p90`: `+1.877`
  - `p95`: `+2.956` (right at the $3\sigma$ anomaly boundary)
  - `p99`: `+6.388`

---

## 4. Model C — 24h Risk Escalation Model (LightGBM Binary)

### 4.1 Scope & Leakage Prevention
- **Duty-Cycle Filter (`duty_cycle < 0.2`):** Filtered out 591 steady industrial flare detections, retaining **7,909 non-routine detections** (`6,328` train / `1,581` validation).
- **Positive Escalation Base Rate:** **471 / 7,909** (**5.96%**).
- **Feature Guard:** Explicitly drops `days_active_last_30` from `MODEL_C_FEATURES` (18 features retained).

### 4.2 Validation & Full-Dataset Metrics

| Metric | Held-Out Validation ($N = 1,581$) | Full Non-Routine Corpus ($N = 7,909$) |
| :--- | ---: | ---: |
| **Accuracy** | **99.81%** | **99.96%** |
| **ROC-AUC** | **1.0000** | **1.0000** |
| **Precision (Escalating = 1)** | **98.92%** | **99.58%** |
| **Recall (Escalating = 1)** | **97.87%** | **99.79%** |
| **F1-Score (Escalating = 1)** | **0.9840** | **0.9968** |

**Validation Confusion Matrix ($N = 1,581$):**
- True Negative (Non-escalating correctly predicted): **1,486**
- False Positive (Non-escalating predicted as escalating): **1**
- False Negative (Escalating missed): **2**
- True Positive (Escalating correctly predicted): **92**

### 4.3 Top Feature Importances (Model C Split Counts)
1. `cluster_growth_rate` (372 splits)
2. `frp` (316 splits)
3. `frp_zscore_vs_facility_baseline` (223 splits)
4. `brightness_ti5` (213 splits)
5. `brightness_ti4` (181 splits)
6. `humidity` (172 splits)
7. `dist_to_nearest_industrial_m` (159 splits)
8. `wind_speed` (141 splits)
9. `temperature` (106 splits)
10. `dist_to_critical_infra_m` (74 splits)

> **Methodological Caveat (Documented in `models/model_c_risk_model.py`):** As noted in `ModelCRiskModel.derive_target_labels()`, the current escalation label is derived from a present-tense physical heuristic (`cluster_growth_rate > 0.5` OR `(z_score > 2.5 AND wind_speed > 15)` OR `(frp > 60 AND cluster_growth_rate > 0)`). Because those four variables are also input features to Model C, LightGBM recovers the decision boundary with `ROC-AUC = 1.0000` (or `0.8398` when those 4 columns are ablated). Consequently, `risk_24h` and `escalating_24h` are kept strictly separate from `final_risk_score` in `ScoreCombiner` and used only as a supplementary triage indicator.

---

## 5. ScoreCombiner — Final Risk Score Synthesis

`ScoreCombiner` synthesizes the outputs of Model A and Model B with spatial exposure using the deterministic formula:
$$\text{final\_risk\_score} = \text{hazard\_weight}[\text{event\_type}] \times \text{anomaly\_score\_normalized} \times \text{exposure\_factor}$$

- Neither Model A confidence (`event_type_confidence`) nor Model C escalation probability (`risk_24h`) is multiplied into `final_risk_score`.
- Across the **7,203 unique clustered events** in the `all_india` snapshot:
  - Events with `needs_manual_review = True` are automatically surfaced whenever `event_type_confidence < 0.60`, `event_type` is `industrial_accident` or `unknown_source`, or `hazard_weight >= 0.9` with `final_risk_score >= 0.30`.

