"""
Pydantic Response Schemas Module
================================

HOW TO RUN:
    This module defines the response data models used by the FastAPI server in api/main.py.
    To validate schema definitions:
        python -m api.schemas

INPUTS:
    - Raw dictionary payloads from pipeline and inference engines.

OUTPUT:
    - Strictly typed Pydantic models enforcing the exact observed vs inferred separation required by the spec.

STRICT RULES OBSERVED:
    - Response structure separates `observed` (satellite evidence: FRP, brightness_ti4, confidence,
      acq_date, acq_time, daynight, satellite, land_cover, industrial type/name/dist)
      from `inferred` (model output: event_type, z_score, is_anomalous, anomaly_norm, hazard_weight,
      exposure, final_risk_score, escalating_24h, risk_24h, needs_manual_review, top_shap_features,
      top_shap_factors).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ShapDriver(BaseModel):
    """
    Signed SHAP attribution for a single feature from TreeExplainer.
    """
    feature: str = Field(..., description="Canonical feature name")
    direction: str = Field(..., description="'+' if pushing toward predicted class, '-' if pushing away")
    weight: float = Field(..., description="Normalized magnitude of SHAP attribution (0.0 - 1.0)")


class HotspotPixel(BaseModel):
    """
    Individual 375m VIIRS detection pixel belonging to an aggregated event cluster.
    """
    lat: float
    lon: float
    frp: float
    brightness_ti4: float
    acq_date: str
    acq_time: str


class ObservedData(BaseModel):
    """
    Direct satellite and geospatial observations (pure evidence, zero model inference).
    """
    frp: float = Field(..., description="Peak Fire Radiative Power in MW from FIRMS sensor")
    frp_total: float = Field(..., description="Sum of Fire Radiative Power across cluster detections in MW")
    frp_max: float = Field(..., description="Maximum pixel Fire Radiative Power in MW")
    brightness_ti4: float = Field(315.0, description="Maximum I4 band brightness temperature in Kelvin")
    confidence: float = Field(..., description="Detection confidence percentage (0-100)")
    acq_date: str = Field(..., description="Acquisition date (YYYY-MM-DD)")
    acq_time: str = Field("0000", description="Acquisition time in UTC (HHMM)")
    daynight: str = Field("D", description="Day ('D') or Night ('N') satellite pass")
    satellite: str = Field("SNPP", description="Satellite identifier (e.g., SNPP, N20)")
    land_cover: str = Field("unknown", description="ESA WorldCover 10m land cover class")
    nearest_industrial_type: str = Field(..., description="OSM facility type (refinery, power_plant, etc.)")
    nearest_industrial_name: Optional[str] = Field(None, description="OSM facility name tag if available")
    dist_to_industrial_m: float = Field(..., description="Exact distance to nearest facility boundary in meters")


class InferredData(BaseModel):
    """
    Synthesized machine learning and statistical inferences.
    """
    event_type: str = Field(..., description="Predicted class from Model A")
    event_type_confidence: float = Field(..., description="Classification probability (0.0 - 1.0)")
    z_score: float = Field(0.0, description="Model B signed robust z-score in sigma units")
    is_anomalous: bool = Field(False, description="Model B boolean anomaly gate (|z| > 3 with >=10 history samples)")
    anomaly_score_normalized: float = Field(..., description="Model B normalized robust z-score deviation (0.0 - 1.0)")
    hazard_weight: float = Field(..., description="Deterministic hazard weight multiplier for this class")
    exposure_factor: float = Field(..., description="Spatial exposure factor derived from populated area & infra distance")
    final_risk_score: float = Field(..., description="Combined risk score: hazard_weight * anomaly_score * exposure_factor")
    escalating_24h: bool = Field(..., description="Model C boolean flag indicating spread/intensification risk")
    risk_24h: float = Field(..., description="Model C escalation probability (0.0 - 1.0)")
    needs_manual_review: bool = Field(..., description="Routing flag for analyst review queue")
    top_shap_features: List[str] = Field(..., description="Top feature driver names from SHAP TreeExplainer")
    top_shap_factors: List[ShapDriver] = Field(
        default_factory=list,
        description="Signed SHAP attributions per driver from TreeExplainer (omitted if unavailable)",
    )


class HotspotResponse(BaseModel):
    """
    Canonical response schema for each aggregated thermal event in /hotspots.
    One event_id corresponds to one unique cluster record.
    """
    event_id: str
    location_key: str
    lat: float
    lon: float
    bbox: List[float] = Field(..., description="[west, south, east, north] bounding box of cluster pixels")
    pixel_count: int = Field(1, description="Number of satellite detection pixels in this event cluster")
    first_seen: str = Field(..., description="Earliest detection timestamp in cluster (ISO-8601 UTC)")
    last_seen: str = Field(..., description="Latest detection timestamp in cluster (ISO-8601 UTC)")
    observed: ObservedData
    inferred: InferredData


class HotspotDetailResponse(HotspotResponse):
    """
    Full diagnostic feature vector, SHAP explanation breakdown, constituent pixels,
    and class probability distribution for /hotspot/{event_id}.
    """
    pixels: List[HotspotPixel] = Field(default_factory=list)
    full_feature_vector: Dict[str, Any]
    class_probabilities: Dict[str, float]


class FacilityHistoryPoint(BaseModel):
    """
    Single historical time-series observation at a facility.
    """
    date: str
    time: str
    frp: float
    confidence: float
    daynight: str


class FacilityHistoryResponse(BaseModel):
    """
    Time series of FRP readings overlaid with MAD baseline thresholds for /facility/{location_key}/history.
    """
    location_key: str
    median_frp: float
    mad_frp: float
    upper_3sigma_threshold: float
    lower_3sigma_threshold: float
    sample_count: int
    history: List[FacilityHistoryPoint]


class RegionBBox(BaseModel):
    """
    Bounding box coordinates in degrees.
    """
    west: float = Field(..., description="Western longitude boundary")
    south: float = Field(..., description="Southern latitude boundary")
    east: float = Field(..., description="Eastern longitude boundary")
    north: float = Field(..., description="Northern latitude boundary")


class RegionInfo(BaseModel):
    """
    Predefined Indian region metadata for frontend selection dropdowns.
    """
    name: str = Field(..., description="Region key/identifier (e.g. 'gujarat', 'maharashtra')")
    label: str = Field(..., description="Human-readable title and coverage cities")
    bbox: RegionBBox = Field(..., description="Bounding box definition")
    description: Optional[str] = Field("", description="Regional industrial profile description")


class ArtifactInfo(BaseModel):
    """
    File metadata for a trained model or baseline artifact on disk.
    """
    filename: str
    exists: bool
    loaded: bool
    size_bytes: Optional[int] = None
    modified_at: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Service health, data currency timestamp, and artifact status from GET /health.
    """
    model_config = {"protected_namespaces": ()}

    status: str = Field(..., description="'ok' when all model artifacts are loaded, 'error' otherwise")
    error: Optional[str] = Field(None, description="Clear error description if any required artifacts are missing")
    missing_artifacts: List[str] = Field(default_factory=list)
    data_as_of: Optional[str] = Field(None, description="Max acquisition timestamp in dataset (ISO-8601 UTC)")
    active_events_count: int = Field(0, description="Number of distinct event_id clusters currently available")
    model_artifacts: Dict[str, ArtifactInfo]


if __name__ == "__main__":
    sample = HotspotResponse(
        event_id="evt_20260904_0032",
        location_key="osm_way_1001",
        lat=21.17,
        lon=72.83,
        bbox=[72.828, 21.168, 72.832, 21.172],
        pixel_count=2,
        first_seen="2026-09-04T01:30:00Z",
        last_seen="2026-09-04T13:30:00Z",
        observed=ObservedData(
            frp=42.3,
            frp_total=78.4,
            frp_max=42.3,
            brightness_ti4=365.2,
            confidence=87.0,
            acq_date="2026-09-04",
            acq_time="1330",
            daynight="N",
            satellite="SNPP",
            land_cover="built_up",
            nearest_industrial_type="refinery",
            nearest_industrial_name="Hazira Petrochemical Refinery",
            dist_to_industrial_m=210.0,
        ),
        inferred=InferredData(
            event_type="industrial_flare",
            event_type_confidence=0.91,
            z_score=0.72,
            is_anomalous=False,
            anomaly_score_normalized=0.12,
            hazard_weight=0.1,
            exposure_factor=0.4,
            final_risk_score=0.048,
            escalating_24h=False,
            risk_24h=0.08,
            needs_manual_review=False,
            top_shap_features=["dist_to_nearest_industrial_m", "night_detection_fraction", "daynight"],
            top_shap_factors=[
                ShapDriver(feature="dist_to_nearest_industrial_m", direction="+", weight=0.42),
                ShapDriver(feature="night_detection_fraction", direction="+", weight=0.31),
                ShapDriver(feature="daynight", direction="+", weight=0.18),
            ],
        ),
    )
    print("Schema Validation Test Successful:")
    print(sample.model_dump_json(indent=2))
