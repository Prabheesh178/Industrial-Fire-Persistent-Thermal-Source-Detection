"""
Build Slim Facilities File for Lightweight Runtime Serving
=========================================================

Reads data/osm_cache.json (or fetches minimal geometries) and generates
data/facilities_slim.json (~5 MB) containing:
1. id -> name lookup map
2. GeoJSON FeatureCollection of facility points with category and facility_id
"""

import os
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import OSM_CACHE_PATH, DATA_DIR

SLIM_OUTPUT_PATH = Path(DATA_DIR) / "facilities_slim.json"


def build_slim_facilities(source_path: Path = Path(OSM_CACHE_PATH), dest_path: Path = SLIM_OUTPUT_PATH):
    print(f"[Slim Facilities] Reading OSM extract from {source_path}...")
    if not source_path.exists():
        print(f"[Slim Facilities] Warning: {source_path} does not exist.")
        return False

    with open(source_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)

    from api.main import _classify_osm_element, _extract_element_coords

    names = {}
    features = []

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
        features.append({
            "type": "Feature",
            "id": fac_id,
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 4), round(lat, 4)],
            },
            "properties": {
                "facility_id": fac_id,
                "name": name or f"{category.replace('_', ' ').title()} ({fac_id})",
                "category": category,
            },
        })

    payload = {
        "names": names,
        "facilities": {
            "type": "FeatureCollection",
            "features": features,
        },
    }

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
    print(f"[Slim Facilities] Successfully generated {dest_path} ({size_mb} MB, {len(names)} names, {len(features)} facility points).")
    return True


if __name__ == "__main__":
    build_slim_facilities()
