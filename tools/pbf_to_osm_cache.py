"""Build the OSM cache from Geofabrik PBF extracts instead of Overpass.

Overpass is not usable from this IP: overpass-api.de (and its lz4/z mirrors) TCP-refuse
us after an earlier burst of oversized queries, and kumi.systems 429s on region-sized
queries. Geofabrik extracts are static files, so this path is deterministic and
rate-limit free. Output matches the Overpass `out geom;` shape that
processing/spatial_join.parse_osm_geometries expects.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))   # data/ paths are repo-root relative

from dotenv import load_dotenv; load_dotenv(".env")
import json, sys, pathlib
from pyrosm import OSM
from config.settings import REGIONS, OSM_CACHE_PATH

ZONES = {
    "northern-zone": ["punjab_haryana", "rajasthan_solar"],
    "western-zone":  ["gujarat"],
    "eastern-zone":  ["odisha"],
    "central-zone":  ["chhattisgarh_jharkhand"],
}
KEEP = {
    "man_made":  ["works", "petroleum_well", "chimney"],
    "landuse":   ["industrial", "quarry", "landfill", "residential"],
    "power":     ["plant", "generator", "substation"],
    "amenity":   ["waste_disposal"],
    "plant:source": True,
    "place":     ["city", "town", "village"],
}

def coords(geom):
    """Overpass-style [{lat, lon}] ring from a shapely geometry."""
    g = geom.geoms[0] if geom.geom_type.startswith("Multi") else geom
    if g.geom_type == "Polygon":
        return [{"lat": y, "lon": x} for x, y in g.exterior.coords]
    if g.geom_type in ("LineString", "LinearRing"):
        return [{"lat": y, "lon": x} for x, y in g.coords]
    return None

elements, seen = [], set()
for zone, regs in ZONES.items():
    path = pathlib.Path("data/pbf") / f"{zone}.osm.pbf"
    if not path.exists():
        print(f"  {zone}: MISSING {path}", flush=True); continue
    for reg in regs:
        b = REGIONS[reg]["bbox"]
        osm = OSM(str(path), bounding_box=[b["west"], b["south"], b["east"], b["north"]])
        try:
            gdf = osm.get_data_by_custom_criteria(
                custom_filter=KEEP, filter_type="keep",
                keep_nodes=True, keep_ways=True, keep_relations=True)
        except Exception as exc:
            print(f"  {zone}/{reg}: extract failed {type(exc).__name__}: {exc}", flush=True); continue
        if gdf is None or gdf.empty:
            print(f"  {zone}/{reg}: 0 features", flush=True); continue

        n0 = len(elements)
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            oid = int(row.get("id", 0) or 0)
            tags = {k: v for k, v in row.items()
                    if k not in ("geometry", "id", "timestamp", "version", "tags", "osm_type")
                    and v is not None and str(v) != "nan"}
            extra = row.get("tags")
            if isinstance(extra, dict):
                tags.update({k: v for k, v in extra.items() if v is not None})
            elif isinstance(extra, str) and extra.startswith("{"):
                try: tags.update(json.loads(extra))
                except Exception: pass

            if geom.geom_type == "Point":
                el = {"type": "node", "id": oid, "lat": geom.y, "lon": geom.x, "tags": tags}
            else:
                ring = coords(geom)
                if not ring: continue
                el = {"type": "way", "id": oid, "geometry": ring, "tags": tags}
            key = (el["type"], oid, round(el.get("lat", ring[0]["lat"] if el["type"]=="way" else 0), 5))
            if key in seen: continue
            seen.add(key); elements.append(el)
        print(f"  {zone:15s} {reg:24s} -> +{len(elements)-n0:6,} (total {len(elements):,})", flush=True)

with open(OSM_CACHE_PATH, "w") as fh:
    json.dump({"elements": elements}, fh)
print(f"\nOSM_CACHE_BUILT {len(elements):,} real elements -> {OSM_CACHE_PATH}")
if not elements:
    print("FATAL: zero elements"); sys.exit(1)
