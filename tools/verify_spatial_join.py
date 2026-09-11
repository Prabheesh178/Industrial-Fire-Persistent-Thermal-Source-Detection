"""Prove the sjoin_nearest rewrite matches the original per-point scan.

Compares against the ORIGINAL implementation held at the same CRS, so the
algorithm change is isolated from the EPSG:3857 -> EPSG:7755 change.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))   # data/ paths are repo-root relative

from dotenv import load_dotenv; load_dotenv(".env")
import json, time, importlib.util
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from config.settings import OSM_CACHE_PATH
from processing.spatial_join import parse_osm_geometries, compute_boundary_distances

ORIG = os.environ.get("SPATIAL_JOIN_BASELINE", "")
if not ORIG or not os.path.exists(ORIG):
    sys.exit("Set SPATIAL_JOIN_BASELINE to a copy of the pre-sjoin_nearest spatial_join.py\n"
             "  git show <commit-before-the-fix>:processing/spatial_join.py > /tmp/sj_orig.py\n"
             "  SPATIAL_JOIN_BASELINE=/tmp/sj_orig.py python tools/verify_spatial_join.py")
spec = importlib.util.spec_from_file_location("orig", ORIG)
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)

osm = json.load(open(OSM_CACHE_PATH))
ind, pop, infra = parse_osm_geometries(osm)
print(f"  OSM: {len(ind):,} industrial | {len(pop):,} populated | {len(infra):,} infra")

df = pd.read_csv("data/firms_corpus_balanced.csv").sample(150, random_state=7).reset_index(drop=True)

t = time.perf_counter(); new = compute_boundary_distances(df, ind, pop, infra); t_new = time.perf_counter() - t
t = time.perf_counter(); old = orig.compute_boundary_distances(df, ind, pop, infra); t_old = time.perf_counter() - t
print(f"  150 pts: new {t_new:.2f}s | original {t_old:.2f}s | speedup {t_old/max(t_new,1e-9):.0f}x")
print(f"  projected to 9,206 rows: new {t_new/150*9206:.0f}s | original {t_old/150*9206/60:.0f} min")

# same nearest facility chosen?
agree = (new["nearest_industrial_type"].values == old["nearest_industrial_type"].values).mean()
print(f"  nearest_industrial_type agreement: {agree*100:.1f}%")

# distances: ratio should be the Mercator factor 1/cos(lat), not arbitrary
ratio = old["dist_to_nearest_industrial_m"].values / np.maximum(new["dist_to_nearest_industrial_m"].values, 1e-6)
expect = 1.0 / np.cos(np.radians(df["latitude"].values))
ok = np.isfinite(ratio) & (new["dist_to_nearest_industrial_m"].values > 1.0)
print(f"  old/new distance ratio  median {np.median(ratio[ok]):.3f}")
print(f"  expected 1/cos(lat)     median {np.median(expect[ok]):.3f}")
print(f"  max |ratio - 1/cos(lat)| = {np.abs(ratio[ok]-expect[ok]).max():.4f}")
