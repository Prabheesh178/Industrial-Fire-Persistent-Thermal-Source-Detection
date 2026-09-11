"""Train Models A/B/C on the multi-region, multi-season corpus.

train_all.py refetches from FIRMS on every invocation and only expresses
"N days back from today", so it cannot train on a corpus that spans the
Oct-Nov 2025 stubble season and the Mar 2026 fire season at once.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))   # data/ paths are repo-root relative

from dotenv import load_dotenv; load_dotenv(".env")
import json, sys, time
import pandas as pd
from config.settings import ARTIFACTS_DIR, DB_PATH, OSM_CACHE_PATH, EVENT_TYPES
from processing.spatial_join import perform_spatial_join
from processing.clustering import cluster_hotspots_spatiotemporal
from processing.persistence_log import update_persistence_log
from processing.feature_engineering import build_feature_table
from processing.labeling_heuristic import apply_labeling_heuristic
from models.model_a_classifier import ModelAClassifier
from models.model_b_anomaly_engine import ModelBAnomalyEngine
from models.model_c_risk_model import ModelCRiskModel

t0 = time.time()
def step(msg): print(f"\n>>> {msg}  [{time.time()-t0:.0f}s]", flush=True)

# Refuse to train on the synthetic OSM fallback -- it is 6 fabricated polygons and
# dist_to_nearest_industrial_m drives 4 of the 7 label rules.
with open(OSM_CACHE_PATH) as fh:
    osm = json.load(fh)
n_osm = len(osm.get("elements", []))
if n_osm < 100:
    sys.exit(f"ABORT: OSM cache has only {n_osm} elements -- that is the synthetic "
             f"fallback, not real data. Fix ingestion before training.")
print(f"OSM cache: {n_osm:,} real elements")

df = pd.read_csv("data/firms_corpus_balanced.csv")
print(f"Corpus: {len(df):,} rows")

step("1/6 spatial join (boundary distances vs real OSM polygons)")
df = perform_spatial_join(df, str(OSM_CACHE_PATH))
print("  dist_to_nearest_industrial_m:",
      df["dist_to_nearest_industrial_m"].describe()[["min","50%","max"]].round(0).to_dict())
print("  nearest_industrial_type:", df["nearest_industrial_type"].value_counts().head(8).to_dict())

step("2/6 space-time DBSCAN clustering")
df = cluster_hotspots_spatiotemporal(df)
print(f"  events: {df['event_id'].nunique():,} from {len(df):,} detections")

step("3/6 persistence log")
df = update_persistence_log(df, str(DB_PATH))
print("  days_active_last_30:", df["days_active_last_30"].describe()[["min","50%","max"]].to_dict())

step("4/6 Model B baselines + feature table")
engine = ModelBAnomalyEngine(ARTIFACTS_DIR, DB_PATH)
engine.fit_from_db()
df = build_feature_table(df, anomaly_engine=engine)
print(f"  feature table: {df.shape}")

step("5/6 heuristic labels")
df = apply_labeling_heuristic(df)
df.to_csv("data/labeled_dataset.csv", index=False)
counts = df["event_type"].value_counts()
print("\n  CLASS DISTRIBUTION (the number that decides if any of this means anything):")
for c in EVENT_TYPES:
    n = int(counts.get(c, 0))
    flag = "  <-- TOO FEW TO LEARN" if n < 30 else ""
    print(f"    {c:24s} {n:6,}{flag}")
missing = [c for c in EVENT_TYPES if counts.get(c, 0) < 30]

step("6/6 train A and C")
a = ModelAClassifier(ARTIFACTS_DIR); ma = a.train(df)
print("  Model A:", {k: round(v, 4) for k, v in ma.items() if isinstance(v, (int, float))})
c = ModelCRiskModel(ARTIFACTS_DIR); mc = c.train(df)
print("  Model C:", {k: round(v, 4) for k, v in mc.items() if isinstance(v, (int, float))})

print(f"\nDONE in {time.time()-t0:.0f}s")
if missing:
    print(f"CAVEAT: {len(missing)} class(es) under 30 examples: {missing}")
    print("The headline accuracy does NOT describe these classes.")
