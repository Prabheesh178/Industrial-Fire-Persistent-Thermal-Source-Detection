"""Watch three real detections move through the pipeline, then through training.

Runs the real production functions once over the full corpus (clustering and
persistence need the neighbours), snapshots the frame after every stage, then prints
three tracked rows at each stage, trains Model A, and asks it about them.

    python tools/trace_training.py               # picks three illustrative rows itself
    python tools/trace_training.py picks.json    # or supply {"name": row_id, ...}

Input : data/firms_corpus_balanced.csv  (python tools/build_corpus.py)
Writes: data/hotspots.db, artifacts/model_a.joblib, artifacts/facility_baselines.json
        -- the same artifacts tools/run_training.py produces.
"""
import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))   # data/ paths are repo-root relative
from dotenv import load_dotenv; load_dotenv(".env")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from config.settings import ARTIFACTS_DIR, OSM_CACHE_PATH
from processing.spatial_join import perform_spatial_join
from processing.clustering import cluster_hotspots_spatiotemporal
from processing.persistence_log import update_persistence_log
from processing.feature_engineering import build_feature_table
from processing.labeling_heuristic import apply_labeling_heuristic
from models.model_a_classifier import ModelAClassifier
from models.model_b_anomaly_engine import ModelBAnomalyEngine

CORPUS, DB = "data/firms_corpus_balanced.csv", "data/hotspots.db"
W = 22


def auto_pick(d: pd.DataFrame) -> dict:
    """One crop fire, one spreading forest fire, one detection inside an industrial site."""
    d = d.sort_values("_id")
    pull = d["pull"] if "pull" in d.columns else pd.Series("", index=d.index)
    picks = {}
    q = d[(pull == "agri-punjab-2025") & (d.event_type == "agricultural_burn")
          & (d.land_cover_class == "cropland")]
    if len(q): picks["crop fire"] = int(q["_id"].iloc[len(q) // 2])
    q = d[(pull == "wildfire-odisha") & (d.event_type == "wildfire")
          & (d.land_cover_class == "forest") & (d.cluster_growth_rate > 0)]
    if not len(q): q = d[d.event_type == "wildfire"]
    if len(q): picks["forest fire"] = int(q["_id"].iloc[0])
    q = d[(d.event_type == "industrial_accident") & (d.dist_to_nearest_industrial_m < 300)]
    if not len(q): q = d[d.event_type == "industrial_accident"]
    if len(q): picks["industrial"] = int(q["_id"].iloc[0])
    if not picks:
        sys.exit("Could not find illustrative rows in this corpus; pass a picks.json.")
    return picks


def why(r) -> str:
    """Which branch of classify_point_heuristic produced this row's label."""
    d, t, lc = r.dist_to_nearest_industrial_m, r.nearest_industrial_type, str(r.land_cover_class)
    duty = r.get("duty_cycle", r.days_active_last_30 / 30); z = r.frp_zscore_vs_facility_baseline
    if d < 500 and t == "solar_farm" and r.daynight == "D" and r.cluster_growth_rate <= 0:
        return "solar panel, daytime, not spreading"
    if d < 500 and duty > 0.6: return f"{d:.0f} m from site AND burns {duty:.0%} of nights"
    if d < 500 and (z > 3 or r.frp > 50) and duty < 0.2 and r.days_active_last_30 <= 1:
        return f"{d:.0f} m from site, NO history, spike (z={z:.1f} / {r.frp:.0f} MW)"
    if d < 500 and t in ("waste_landfill", "quarry_mining") and r.frp < 80:
        return f"{d:.0f} m from a {t}"
    if lc == "cropland" and r.is_agri_burn_season == 1 and d >= 500:
        return "cropland + stubble season + not industrial"
    if lc in ("forest", "grassland", "shrubland") and d >= 500:
        return f"{lc} + {d / 1000:.1f} km from any facility"
    if d >= 2000 and lc not in ("forest", "grassland", "cropland"):
        return "no facility within 2 km, not vegetation"
    return "FALLBACK branch (no specific rule matched)"


if not os.path.exists(CORPUS):
    sys.exit(f"{CORPUS} not found - run: python tools/build_corpus.py")

df = pd.read_csv(CORPUS)
df["_id"] = range(len(df))
if os.path.exists(DB): os.remove(DB)
T = time.time()

# -- run the real pipeline once, snapshotting after every stage ----------------------
snaps = {0: df.copy()}
df = perform_spatial_join(df, str(OSM_CACHE_PATH));             snaps[1] = df.copy()
df = cluster_hotspots_spatiotemporal(df);                       snaps[2] = df.copy()
df = update_persistence_log(df, DB);                            snaps[3] = df.copy()
engine = ModelBAnomalyEngine(ARTIFACTS_DIR, DB); engine.fit_from_db()
df = build_feature_table(df, anomaly_engine=engine);            snaps[4] = df.copy()
df = apply_labeling_heuristic(df);                              snaps[5] = df.copy()

PICKS = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else auto_pick(df)
NAMES = list(PICKS)

def bar(t): print("\n" + "=" * 86 + f"\n  {t}\n" + "=" * 86)
def show(frame, rows):
    print("  " + " " * 30 + "".join(f"{n:>{W}}" for n in NAMES))
    for label, col, fmt in rows:
        vals = [fmt(frame.loc[frame._id == PICKS[n], col].iloc[0]) if col in frame.columns else "—"
                for n in NAMES]
        print(f"  {label:30s}" + "".join(f"{str(v):>{W}}" for v in vals))

f1 = lambda v: f"{float(v):,.1f}"; f2 = lambda v: f"{float(v):.2f}"
f0 = lambda v: f"{float(v):,.0f}"; s = lambda v: str(v)[:W - 2]

bar("STAGE 0 — what the satellite actually sent us (one row of the NASA FIRMS CSV)")
print("  VIIRS saw a hot pixel ~375 m across and reported this. Nothing else is known yet.\n")
show(snaps[0], [("where (lat)", "latitude", lambda v: f"{v:.4f}"),
                ("where (lon)", "longitude", lambda v: f"{v:.4f}"),
                ("when", "acq_date", s),
                ("fire radiative power (MW)", "frp", f1),
                ("4µm brightness temp (K)", "bright_ti4", f1),
                ("11µm brightness temp (K)", "bright_ti5", f1),
                ("satellite confidence", "confidence", f0),
                ("day or night", "daynight", s)])
print(f"\n  This corpus holds {len(snaps[0]):,} rows like these. The model will see all of them.")

bar("STAGE 1 — what is near it?   (spatial join against real OSM facility outlines)")
print("  For each pixel: the nearest industrial site's OUTLINE, how far away, and what kind.\n")
show(snaps[1], [("nearest facility type", "nearest_industrial_type", s),
                ("distance to it (m)", "dist_to_nearest_industrial_m", f0),
                ("distance to people (m)", "dist_to_populated_area_m", f0),
                ("distance to infrastructure (m)", "dist_to_critical_infra_m", f0)])

bar("STAGE 2 — one fire, or many?   (DBSCAN: within 750 m AND 12 h of each other)")
print("  One fire trips the satellite many times. Nearby-in-space-and-time detections merge.\n")
show(snaps[2], [("event id", "event_id", s), ("cluster growth rate", "cluster_growth_rate", f2)])
print(f"\n  {len(snaps[2]):,} detections -> {snaps[2]['event_id'].nunique():,} events.")

bar("STAGE 3 — has it burned here before?   (persistence log, keyed per site)")
print("  A refinery flare burns nightly — normal. An accident has no history — suspicious.\n")
show(snaps[3], [("days active, last 30", "days_active_last_30", f0),
                ("duty cycle (0-1)", "duty_cycle", f2),
                ("fraction seen at night", "night_detection_fraction", f2)])

bar("STAGE 4 — ground, weather, physics   (ESA WorldCover · Open-Meteo · derived)")
show(snaps[4], [("land cover", "land_cover_class", s),
                ("temperature (C)", "temperature", f1),
                ("humidity (%)", "humidity", f1),
                ("wind (km/h)", "wind_speed", f1),
                ("month", "month", f0),
                ("in stubble-burning season?", "is_agri_burn_season", lambda v: "yes" if int(v) else "no"),
                ("band gap ti4-ti5 (K)", "delta_bt", f1),
                ("power per pixel area", "frp_density", f1),
                ("z vs this site's normal", "frp_zscore_vs_facility_baseline", f2)])

bar("STAGE 5 — a RULE assigns the training label   (this is the weak-label step)")
print("  No human labelled these. An if/else over the features above decides, in this order:\n")
show(snaps[5], [("LABEL", "event_type", s)])
print()
for n in NAMES:
    print(f"  {n:12s} because: {why(snaps[5].loc[snaps[5]._id == PICKS[n]].iloc[0])}")
print("\n  The same rule over all rows gives the training set:")
for k, v in snaps[5]["event_type"].value_counts().items():
    print(f"    {k:24s} {v:6,}")

bar("STAGE 6 — TRAINING   (LightGBM learns to reproduce those labels from its features)")
a = ModelAClassifier(ARTIFACTS_DIR)
print(f"  Input per row : {len(a.feature_names)} numbers -> {', '.join(a.feature_names[:6])}, ...")
print(f"  Answer per row: the label from stage 5")
print(f"  Method        : 300 decision trees, each correcting the last one's mistakes\n")
t = time.time(); m = a.train(df); dt = time.time() - t
print(f"\n  trained on {m['n_train']:,} rows, checked on {m['n_val']:,} held-out rows, in {dt:.1f} s")

bar("STAGE 7 — ask the trained model about the three detections")
print("  This is what the API does for every detection it returns.\n")
for n in NAMES:
    r = df.loc[df._id == PICKS[n]].iloc[0]
    p = a.predict_point(r)
    probs = sorted(p.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:2]
    runner = f"{probs[1][0]} {probs[1][1]:.3f}" if len(probs) > 1 else "—"
    print(f"  {n:12s} -> {p['event_type']:22s} confidence {p['event_type_confidence']:.3f}")
    print(f"  {'':12s}    runner-up: {runner}")
    print(f"  {'':12s}    why (top SHAP): {', '.join(p.get('top_shap_features', []))}\n")

print(f"  raw satellite rows -> trained model -> explained predictions: {time.time() - T:.0f} s")
print(f"  picks used: {json.dumps(PICKS)}")
