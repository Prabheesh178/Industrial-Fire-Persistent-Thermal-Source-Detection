"""
Export Clean Deploy Bundle for Render
=====================================

Builds a self-contained deploy directory (thermalwatch-api-deploy/) containing
ONLY what the runtime API and live FIRMS refresh engine need:
- api/ (main.py, schemas.py, refresh_worker.py)
- models/ (model_a_classifier.py, model_b_anomaly_engine.py, model_c_risk_model.py, score_combiner.py)
- config/ (settings.py)
- data_ingestion/ (fetch_firms.py, fetch_landcover.py, fetch_weather.py)
- processing/ (spatial_join.py, clustering.py, persistence_log.py, feature_engineering.py)
- artifacts/ (model_a.joblib, facility_baselines.json, model_c.joblib)
- data/ (feature_table.csv, hotspots.db, facilities_slim.json, landcover_cache.json, weather_cache.json)
- requirements-serve.txt, render.yaml, .gitignore, .env.example, README.md, DEPLOY_RENDER.md, OUTPUT_SCHEMA.md

Explicitly EXCLUDES:
- train_all.py, training scripts, trace_training.py, build_corpus.py, pbf_to_osm_cache.py
- raw/labeled datasets (labeled_dataset.csv, firms_raw.csv)
- uncompressed 64 MB osm_cache.json
- reports & guides (DASHBOARD_GAP_REPORT.md, TRAINING_GUIDE.md, MODEL_EVALUATION_REPORT.md)
- tests/, notebooks, .env
"""

import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = REPO_ROOT.parent / "thermalwatch-api-deploy"

# Explicit allowlist of files to copy: (source_rel_path, dest_rel_path)
DEPLOY_ALLOWLIST = [
    # Top-level configs & docs
    ("requirements-serve.txt", "requirements-serve.txt"),
    ("render.yaml", "render.yaml"),
    (".env.example", ".env.example"),
    ("OUTPUT_SCHEMA.md", "OUTPUT_SCHEMA.md"),
    ("DEPLOY_RENDER.md", "DEPLOY_RENDER.md"),

    # API package
    ("api/__init__.py", "api/__init__.py"),
    ("api/main.py", "api/main.py"),
    ("api/schemas.py", "api/schemas.py"),
    ("api/refresh_worker.py", "api/refresh_worker.py"),

    # Configuration package
    ("config/__init__.py", "config/__init__.py"),
    ("config/settings.py", "config/settings.py"),

    # Models package
    ("models/__init__.py", "models/__init__.py"),
    ("models/model_a_classifier.py", "models/model_a_classifier.py"),
    ("models/model_b_anomaly_engine.py", "models/model_b_anomaly_engine.py"),
    ("models/model_c_risk_model.py", "models/model_c_risk_model.py"),
    ("models/score_combiner.py", "models/score_combiner.py"),

    # Ingestion & processing modules needed for live refresh
    ("data_ingestion/__init__.py", "data_ingestion/__init__.py"),
    ("data_ingestion/fetch_firms.py", "data_ingestion/fetch_firms.py"),
    ("data_ingestion/fetch_landcover.py", "data_ingestion/fetch_landcover.py"),
    ("data_ingestion/fetch_weather.py", "data_ingestion/fetch_weather.py"),

    ("processing/__init__.py", "processing/__init__.py"),
    ("processing/spatial_join.py", "processing/spatial_join.py"),
    ("processing/clustering.py", "processing/clustering.py"),
    ("processing/persistence_log.py", "processing/persistence_log.py"),
    ("processing/feature_engineering.py", "processing/feature_engineering.py"),

    # Trained Model Artifacts
    ("artifacts/model_a.joblib", "artifacts/model_a.joblib"),
    ("artifacts/facility_baselines.json", "artifacts/facility_baselines.json"),
    ("artifacts/model_c.joblib", "artifacts/model_c.joblib"),

    # Serving Data Caches (NO osm_cache.json, NO labeled_dataset.csv)
    ("data/feature_table.csv", "data/feature_table.csv"),
    ("data/hotspots.db", "data/hotspots.db"),
    ("data/facilities_slim.json", "data/facilities_slim.json"),
    ("data/landcover_cache.json", "data/landcover_cache.json"),
    ("data/weather_cache.json", "data/weather_cache.json"),
]


def export_bundle(dest_dir: Path = DEPLOY_DIR):
    print(f"[Export] Rebuilding deploy bundle at: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing content but preserve .git directory if present
    for item in dest_dir.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except Exception:
                pass


    copied_files = []
    total_bytes = 0

    for src_rel, dst_rel in DEPLOY_ALLOWLIST:
        src_path = REPO_ROOT / src_rel
        dst_path = dest_dir / dst_rel
        if not src_path.exists():
            print(f"[Export] WARNING: Source file missing: {src_rel}")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        sz = dst_path.stat().st_size
        total_bytes += sz
        copied_files.append((dst_rel, sz))

    # Add deploy-specific .gitignore
    deploy_gitignore = """# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class

# Environments
.env
.env*
.venv/
venv/

# Caches and OS
.DS_Store
Thumbs.db
*.log
"""
    with open(dest_dir / ".gitignore", "w", encoding="utf-8") as f:
        f.write(deploy_gitignore)

    # Copy deploy README from tools/DEPLOY_README.md if present, else fallback
    deploy_readme_src = REPO_ROOT / "tools" / "DEPLOY_README.md"
    if deploy_readme_src.exists():
        shutil.copy2(deploy_readme_src, dest_dir / "README.md")
    else:
        deploy_readme = """# Thermal Watch API Service (Render Deploy)
...
"""
        with open(dest_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(deploy_readme)

    print(f"\n[Export] Complete. Total files: {len(copied_files) + 2}, Total size: {round(total_bytes / (1024*1024), 2)} MB")
    return copied_files, total_bytes


if __name__ == "__main__":
    export_bundle()
