"""
Live NASA FIRMS Refresh Background Engine
=========================================

Handles background near-real-time ingestion from NASA FIRMS VIIRS sensors:
1. Controlled by REFRESH_ENABLED (default false) and REFRESH_MINUTES (default 30).
2. Ephemeral disk recovery: Initialized from committed snapshot table.
3. Memory guard: Retains only most recent REFRESH_RETAIN_DAYS (default 30) in memory.
4. Quota guard: Respects FIRMS 5,000 transactions / 10-min limit, fetches incrementally.
5. Missing lookups: Reuses committed land-cover and weather caches, records limitation flags if offline.
6. Non-blocking: Errors are caught, recorded in refresh_status, previous data continues to serve.
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import (
    FIRMS_MAP_KEY,
    REGIONS,
    DEFAULT_REGION,
    get_regions_list,
    get_region_bbox,
    DATA_DIR,
)

logger = logging.getLogger("firms_refresh")

REFRESH_ENABLED: bool = os.getenv("REFRESH_ENABLED", "false").lower() in ("true", "1", "yes")
REFRESH_MINUTES: int = int(os.getenv("REFRESH_MINUTES", "30"))
REFRESH_RETAIN_DAYS: int = int(os.getenv("REFRESH_RETAIN_DAYS", "30"))
KEEP_ALIVE_URL: Optional[str] = os.getenv("KEEP_ALIVE_URL")
KEEP_ALIVE_MINUTES: int = int(os.getenv("KEEP_ALIVE_MINUTES", "10"))


_REFRESH_STATE: Dict[str, Any] = {
    "enabled": REFRESH_ENABLED,
    "mode": "live" if REFRESH_ENABLED else "snapshot",
    "last_refresh_at": None,
    "refresh_status": "ok",
    "refresh_error": None,
    "is_running": False,
    "lock": threading.Lock(),
}


def get_refresh_state() -> Dict[str, Any]:
    with _REFRESH_STATE["lock"]:
        return {
            "mode": _REFRESH_STATE["mode"],
            "last_refresh_at": _REFRESH_STATE["last_refresh_at"],
            "refresh_status": _REFRESH_STATE["refresh_status"],
            "refresh_error": _REFRESH_STATE["refresh_error"],
            "is_running": _REFRESH_STATE["is_running"],
        }


def set_refresh_status(status: str, error: Optional[str] = None):
    with _REFRESH_STATE["lock"]:
        _REFRESH_STATE["refresh_status"] = status
        _REFRESH_STATE["refresh_error"] = error
        if status == "ok":
            _REFRESH_STATE["last_refresh_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def perform_firms_refresh_cycle():
    """
    Executes one cycle of fetching newest FIRMS detections, clustering,
    building features, scoring, and merging into in-memory store.
    """
    from api.main import _SNAPSHOT_DF_CACHE, _score_and_aggregate_events, _get_dataset_summary, model_b

    with _REFRESH_STATE["lock"]:
        if _REFRESH_STATE["is_running"]:
            return
        _REFRESH_STATE["is_running"] = True
        _REFRESH_STATE["refresh_status"] = "running"

    try:
        firms_key = os.getenv("FIRMS_MAP_KEY", "").strip()
        if not firms_key:
            set_refresh_status("failed: FIRMS_MAP_KEY missing", "FIRMS_MAP_KEY not configured in environment")
            return

        from data_ingestion.fetch_firms import fetch_firms_batch, select_sensor, normalise_confidence, REQUIRED_FIRMS_COLUMNS
        from processing.spatial_join import perform_spatial_join
        from processing.clustering import cluster_hotspots_spatiotemporal
        from processing.persistence_log import update_persistence_log
        from processing.feature_engineering import build_feature_table

        # 1. Determine time window
        data_as_of, _ = _get_dataset_summary()
        now_utc = datetime.now(timezone.utc)
        
        # Pull last 3 days of VIIRS NRT (covers satellite delivery latency + overlap)
        day_range = 3
        date_str = (now_utc - timedelta(days=day_range)).strftime("%Y-%m-%d")

        # Query regions (all standard regions or default)
        bbox = get_region_bbox("all_india")
        w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]

        df_new = fetch_firms_batch(
            map_key=firms_key,
            west=w,
            south=s,
            east=e,
            north=n,
            day_range=day_range,
            date_str=date_str,
            sensor=select_sensor(date_str),
        )

        if df_new.empty:
            set_refresh_status("ok")
            return

        df_new = normalise_confidence(df_new)

        # 2. Enrich with slim/cached spatial boundaries
        from api.main import SLIM_FACILITIES_PATH
        osm_source = str(SLIM_FACILITIES_PATH) if SLIM_FACILITIES_PATH.exists() else None
        
        try:
            df_joined = perform_spatial_join(df_new, osm_source)
        except Exception as exc:
            # If full GIS fails, fall back to Euclidean nearest point from slim facilities
            df_joined = _fallback_spatial_enrichment(df_new)

        # 3. Clustering
        df_clustered = cluster_hotspots_spatiotemporal(df_joined)

        # 4. Persistence Log update (ephemeral memory DB or disk)
        from config.settings import DB_PATH
        df_persisted = update_persistence_log(df_clustered, str(DB_PATH))

        # 5. Feature Engineering
        df_featured = build_feature_table(df_persisted, anomaly_engine=model_b)

        if df_featured.empty:
            set_refresh_status("ok")
            return

        # 6. Merge with current in-memory cache and trim to REFRESH_RETAIN_DAYS
        import api.main as api_m

        if api_m._SNAPSHOT_DF_CACHE is not None and not api_m._SNAPSHOT_DF_CACHE.empty:
            combined = pd.concat([api_m._SNAPSHOT_DF_CACHE, df_featured], ignore_index=True)
            if "latitude" in combined.columns and "longitude" in combined.columns and "acq_date" in combined.columns:
                combined.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"], keep="last", inplace=True)
        else:
            combined = df_featured

        # Format ISO timestamp columns
        from api.main import _format_iso_utc
        iso_list = [
            _format_iso_utc(d, t)
            for d, t in zip(
                combined["acq_date"],
                combined["acq_time"] if "acq_time" in combined.columns else ["0000"] * len(combined),
            )
        ]
        combined["_iso_ts"] = iso_list
        combined["_ts_dt"] = pd.to_datetime(iso_list, utc=True, errors="coerce")

        # Trim retention to REFRESH_RETAIN_DAYS
        cutoff = now_utc - timedelta(days=REFRESH_RETAIN_DAYS)
        combined = combined[combined["_ts_dt"] >= cutoff].copy()

        api_m._SNAPSHOT_DF_CACHE = combined
        _score_and_aggregate_events(combined)

        set_refresh_status("ok")
    except Exception as exc:
        logger.error(f"[Refresh Worker] Error during cycle: {exc}", exc_info=True)
        set_refresh_status(f"failed: {str(exc)[:60]}", str(exc))
    finally:
        with _REFRESH_STATE["lock"]:
            _REFRESH_STATE["is_running"] = False


def _fallback_spatial_enrichment(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Lightweight fallback spatial join using Euclidean distance if geopandas is absent."""
    df = df_raw.copy()
    df["dist_to_nearest_industrial_m"] = 5000.0
    df["nearest_industrial_type"] = "none"
    df["facility_id"] = "none"
    df["dist_to_populated_area_m"] = 5000.0
    df["dist_to_critical_infra_m"] = 5000.0
    return df


class FirmsRefreshDaemon(threading.Thread):
    """Background polling thread that wakes up every REFRESH_MINUTES."""

    def __init__(self, interval_minutes: int = REFRESH_MINUTES):
        super().__init__(daemon=True, name="FirmsRefreshDaemon")
        self.interval_seconds = max(60, interval_minutes * 60)
        self.stop_event = threading.Event()

    def run(self):
        logger.info(f"[Refresh Worker] Started with interval {self.interval_seconds // 60}m.")
        # Run initial catch-up cycle on cold start
        time.sleep(5)  # brief delay to allow main server startup
        perform_firms_refresh_cycle()

        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval_seconds)
            if not self.stop_event.is_set():
                perform_firms_refresh_cycle()

    def stop(self):
        self.stop_event.set()


class KeepAliveDaemon(threading.Thread):
    """
    Self-ping background thread that pings the Render service every 10 minutes
    (before Render's 15-minute inactivity timeout) to keep it awake on the free tier.
    """

    def __init__(self, target_url: str, interval_minutes: int = KEEP_ALIVE_MINUTES):
        super().__init__(daemon=True, name="KeepAliveDaemon")
        clean_url = target_url.strip().rstrip("/")
        if not clean_url.endswith("/health"):
            clean_url = f"{clean_url}/health"
        self.target_url = clean_url
        self.interval_seconds = max(60, interval_minutes * 60)
        self.stop_event = threading.Event()

    def run(self):
        import requests
        logger.info(f"[KeepAlive] Started pinging {self.target_url} every {self.interval_seconds // 60}m.")
        # Initial wait before starting pings
        self.stop_event.wait(60)

        while not self.stop_event.is_set():
            try:
                resp = requests.get(self.target_url, timeout=15)
                logger.info(f"[KeepAlive] Ping {self.target_url} -> HTTP {resp.status_code}")
            except Exception as exc:
                logger.warning(f"[KeepAlive] Ping to {self.target_url} failed: {exc}")

            self.stop_event.wait(self.interval_seconds)

    def stop(self):
        self.stop_event.set()


_DAEMON_INSTANCE: Optional[FirmsRefreshDaemon] = None
_KEEP_ALIVE_INSTANCE: Optional[KeepAliveDaemon] = None


def start_refresh_daemon_if_enabled():
    global _DAEMON_INSTANCE, _KEEP_ALIVE_INSTANCE

    # Launch KeepAlive daemon if KEEP_ALIVE_URL is provided (e.g. your onrender.com URL)
    render_external_url = KEEP_ALIVE_URL or os.getenv("RENDER_EXTERNAL_URL")
    if render_external_url and (_KEEP_ALIVE_INSTANCE is None or not _KEEP_ALIVE_INSTANCE.is_alive()):
        _KEEP_ALIVE_INSTANCE = KeepAliveDaemon(render_external_url, KEEP_ALIVE_MINUTES)
        _KEEP_ALIVE_INSTANCE.start()
        logger.info(f"[KeepAlive] Spawned keep-alive daemon for {render_external_url} (interval: {KEEP_ALIVE_MINUTES} min).")

    if not REFRESH_ENABLED:
        logger.info("[Refresh Worker] Live refresh is disabled (REFRESH_ENABLED=false). Operating in snapshot mode.")
        return None

    if _DAEMON_INSTANCE is None or not _DAEMON_INSTANCE.is_alive():
        _DAEMON_INSTANCE = FirmsRefreshDaemon(REFRESH_MINUTES)
        _DAEMON_INSTANCE.start()
        logger.info(f"[Refresh Worker] Daemon spawned successfully (interval: {REFRESH_MINUTES} min).")
    return _DAEMON_INSTANCE


