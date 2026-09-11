# tools/

Reproduction scripts for the ingestion and training fixes. Optional — nothing in
`api/`, `models/` or `processing/` imports them. Run from the repo root.

| script | what it does |
|---|---|
| `build_corpus.py` | Pulls a multi-region, multi-season FIRMS corpus and writes both `data/firms_corpus.csv` (every detection) and `data/firms_corpus_balanced.csv` (capped per pull — what training reads). Recent data alone yields 5 of the 7 label classes: `agricultural_burn` needs the Oct–Nov stubble window and `known_false_positive` needs a solar park in bbox. `--balance-only` re-balances an existing corpus without refetching. |
| `pbf_to_osm_cache.py` | Builds `data/osm_cache.json` from Geofabrik `.osm.pbf` extracts instead of Overpass. Output matches the Overpass `out geom;` shape `parse_osm_geometries` expects. |
| `run_training.py` | Trains A/B/C on a prebuilt corpus. `models/train_all.py` refetches from FIRMS every run and only expresses "N days back from today", so it cannot train across two seasons at once. Aborts if the OSM cache looks like the synthetic fallback, and prints the class distribution before training. |
| `verify_spatial_join.py` | Regression check that the `sjoin_nearest` rewrite selects the same nearest facility as the original per-point scan. Needs `SPATIAL_JOIN_BASELINE` pointing at a pre-fix copy of `processing/spatial_join.py`. |
| `trace_training.py` | Follows three real detections — a crop fire, a spreading forest fire, one inside an industrial site — through all eight stages, then trains Model A and asks it about them. The clearest way to see what the pipeline does. Picks its own rows; pass a `picks.json` to choose others. |

## Why Geofabrik rather than Overpass

`/hotspots` calls Overpass on every request. Overpass rate-limits hard: a
region-sized query with `[timeout:60]` returns 504, a union bbox returns 406, and
sustained retries earn a **TCP-level firewall block** of your IP across
`overpass-api.de` and its `lz4`/`z` mirrors. That happened during this work and took
hours to clear.

Geofabrik extracts are static files (~200–335 MB per India zone), so this path has no
rate limit and no dependency on a third party being reachable at demo time.

```bash
curl -L -o data/pbf/northern-zone.osm.pbf \
  https://download.geofabrik.de/asia/india/northern-zone-latest.osm.pbf
python tools/pbf_to_osm_cache.py
```

## Typical sequence

```bash
python tools/build_corpus.py          # FIRMS -> firms_corpus.csv + firms_corpus_balanced.csv
python tools/pbf_to_osm_cache.py      # PBF   -> data/osm_cache.json
python tools/run_training.py          # -> artifacts/model_{a,c}.joblib
python tools/trace_training.py        # optional: watch three detections go through it
```

Already have `data/firms_corpus.csv`? `python tools/build_corpus.py --balance-only` rebuilds the balanced file without touching FIRMS.

Land-cover and weather caches are disk-backed, so a second run makes no network calls.
