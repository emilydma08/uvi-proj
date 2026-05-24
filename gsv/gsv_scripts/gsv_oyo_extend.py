import pandas as pd
import numpy as np
import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GSV_KEY")

STEP_DEG = 0.001
GSV_RADIUS_M = 75
HEADINGS = [0, 90, 180, 270]
MAX_WORKERS = 10
TARGET_LOCATIONS = 5000
EXISTING_CSV = "gsv_oyo_metadata2.csv"
OUTPUT_CSV = "gsv_oyo_metadata3.csv"

NEW_TILES = [
    (3.85, 7.31, 3.90, 7.36),
    (3.90, 7.31, 3.95, 7.36),
    (3.95, 7.31, 4.00, 7.36),
    (3.85, 7.36, 3.90, 7.41),
    (3.90, 7.36, 3.95, 7.41),
    (3.95, 7.36, 4.00, 7.41),
    (4.00, 7.36, 4.05, 7.41),
    (4.05, 7.36, 4.10, 7.41),
    (3.85, 7.41, 3.90, 7.46),
    (3.90, 7.41, 3.95, 7.46),
    (3.95, 7.41, 4.00, 7.46),
    (4.00, 7.41, 4.05, 7.46),
    (4.05, 7.41, 4.10, 7.46),
]

# load existing pano_ids to avoid duplicates
existing = pd.read_csv(EXISTING_CSV)
seen_panos = set(existing["pano_id"].tolist())
print(f"Loaded {len(seen_panos):,} existing pano_ids")

# build + subsample grid
all_points = []
seen_coords = set()
for (min_lon, min_lat, max_lon, max_lat) in NEW_TILES:
    for lat in np.arange(min_lat, max_lat, STEP_DEG):
        for lon in np.arange(min_lon, max_lon, STEP_DEG):
            key = (round(lat, 6), round(lon, 6))
            if key not in seen_coords:
                seen_coords.add(key)
                all_points.append(key)

if len(all_points) > TARGET_LOCATIONS:
    indices = np.linspace(0, len(all_points) - 1, TARGET_LOCATIONS, dtype=int)
    all_points = [all_points[i] for i in indices]

print(f"Checking {len(all_points):,} new points...")

def check_gsv_metadata(lat, lon):
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"location": f"{lat},{lon}", "radius": GSV_RADIUS_M, "key": API_KEY},
            timeout=10
        )
        data = resp.json()
        return data if data.get("status") == "OK" else None
    except Exception as e:
        print(f"  Error at ({lat},{lon}): {e}")
        return None

def build_url(lat, lon, heading):
    return (f"https://maps.googleapis.com/maps/api/streetview"
            f"?size=640x640&location={lat},{lon}&fov=90&heading={heading}&pitch=0&key={API_KEY}")

new_results = []
completed = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(check_gsv_metadata, lat, lon): (lat, lon) for lat, lon in all_points}
    for future in as_completed(futures):
        lat, lon = futures[future]
        completed += 1
        meta = future.result()
        if meta:
            pano_id = meta.get("pano_id")
            if pano_id and pano_id not in seen_panos:
                seen_panos.add(pano_id)
                new_results.append({
                    "lat": lat, "lon": lon, "pano_id": pano_id,
                    "date": meta.get("date"), "copyright": meta.get("copyright", ""),
                    "headings": ",".join(str(h) for h in HEADINGS),
                    "url_h0":   build_url(lat, lon, 0),
                    "url_h90":  build_url(lat, lon, 90),
                    "url_h180": build_url(lat, lon, 180),
                    "url_h270": build_url(lat, lon, 270),
                })
        if completed % 200 == 0 or completed == len(all_points):
            print(f"  {completed:,}/{len(all_points):,} | New unique: {len(new_results):,}")

# merge and save
combined = pd.concat([existing, pd.DataFrame(new_results)], ignore_index=True)
combined.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(combined):,} total locations to {OUTPUT_CSV}")
print(f"  ({len(existing):,} original + {len(new_results):,} new)")