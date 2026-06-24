import os
import csv
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()
gsv_key = os.getenv("GSV_KEY")

API_KEY = gsv_key
OUTPUT_CSV = "gsv_lagos_75M_15000L_0.001SD.csv"

STEP_DEG = 0.001        # 110m spacing
GSV_RADIUS_M = 75          
HEADINGS = [0, 90, 180, 270]
MAX_WORKERS = 10
TARGET_LOCATIONS = 15000       


TILES = [
    (3.1,  6.45, 3.15, 6.5 ),
    (3.1,  6.5,  3.15, 6.55),
    (3.15, 6.45, 3.2,  6.5 ),
    (3.15, 6.65, 3.2,  6.7 ),
    (3.2,  6.45, 3.25, 6.5 ),
    (3.2,  6.5,  3.25, 6.55),
    (3.2,  6.55, 3.25, 6.6 ),
    (3.25, 6.45, 3.3,  6.5 ),
    (3.25, 6.55, 3.3,  6.6 ),
    (3.25, 6.6,  3.3,  6.65),
    (3.3,  6.45, 3.35, 6.5 ),
    (3.3,  6.5,  3.35, 6.55),
    (3.3,  6.55, 3.35, 6.6 ),
    (3.3,  6.6,  3.35, 6.65),
    (3.35, 6.4,  3.4,  6.45),
    (3.35, 6.45, 3.4,  6.5 ),
    (3.35, 6.5,  3.4,  6.55),
    (3.35, 6.55, 3.4,  6.6 ),
    (3.35, 6.6,  3.4,  6.65),
    (3.4,  6.4,  3.45, 6.45),
    (3.4,  6.45, 3.45, 6.5 ),
    (3.4,  6.5,  3.45, 6.55),
    (3.4,  6.55, 3.45, 6.6 ),
    (3.4,  6.6,  3.45, 6.65),
    (3.45, 6.4,  3.5,  6.45),
    (3.45, 6.45, 3.5,  6.5 ),
    (3.45, 6.55, 3.5,  6.6 ),
    (3.45, 6.6,  3.5,  6.65),
    (3.5,  6.4,  3.55, 6.45),
    (3.5,  6.5,  3.55, 6.55),
    (3.5,  6.55, 3.55, 6.6 ),
    (3.5,  6.6,  3.55, 6.65),
    (3.5,  6.65, 3.55, 6.7 ),
    (3.55, 6.4,  3.6,  6.45),
    (3.55, 6.45, 3.6,  6.5 ),
    (3.55, 6.5,  3.6,  6.55),
    (3.55, 6.55, 3.6,  6.6 ),
    (3.55, 6.6,  3.6,  6.65),
    (3.55, 6.65, 3.6,  6.7 ),

    (3.1,  6.65, 3.15, 6.7 ),
    (3.15, 6.4,  3.2,  6.45),
    (3.15, 6.5,  3.2,  6.55),
    (3.15, 6.55, 3.2,  6.6 ),
    (3.15, 6.6,  3.2,  6.65),
    (3.2,  6.4,  3.25, 6.45),
    (3.2,  6.6,  3.25, 6.65),
    (3.2,  6.65, 3.25, 6.7 ),
    (3.25, 6.4,  3.3,  6.45),
    (3.25, 6.5,  3.3,  6.55),
    (3.25, 6.65, 3.3,  6.7 ),
    (3.3,  6.4,  3.35, 6.45),
    (3.3,  6.65, 3.35, 6.7 ),
    (3.35, 6.65, 3.4,  6.7 ),
    (3.4,  6.65, 3.45, 6.7 ),
    (3.45, 6.5,  3.5,  6.55),
    (3.45, 6.65, 3.5,  6.7 ),
    (3.5,  6.45, 3.55, 6.5 ),

    # Lekki-Epe Expressway corridor
    (3.6,  6.35, 3.65, 6.4 ),
    (3.6,  6.4,  3.65, 6.45),
    (3.6,  6.45, 3.65, 6.5 ),
    (3.6,  6.5,  3.65, 6.55),
    (3.6,  6.55, 3.65, 6.6 ),
    (3.6,  6.6,  3.65, 6.65),
    (3.6,  6.65, 3.65, 6.7 ),
    (3.65, 6.35, 3.7,  6.4 ),
    (3.65, 6.4,  3.7,  6.45),
    (3.65, 6.45, 3.7,  6.5 ),
    (3.65, 6.5,  3.7,  6.55),
    (3.65, 6.55, 3.7,  6.6 ),
    (3.65, 6.6,  3.7,  6.65),
    (3.65, 6.65, 3.7,  6.7 ),
    (3.7,  6.35, 3.75, 6.4 ),
    (3.7,  6.4,  3.75, 6.45),
    (3.7,  6.45, 3.75, 6.5 ),
    (3.7,  6.5,  3.75, 6.55),
    (3.7,  6.55, 3.75, 6.6 ),
    (3.7,  6.6,  3.75, 6.65),
    (3.7,  6.65, 3.75, 6.7 ),
    (3.75, 6.35, 3.8,  6.4 ),
    (3.75, 6.4,  3.8,  6.45),
    (3.75, 6.45, 3.8,  6.5 ),
    (3.75, 6.5,  3.8,  6.55),
    (3.75, 6.55, 3.8,  6.6 ),
    (3.75, 6.6,  3.8,  6.65),
    (3.8,  6.35, 3.85, 6.4 ),
    (3.8,  6.4,  3.85, 6.45),
    (3.8,  6.45, 3.85, 6.5 ),
    (3.8,  6.5,  3.85, 6.55),
    (3.85, 6.35, 3.9,  6.4 ),
    (3.85, 6.4,  3.9,  6.45),
    (3.85, 6.45, 3.9,  6.5 ),
    (3.9,  6.35, 3.95, 6.4 ),
    (3.9,  6.4,  3.95, 6.45),
    (3.9,  6.45, 3.95, 6.5 ),

    # Lagos Island/CMS
    (3.35, 6.35, 3.4,  6.4 ),
    (3.4,  6.35, 3.45, 6.4 ),
    (3.45, 6.35, 3.5,  6.4 ),
    (3.5,  6.35, 3.55, 6.4 ),
    (3.55, 6.35, 3.6,  6.4 ),

    # Ikeja extended north
    (3.3,  6.7,  3.35, 6.75),
    (3.35, 6.7,  3.4,  6.75),
    (3.4,  6.7,  3.45, 6.75),
    (3.45, 6.7,  3.5,  6.75),
]


def build_sample_points(tiles, step, target):
    print(f"Building sample grid (step={step}deg ≈ {step*111000:.0f}m, target={target:,})...")
    all_points = []
    seen = set()

    for (min_lon, min_lat, max_lon, max_lat) in tiles:
        lats = np.arange(min_lat, max_lat, step)
        lons = np.arange(min_lon, max_lon, step)
        for lat in lats:
            for lon in lons:
                key = (round(lat, 6), round(lon, 6))
                if key not in seen:
                    seen.add(key)
                    all_points.append(key)

    print(f"  Raw grid points: {len(all_points):,}")

    if len(all_points) > target:
        indices = np.linspace(0, len(all_points) - 1, target, dtype=int)
        all_points = [all_points[i] for i in indices]
        print(f"  Subsampled to: {len(all_points):,}")

    return all_points


def check_gsv_metadata(lat, lon, api_key, radius):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "key": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK":
            return data
        return None
    except Exception as e:
        print(f"  Metadata error at ({lat},{lon}): {e}")
        return None


def build_image_url(lat, lon, heading, api_key, size="640x640", fov=90, pitch=0):
    return (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={lat},{lon}"
        f"&fov={fov}&heading={heading}&pitch={pitch}&key={api_key}"
    )


def run_pipeline():
    sample_points = build_sample_points(TILES, STEP_DEG, TARGET_LOCATIONS)
    print(f"\nTotal points to check     : {len(sample_points):,}")
    print(f"Max panoramas if all valid : {len(sample_points) * len(HEADINGS):,}")
    print(f"Max download cost          : ${len(sample_points) * len(HEADINGS) * 0.007:.2f}\n")

    print(f"Checking GSV metadata with {MAX_WORKERS} workers (radius={GSV_RADIUS_M}m)...")
    results_by_pano = {} 
    completed = 0
    total = len(sample_points)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(check_gsv_metadata, lat, lon, API_KEY, GSV_RADIUS_M): (lat, lon)
            for lat, lon in sample_points
        }

        for future in as_completed(futures):
            lat, lon = futures[future]
            completed += 1

            try:
                meta = future.result()
            except Exception as e:
                print(f"  FAILED ({lat},{lon}): {e}")
                continue

            if meta:
                pano_id = meta.get("pano_id")
                if pano_id and pano_id not in results_by_pano:
                    results_by_pano[pano_id] = {
                        "lat": lat,
                        "lon": lon,
                        "pano_id": pano_id,
                        "date": meta.get("date"),
                        "copyright": meta.get("copyright", ""),
                        "headings": ",".join(str(h) for h in HEADINGS),
                        "url_h0":   build_image_url(lat, lon, 0,   API_KEY),
                        "url_h90":  build_image_url(lat, lon, 90,  API_KEY),
                        "url_h180": build_image_url(lat, lon, 180, API_KEY),
                        "url_h270": build_image_url(lat, lon, 270, API_KEY),
                    }

            if completed % 500 == 0 or completed == total:
                pct = len(results_by_pano) / completed * 100
                print(f"  {completed:,}/{total:,} checked | Unique locations: {len(results_by_pano):,} ({pct:.1f}% yield)")

    valid_locations = list(results_by_pano.values())
    print(f"\nWriting {len(valid_locations):,} unique locations to {OUTPUT_CSV}...")

    fieldnames = ["lat", "lon", "pano_id", "date", "copyright", "headings",
                  "url_h0", "url_h90", "url_h180", "url_h270"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_locations)

    total_panos = len(valid_locations) * len(HEADINGS)
    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Unique locations with GSV coverage : {len(valid_locations):,}")
    print(f"  Total panoramas (4 headings)        : {total_panos:,}")
    print(f"  Estimated download cost             : ${total_panos * 0.007:.2f}")
    print(f"  CSV saved to                        : {os.path.abspath(OUTPUT_CSV)}")
    print(f"─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    run_pipeline()