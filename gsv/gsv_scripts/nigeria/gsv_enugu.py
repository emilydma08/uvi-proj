import os
import csv
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
gsv_key = os.getenv("GSV_KEY")

API_KEY = gsv_key
OUTPUT_CSV = "gsv_enugu_metadata.csv"

STEP_DEG = 0.001        # ~110m spacing
GSV_RADIUS_M = 150
HEADINGS = [0, 90, 180, 270]
MAX_WORKERS = 10
TARGET_LOCATIONS = 15000


TILES = [
    # ── Enugu city north (clusters 866, 868–869) ─────────────
    (7.35, 6.92, 7.40, 6.97),
    (7.40, 6.92, 7.45, 6.97),
    (7.45, 6.92, 7.50, 6.97),
    (7.50, 6.92, 7.55, 6.97),
    (7.35, 6.97, 7.40, 7.02),
    (7.40, 6.97, 7.45, 7.02),
    (7.45, 6.97, 7.50, 7.02),
    (7.50, 6.97, 7.55, 7.02),

    # ── Enugu city south / Transekulu (clusters 870–871, 874–875) ──
    (7.40, 6.82, 7.45, 6.87),
    (7.45, 6.82, 7.50, 6.87),
    (7.48, 6.87, 7.53, 6.92),
    (7.53, 6.87, 7.58, 6.92),
    (7.58, 6.87, 7.63, 6.92),
    (7.40, 6.87, 7.45, 6.92),
    (7.45, 6.87, 7.50, 6.92),
    (7.50, 6.87, 7.55, 6.92),
    (7.55, 6.87, 7.60, 6.92),
    (7.60, 6.87, 7.65, 6.92),

    # ── Awgu area (clusters 872–873) ─────────────────────────
    (7.37, 6.62, 7.42, 6.67),
    (7.42, 6.62, 7.47, 6.67),
    (7.37, 6.67, 7.42, 6.72),
    (7.42, 6.67, 7.47, 6.72),

    # ── Oji River (cluster 876) ───────────────────────────────
    (7.13, 6.79, 7.18, 6.84),
    (7.18, 6.79, 7.23, 6.84),

    # ── Agbani / Nkanu (clusters 877–886) ────────────────────
    (7.17, 6.42, 7.22, 6.47),
    (7.22, 6.42, 7.27, 6.47),
    (7.42, 6.42, 7.47, 6.47),
    (7.47, 6.42, 7.52, 6.47),
    (7.52, 6.42, 7.57, 6.47),
    (7.57, 6.42, 7.62, 6.47),
    (7.42, 6.47, 7.47, 6.52),
    (7.47, 6.47, 7.52, 6.52),
    (7.52, 6.47, 7.57, 6.52),
    (7.57, 6.47, 7.62, 6.52),
    (7.42, 6.52, 7.47, 6.57),
    (7.47, 6.52, 7.52, 6.57),
    (7.52, 6.52, 7.57, 6.57),
    (7.47, 6.57, 7.52, 6.62),
    (7.52, 6.57, 7.57, 6.62),

    # ── Okposi fringe (cluster 882) ───────────────────────────
    (7.66, 6.14, 7.71, 6.19),
    (7.71, 6.14, 7.76, 6.19),
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