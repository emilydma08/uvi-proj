import os
import csv
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
gsv_key = os.getenv("GSV_KEY")

API_KEY = gsv_key
OUTPUT_CSV = "gsv_edo_metadata.csv"

STEP_DEG = 0.001        # ~110m spacing
GSV_RADIUS_M = 150
HEADINGS = [0, 90, 180, 270]
MAX_WORKERS = 10
TARGET_LOCATIONS = 15000


# Abuja FCT urban tiles (lon_min, lat_min, lon_max, lat_max)
# Core: Maitama, Wuse, Garki, Asokoro, Central Area
# Extended: Gwarinpa, Kubwa, Lugbe, Airport corridor
TILES = [
    # ── Auchi (clusters 1086–1087) ───────────────────────────
    (6.08, 7.27, 6.13, 7.32),
    (6.13, 7.27, 6.18, 7.32),
    (6.08, 7.32, 6.13, 7.37),  # buffer north
    (6.35, 7.27, 6.40, 7.32),
    (6.40, 7.27, 6.45, 7.32),

    # ── Ekpoma / Uromi (clusters 1088–1089) ──────────────────
    (6.27, 7.05, 6.32, 7.10),
    (6.32, 7.05, 6.37, 7.10),
    (6.37, 7.05, 6.42, 7.10),
    (6.47, 7.05, 6.52, 7.10),
    (6.42, 7.05, 6.47, 7.10),

    # ── Sabongida-Ora (cluster 1090) ─────────────────────────
    (5.90, 6.97, 5.95, 7.02),
    (5.95, 6.97, 6.00, 7.02),

    # ── Agbor / Owa (clusters 1091–1093) ─────────────────────
    (6.10, 6.67, 6.15, 6.72),
    (6.15, 6.67, 6.20, 6.72),
    (6.20, 6.67, 6.25, 6.72),
    (6.35, 6.67, 6.40, 6.72),
    (6.25, 6.72, 6.30, 6.77),
    (6.30, 6.72, 6.35, 6.77),
    (6.35, 6.72, 6.40, 6.77),
    (6.25, 6.77, 6.30, 6.82),
    (6.30, 6.77, 6.35, 6.82),

    # ── Abraka / Ughelli fringe (clusters 1094–1095) ─────────
    (6.25, 6.51, 6.30, 6.56),
    (6.30, 6.51, 6.35, 6.56),
    (6.25, 6.56, 6.30, 6.61),
    (6.30, 6.56, 6.35, 6.61),

    # ── Benin City core (clusters 1096–1104) ─────────────────
    (5.58, 6.28, 5.63, 6.33),
    (5.63, 6.28, 5.68, 6.33),
    (5.68, 6.28, 5.73, 6.33),
    (5.58, 6.33, 5.63, 6.38),
    (5.63, 6.33, 5.68, 6.38),
    (5.68, 6.33, 5.73, 6.38),
    (5.58, 6.38, 5.63, 6.43),  # buffer north
    (5.63, 6.38, 5.68, 6.43),
    (5.68, 6.38, 5.73, 6.43),

    # ── Sapele (cluster 1105) ─────────────────────────────────
    (6.15, 5.95, 6.20, 6.00),
    (6.20, 5.95, 6.25, 6.00),
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