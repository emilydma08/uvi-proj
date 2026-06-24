import os
import csv
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
gsv_key = os.getenv("GSV_KEY")

API_KEY = gsv_key
OUTPUT_CSV = "gsv_abuja_75M_15000L_0.001SD.csv"

STEP_DEG = 0.001        # ~110m spacing
GSV_RADIUS_M = 75
HEADINGS = [0, 90, 180, 270]
MAX_WORKERS = 10
TARGET_LOCATIONS = 15000


# Abuja FCT urban tiles (lon_min, lat_min, lon_max, lat_max)
# Core: Maitama, Wuse, Garki, Asokoro, Central Area
# Extended: Gwarinpa, Kubwa, Lugbe, Airport corridor
TILES = [
    # ── Central Area / Three Arms Zone ──────────────────────────
    (7.48, 9.04, 7.53, 9.08),
    (7.53, 9.04, 7.58, 9.08),

    # ── Garki (Districts 1 & 2) ──────────────────────────────────
    (7.48, 9.00, 7.53, 9.04),
    (7.53, 9.00, 7.58, 9.04),
    (7.58, 9.00, 7.63, 9.04),

    # ── Wuse (Zones 1–7) ─────────────────────────────────────────
    (7.45, 9.04, 7.50, 9.08),
    (7.45, 9.08, 7.50, 9.12),
    (7.50, 9.08, 7.55, 9.12),
    (7.55, 9.08, 7.60, 9.12),

    # ── Maitama ──────────────────────────────────────────────────
    (7.48, 9.08, 7.53, 9.12),
    (7.53, 9.08, 7.58, 9.12),
    (7.53, 9.12, 7.58, 9.16),
    (7.48, 9.12, 7.53, 9.16),

    # ── Asokoro / Mabushi ────────────────────────────────────────
    (7.53, 8.97, 7.58, 9.00),
    (7.58, 8.97, 7.63, 9.00),
    (7.63, 8.97, 7.68, 9.00),
    (7.58, 9.00, 7.63, 9.04),
    (7.63, 9.00, 7.68, 9.04),

    # ── Utako / Jabi / Wuse 2 ────────────────────────────────────
    (7.42, 9.04, 7.47, 9.08),
    (7.42, 9.08, 7.47, 9.12),
    (7.38, 9.08, 7.43, 9.12),

    # ── Gudu / Apo / Lokogoma ────────────────────────────────────
    (7.48, 8.93, 7.53, 8.97),
    (7.53, 8.93, 7.58, 8.97),
    (7.58, 8.93, 7.63, 8.97),
    (7.63, 8.93, 7.68, 8.97),

    # ── Lugbe / Airport corridor ─────────────────────────────────
    (7.30, 8.97, 7.35, 9.01),
    (7.35, 8.97, 7.40, 9.01),
    (7.40, 8.97, 7.45, 9.01),
    (7.45, 8.97, 7.50, 9.01),
    (7.30, 8.93, 7.35, 8.97),
    (7.35, 8.93, 7.40, 8.97),
    (7.40, 8.93, 7.45, 8.97),

    # ── Kubwa ────────────────────────────────────────────────────
    (7.28, 9.13, 7.33, 9.17),
    (7.33, 9.13, 7.38, 9.17),
    (7.38, 9.13, 7.43, 9.17),
    (7.28, 9.17, 7.33, 9.21),
    (7.33, 9.17, 7.38, 9.21),

    # ── Gwarinpa / Kado ──────────────────────────────────────────
    (7.38, 9.12, 7.43, 9.16),
    (7.43, 9.12, 7.48, 9.16),
    (7.38, 9.16, 7.43, 9.20),
    (7.43, 9.16, 7.48, 9.20),

    # ── Life Camp / Jahi ─────────────────────────────────────────
    (7.43, 9.08, 7.48, 9.12),
    (7.43, 9.04, 7.48, 9.08),

    # ── Nyanya / Karu (eastern satellite) ────────────────────────
    (7.60, 8.93, 7.65, 8.97),
    (7.65, 8.93, 7.70, 8.97),
    (7.60, 8.97, 7.65, 9.01),
    (7.65, 8.97, 7.70, 9.01),

    # ── Kaura / Galadimawa ───────────────────────────────────────
    (7.40, 8.97, 7.45, 9.01),
    (7.35, 9.01, 7.40, 9.05),
    (7.40, 9.01, 7.45, 9.05),
    (7.45, 9.01, 7.50, 9.05),
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