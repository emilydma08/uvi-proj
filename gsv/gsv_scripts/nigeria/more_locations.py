import os
import csv
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
gsv_key = os.getenv("GSV_KEY")

API_KEY = gsv_key
OUTPUT_CSV = "more_merged_metadata.csv"

STEP_DEG = 0.001        # ~110m spacing
GSV_RADIUS_M = 150
HEADINGS = [0, 90, 180, 270]
MAX_WORKERS = 12
TARGET_LOCATIONS = 25000


TILES = [
    # ── Warri / Effurun (clusters 1127–1135) ─────────────────
    (5.74, 4.73, 5.79, 4.78),
    (5.79, 4.73, 5.84, 4.78),
    (5.84, 4.73, 5.89, 4.78),
    (5.89, 4.73, 5.94, 4.78),
    (5.94, 4.73, 5.99, 4.78),
    (5.74, 4.78, 5.79, 4.83),
    (5.79, 4.78, 5.84, 4.83),
    (5.84, 4.78, 5.89, 4.83),
    (5.89, 4.78, 5.94, 4.83),
    (5.99, 4.78, 6.04, 4.83),
    (6.04, 4.78, 6.09, 4.83),
    (5.74, 4.83, 5.79, 4.88),
    (5.79, 4.83, 5.84, 4.88),
    (5.84, 4.83, 5.89, 4.88),
    (5.89, 4.83, 5.94, 4.88),
    (6.09, 4.83, 6.14, 4.88),
    (6.14, 4.83, 6.19, 4.88),
    (6.09, 4.88, 6.14, 4.93),
    (6.14, 4.88, 6.19, 4.93),

    # ── Agbor / Asaba (clusters 799, 805, 808, 813) ──────────
    (6.80, 6.02, 6.85, 6.07),
    (6.85, 6.02, 6.90, 6.07),
    (6.80, 6.07, 6.85, 6.12),
    (6.85, 6.07, 6.90, 6.12),
    (6.90, 6.07, 6.95, 6.12),
    (6.80, 6.12, 6.85, 6.17),
    (6.85, 6.12, 6.90, 6.17),
    (7.05, 6.21, 7.10, 6.26),
    (7.10, 6.21, 7.15, 6.26),

    # ── Yenagoa (clusters 755–756) ────────────────────────────
    (7.82, 5.57, 7.87, 5.62),
    (7.87, 5.57, 7.92, 5.62),
    (7.82, 5.62, 7.87, 5.67),
    (7.87, 5.62, 7.92, 5.67),

    # ── Port Harcourt south (clusters 757–762) ────────────────
    (7.33, 5.08, 7.38, 5.13),
    (7.38, 5.08, 7.43, 5.13),
    (7.33, 5.13, 7.38, 5.18),
    (7.38, 5.13, 7.43, 5.18),
    (7.43, 5.13, 7.48, 5.18),

    # ── Akure (clusters 118–125) ──────────────────────────────
    (4.46, 8.47, 4.51, 8.52),
    (4.51, 8.47, 4.56, 8.52),
    (4.56, 8.47, 4.61, 8.52),
    (4.46, 8.52, 4.51, 8.57),
    (4.51, 8.52, 4.56, 8.57),
    (4.56, 8.52, 4.61, 8.57),

    # ── Ondo town (clusters 129–130) ─────────────────────────
    (4.69, 8.12, 4.74, 8.17),
    (4.74, 8.12, 4.79, 8.17),

    # ── Osogbo (clusters 1250–1254) ──────────────────────────
    (3.31, 7.11, 3.36, 7.16),
    (3.36, 7.11, 3.41, 7.16),
    (3.31, 7.16, 3.36, 7.21),
    (3.36, 7.16, 3.41, 7.21),
    (3.41, 7.16, 3.46, 7.21),

    # ── Abeokuta (clusters 1255, 1261) ───────────────────────
    (3.19, 6.80, 3.24, 6.85),
    (3.24, 6.80, 3.29, 6.85),
    (3.29, 6.80, 3.34, 6.85),
    (3.34, 6.80, 3.39, 6.85),
    (3.39, 6.80, 3.44, 6.85),
    (3.19, 6.85, 3.24, 6.90),
    (3.24, 6.85, 3.29, 6.90),

    # ── Ogun south / Sagamu (clusters 1257–1260) ─────────────
    (3.12, 6.53, 3.17, 6.58),
    (3.17, 6.53, 3.22, 6.58),
    (3.22, 6.53, 3.27, 6.58),
    (3.27, 6.53, 3.32, 6.58),
    (3.12, 6.58, 3.17, 6.63),
    (3.17, 6.58, 3.22, 6.63),
    (3.22, 6.58, 3.27, 6.63),
    (3.27, 6.58, 3.32, 6.63),
    (3.32, 6.63, 3.37, 6.68),
    (3.37, 6.63, 3.42, 6.68),

    # ── Ijebu-Ode / Shagamu (clusters 1263–1267) ─────────────
    (3.63, 6.82, 3.68, 6.87),
    (3.68, 6.82, 3.73, 6.87),
    (3.63, 6.87, 3.68, 6.92),
    (3.68, 6.87, 3.73, 6.92),
    (3.73, 6.87, 3.78, 6.92),
    (3.91, 6.97, 3.96, 7.02),
    (3.96, 6.97, 4.01, 7.02),
    (4.01, 6.97, 4.06, 7.02),

    # ── Ado-Ekiti (clusters 1288–1290) ───────────────────────
    (5.69, 7.49, 5.74, 7.54),
    (5.74, 7.49, 5.79, 7.54),
    (5.79, 7.49, 5.84, 7.54),
    (5.69, 7.54, 5.74, 7.59),
    (5.74, 7.54, 5.79, 7.59),

    # ── Ikere / Igbara-Oke (clusters 1295–1297) ──────────────
    (5.19, 7.19, 5.24, 7.24),
    (5.24, 7.19, 5.29, 7.24),
    (5.55, 7.19, 5.60, 7.24),
    (5.60, 7.19, 5.65, 7.24),

    # ── Ondo state fringe (clusters 1300–1303) ────────────────
    (4.73, 6.57, 4.78, 6.62),
    (4.78, 6.57, 4.83, 6.62),
    (4.83, 6.72, 4.88, 6.77),
    (4.88, 6.72, 4.93, 6.77),
    (4.83, 7.07, 4.88, 7.12),
    (4.88, 7.07, 4.93, 7.12),

    # ── Ilesha / Ife (clusters 1324–1334) ────────────────────
    (4.45, 7.76, 4.50, 7.81),
    (4.50, 7.76, 4.55, 7.81),
    (4.55, 7.76, 4.60, 7.81),
    (4.45, 7.81, 4.50, 7.86),
    (4.50, 7.81, 4.55, 7.86),
    (4.55, 7.81, 4.60, 7.86),
    (4.60, 7.81, 4.65, 7.86),
    (4.65, 7.81, 4.70, 7.86),
    (4.70, 7.81, 4.75, 7.86),
    (4.45, 7.86, 4.50, 7.91),
    (4.50, 7.86, 4.55, 7.91),
    (4.55, 7.86, 4.60, 7.91),
    (4.60, 7.86, 4.65, 7.91),
    (4.65, 7.86, 4.70, 7.91),
    (4.70, 7.86, 4.75, 7.91),
    (4.75, 7.86, 4.80, 7.91),
    (4.80, 7.86, 4.85, 7.91),
    (4.85, 7.86, 4.90, 7.91),
    (4.90, 7.86, 4.95, 7.91),
    (4.45, 7.91, 4.50, 7.96),
    (4.50, 7.91, 4.55, 7.96),
    (4.55, 7.91, 4.60, 7.96),
    (4.60, 7.91, 4.65, 7.96),
    (4.65, 7.91, 4.70, 7.96),
    (4.70, 7.91, 4.75, 7.96),
    (4.75, 7.91, 4.80, 7.96),
    (4.80, 7.91, 4.85, 7.96),
    (4.85, 7.91, 4.90, 7.96),
    (4.90, 7.91, 4.95, 7.96),
    (4.95, 7.91, 5.00, 7.96),
    (4.45, 7.96, 4.50, 8.01),
    (4.50, 7.96, 4.55, 8.01),
    (4.55, 7.96, 4.60, 8.01),
    (4.60, 7.96, 4.65, 8.01),

    # ── Ilesha east (clusters 1337–1338) ─────────────────────
    (4.71, 7.62, 4.76, 7.67),
    (4.76, 7.62, 4.81, 7.67),

    # ── Ede / Osogbo south (clusters 1340–1341) ──────────────
    (4.14, 7.60, 4.19, 7.65),
    (4.19, 7.60, 4.24, 7.65),

    # ── Iwo area (clusters 1342–1343) ────────────────────────
    (4.17, 7.36, 4.22, 7.41),
    (4.22, 7.36, 4.27, 7.41),
    (4.17, 7.41, 4.22, 7.46),
    (4.22, 7.41, 4.27, 7.46),

    # ── Ilesha / Ijebu fringe (clusters 1345–1348) ───────────
    (4.45, 7.48, 4.50, 7.53),
    (4.50, 7.48, 4.55, 7.53),
    (4.55, 7.48, 4.60, 7.53),
    (4.45, 7.53, 4.50, 7.58),
    (4.50, 7.53, 4.55, 7.58),
    (4.55, 7.53, 4.60, 7.58),
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
                    date = meta.get("date")
                    year = date.split("-")[0] if date else None
                    results_by_pano[pano_id] = [
                        {
                            "image_id": f"{pano_id}_{heading}",
                            "source": "gsv",
                            "date": date,
                            "lat": lat,
                            "lon": lon,
                            "heading": heading,
                            "image_url": build_image_url(lat, lon, heading, API_KEY),
                            "year": year,
                        }
                        for heading in HEADINGS
                    ]

            if completed % 500 == 0 or completed == total:
                pct = len(results_by_pano) / completed * 100
                print(f"  {completed:,}/{total:,} checked | Unique locations: {len(results_by_pano):,} ({pct:.1f}% yield)")

    all_rows = [row for rows in results_by_pano.values() for row in rows]
    n_locations = len(results_by_pano)
    print(f"\nWriting {len(all_rows):,} rows ({n_locations:,} locations × {len(HEADINGS)} headings) to {OUTPUT_CSV}...")

    fieldnames = ["image_id", "source", "date", "lat", "lon", "heading", "image_url", "year"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_panos = len(all_rows)
    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Unique locations with GSV coverage : {n_locations:,}")
    print(f"  Total rows (4 headings each)        : {total_panos:,}")
    print(f"  Estimated download cost             : ${total_panos * 0.007:.2f}")
    print(f"  CSV saved to                        : {os.path.abspath(OUTPUT_CSV)}")
    print(f"─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    run_pipeline()