import requests
import csv
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()
mapillary_token = os.getenv("MAPILLARY_TOKEN")

TOKEN = mapillary_token

MAX_RETRIES = 6
RETRY_PASS_WAIT = 30
MAX_WORKERS = 5          
OUTPUT_CSV = "mapillary_lagos.csv"

HIGH_YIELD_TILES = [
    (3.15, 6.65, 3.2,  6.7 ),  # tile 12
    (3.2,  6.45, 3.25, 6.5 ),  # tile 14
    (3.2,  6.5,  3.25, 6.55),  # tile 15
    (3.25, 6.45, 3.3,  6.5 ),  # tile 20
    (3.3,  6.45, 3.35, 6.5 ),  # tile 26
    (3.3,  6.5,  3.35, 6.55),  # tile 27
    (3.35, 6.4,  3.4,  6.45),  # tile 31
    (3.4,  6.6,  3.45, 6.65),  # tile 41
    (3.5,  6.4,  3.55, 6.45),  # tile 49
    (3.5,  6.5,  3.55, 6.55),  # tile 51
    (3.5,  6.55, 3.55, 6.6 ),  # tile 52
    (3.5,  6.6,  3.55, 6.65),  # tile 53
    (3.5,  6.65, 3.55, 6.7 ),  # tile 54
]

LOW_YIELD_TILES = [
    (3.1,  6.45, 3.15, 6.5 ),  # tile 2  - 80 images
    (3.1,  6.5,  3.15, 6.55),  # tile 3  - 11 images
    (3.15, 6.45, 3.2,  6.5 ),  # tile 8  - 95 images
    (3.2,  6.55, 3.25, 6.6 ),  # tile 16 - 98 images
    (3.25, 6.55, 3.3,  6.6 ),  # tile 22 - 29 images
    (3.25, 6.6,  3.3,  6.65),  # tile 23 - 16 images
    (3.3,  6.55, 3.35, 6.6 ),  # tile 28 - 51 images
    (3.3,  6.6,  3.35, 6.65),  # tile 29 - 47 images
    (3.35, 6.45, 3.4,  6.5 ),  # tile 32 - 98 images
    (3.35, 6.5,  3.4,  6.55),  # tile 33 - 63 images
    (3.35, 6.55, 3.4,  6.6 ),  # tile 34 - 10 images
    (3.35, 6.6,  3.4,  6.65),  # tile 35 - 52 images
    (3.4,  6.4,  3.45, 6.45),  # tile 37 - 12 images
    (3.4,  6.45, 3.45, 6.5 ),  # tile 38 - 32 images
    (3.4,  6.5,  3.45, 6.55),  # tile 39 - 64 images
    (3.4,  6.55, 3.45, 6.6 ),  # tile 40 - 80 images
    (3.45, 6.4,  3.5,  6.45),  # tile 43 - 88 images
    (3.45, 6.45, 3.5,  6.5 ),  # tile 44 - 27 images
    (3.45, 6.55, 3.5,  6.6 ),  # tile 46 - 85 images
    (3.45, 6.6,  3.5,  6.65),  # tile 47 - 87 images
]

UNKNOWN_TILES = [
    (3.55, 6.4,  3.6,  6.45),  # tile 55
    (3.55, 6.45, 3.6,  6.5 ),  # tile 56
    (3.55, 6.5,  3.6,  6.55),  # tile 57
    (3.55, 6.55, 3.6,  6.6 ),  # tile 58
    (3.55, 6.6,  3.6,  6.65),  # tile 59
    (3.55, 6.65, 3.6,  6.7 ),  # tile 60
]

# Skipped entirely (returned 0 or consistently errored):
# Zeros: 1, 4, 5
# Errors: 6, 7, 9, 10, 11, 13, 17, 18, 19, 21, 24, 25, 30, 36, 42, 45, 48, 50


def fetch_tile_images(tile_bbox, token, limit):
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    url = "https://graph.mapillary.com/images"

    params = {
        "access_token": token,
        "fields": "id,captured_at,sequence_id,thumb_2048_url",
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "limit": limit,
    }

    all_images = []
    page = 1

    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            resp = requests.get(url, params=params)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(f"  [{tile_bbox[0]},{tile_bbox[1]}] Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if resp.status_code == 500:
                wait = 5 + (attempt * 3)
                print(f"  [{tile_bbox[0]},{tile_bbox[1]}] 500 error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s... | {resp.text[:200]}")
                time.sleep(wait)
                continue

            break

        resp.raise_for_status()
        data = resp.json()
        images = data.get("data", [])
        all_images.extend(images)
        print(f"  [{tile_bbox[0]},{tile_bbox[1]}] Page {page}: {len(images)} images")

        next_url = data.get("paging", {}).get("next")
        if not next_url:
            break

        url = next_url
        params = {"access_token": token}
        page += 1
        time.sleep(0.5)

    return all_images


def fetch_tile_task(tile, limit, token):
    images = fetch_tile_images(tile, token, limit)
    return tile, limit, images


def process_tiles_parallel(tiles_with_limits, token, writer, seen_ids, write_lock, label=""):
    """
    Returns (total_new, failed_tiles_with_limits).
    """
    total_new = 0
    failed = []
    completed = 0
    total = len(tiles_with_limits)

    print(f"  Submitting {total} tiles with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_tile_task, tile, limit, token): (tile, limit)
            for tile, limit in tiles_with_limits
        }

        for future in as_completed(futures):
            tile, limit = futures[future]
            completed += 1
            prefix = f"[{label}{completed}/{total}]"

            try:
                _, _, images = future.result()
            except requests.exceptions.HTTPError as e:
                print(f"{prefix} FAILED {tile[0]},{tile[1]} -> {tile[2]},{tile[3]}: {e}")
                failed.append((tile, limit))
                continue

            with write_lock:
                new_count = 0
                for img in images:
                    img_id = img["id"]
                    if img_id in seen_ids:
                        continue
                    seen_ids.add(img_id)
                    writer.writerow([
                        img_id,
                        img.get("captured_at"),
                        img.get("sequence_id"),
                        img.get("thumb_2048_url", ""),
                    ])
                    new_count += 1
                total_new += new_count

            dupes = len(images) - new_count
            print(f"{prefix} {tile[0]},{tile[1]} -> {tile[2]},{tile[3]} | New: {new_count} | Dupes: {dupes} | Running total: {total_new}")

    return total_new, failed


def run_pipeline():
    tiles_with_limits = (
        [(t, 500) for t in HIGH_YIELD_TILES] +
        [(t, 100) for t in LOW_YIELD_TILES] +
        [(t, 300) for t in UNKNOWN_TILES]
    )

    print(f"Total tiles to process: {len(tiles_with_limits)}")
    print(f"  High-yield (limit=300): {len(HIGH_YIELD_TILES)}")
    print(f"  Low-yield  (limit=100): {len(LOW_YIELD_TILES)}")
    print(f"  Unknown    (limit=300): {len(UNKNOWN_TILES)}")
    print(f"  Workers: {MAX_WORKERS}\n")

    seen_ids = set()
    total_images = 0
    write_lock = threading.Lock()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "captured_at", "sequence_id", "thumb_2048_url"])

        print("=== First pass ===")
        new, failed = process_tiles_parallel(tiles_with_limits, TOKEN, writer, seen_ids, write_lock)
        total_images += new

        if failed:
            print(f"\n=== Retry pass: {len(failed)} failed tiles. Waiting {RETRY_PASS_WAIT}s... ===\n")
            time.sleep(RETRY_PASS_WAIT)
            new, still_failed = process_tiles_parallel(failed, TOKEN, writer, seen_ids, write_lock, label="retry ")
            total_images += new

            if still_failed:
                print(f"\n=== {len(still_failed)} tiles failed permanently ===")
                for t, _ in still_failed:
                    print(f"  FAILED: {t}")

    print(f"\nDone. Total unique images: {total_images}")
    print(f"Saved to {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    run_pipeline()