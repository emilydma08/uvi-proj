import requests
import numpy as np
import csv
import time
import os
from dotenv import load_dotenv
import os

load_dotenv()
mapillary_token = os.getenv("MAPILLARY_TOKEN")

TOKEN = mapillary_token

MIN_LON, MIN_LAT = 3.1, 6.4
MAX_LON, MAX_LAT = 3.6, 6.7

TILE_SIZE_LON = 0.05
TILE_SIZE_LAT = 0.05

LIMIT = 100
MAX_RETRIES = 6
RETRY_PASS_WAIT = 30 

OUTPUT_CSV = "mapillary_lagos.csv"


def generate_tiles(min_lon, min_lat, max_lon, max_lat, tile_lon, tile_lat):
    """Split the bounding box into tiles that respect the 0.01 sq degree API limit."""
    lons = np.arange(min_lon, max_lon, tile_lon)
    lats = np.arange(min_lat, max_lat, tile_lat)
    tiles = []
    for lon in lons:
        for lat in lats:
            tiles.append((
                round(lon, 6),
                round(lat, 6),
                round(min(lon + tile_lon, max_lon), 6),
                round(min(lat + tile_lat, max_lat), 6),
            ))
    return tiles


def fetch_tile_images(tile_bbox, token, limit=100):
    """
    Fetch all images for a single tile with pagination and exponential backoff.
    Auth is passed as access_token query param (correct for Mapillary Graph API v4).
    """
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    url = "https://graph.mapillary.com/images"

    params = {
        "access_token": token,
        "fields": "id,captured_at,sequence_id",
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "limit": limit,
    }

    all_images = []
    page = 1

    while True:
        resp = None
        for attempt in range(1, MAX_RETRIES + 1):
            resp = requests.get(url, params=params)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(f"    Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if resp.status_code == 500:
                print(f"    500 error body: {resp.text[:300]}")
                wait = 5 + (attempt * 3) + (time.time() % 2)  
                print(f"    500 error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue

            break  

        resp.raise_for_status()
        data = resp.json()
        images = data.get("data", [])
        all_images.extend(images)
        print(f"    Page {page}: got {len(images)} images")

        next_url = data.get("paging", {}).get("next")
        if not next_url:
            break

        url = next_url
        params = {"access_token": token}
        page += 1
        time.sleep(0.5)

    return all_images


def write_images(writer, images, seen_ids, token):
    """Deduplicate and write images to CSV. Returns count of new images written."""
    new_count = 0
    for img in images:
        img_id = img["id"]
        if img_id in seen_ids:
            continue
        seen_ids.add(img_id)
        thumb_url = f"https://graph.mapillary.com/{img_id}/thumb-2048?access_token={token}"
        writer.writerow([
            img_id,
            img.get("captured_at"),
            img.get("sequence_id"),
            thumb_url,
        ])
        new_count += 1
    return new_count


def process_tiles(tiles, token, writer, seen_ids, label=""):
    total_new = 0
    failed_tiles = []

    for idx, tile in enumerate(tiles, start=1):
        prefix = f"[{label}{idx}/{len(tiles)}]" if label else f"[{idx}/{len(tiles)}]"
        print(f"{prefix} Tile: {tile[0]},{tile[1]} -> {tile[2]},{tile[3]}")
        print(f"  Tile area: {(tile[2]-tile[0]) * (tile[3]-tile[1]):.6f} sq deg")


        try:
            images = fetch_tile_images(tile, token, LIMIT)
        except requests.exceptions.HTTPError as e:
            print(f"    ERROR: {e} — queuing for retry")
            failed_tiles.append(tile)
            continue

        new_count = write_images(writer, images, seen_ids, token)
        dupes = len(images) - new_count
        total_new += new_count
        print(f"    New: {new_count} | Dupes skipped: {dupes} | Running total: {sum([total_new])}")

        time.sleep(0.5)

    return total_new, failed_tiles


def run_pipeline():
    tiles = generate_tiles(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, TILE_SIZE_LON, TILE_SIZE_LAT)
    print(f"Lagos bbox: ({MIN_LON},{MIN_LAT}) -> ({MAX_LON},{MAX_LAT})")
    print(f"Tile size: {TILE_SIZE_LON} x {TILE_SIZE_LAT} = {TILE_SIZE_LON * TILE_SIZE_LAT:.4f} sq deg")
    print(f"Total tiles: {len(tiles)}\n")

    seen_ids = set()
    total_images = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "captured_at", "sequence_id", "thumb_url"])

        print("=== First pass ===")
        new, failed_tiles = process_tiles(tiles, TOKEN, writer, seen_ids)
        total_images += new

        if failed_tiles:
            print(f"\n=== Retry pass: {len(failed_tiles)} failed tiles. Waiting {RETRY_PASS_WAIT}s... ===\n")
            time.sleep(RETRY_PASS_WAIT)
            new, still_failed = process_tiles(failed_tiles, TOKEN, writer, seen_ids, label="retry ")
            total_images += new

            if still_failed:
                print(f"\n=== Second retry pass: {len(still_failed)} tiles. Waiting {RETRY_PASS_WAIT}s... ===\n")
                time.sleep(RETRY_PASS_WAIT)
                new, permanent_failures = process_tiles(still_failed, TOKEN, writer, seen_ids, label="retry2 ")
                total_images += new

    print(f"\nDone. Total unique images: {total_images}")
    print(f"Saved to {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    run_pipeline()