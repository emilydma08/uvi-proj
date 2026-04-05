import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()
mapillary_token = os.getenv("MAPILLARY_TOKEN")

TOKEN = mapillary_token
INPUT_CSV = "mapillary_lagos_reruns.csv"
OUTPUT_CSV = "mapillary_lagos__reruns_coords.csv"
MAX_WORKERS = 20  

def fetch_coords(image_id, token):
    url = f"https://graph.mapillary.com/{image_id}"
    params = {
        "access_token": token,
        "fields": "id,geometry,captured_at,compass_angle"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    coords = data.get("geometry", {}).get("coordinates", [None, None])
    return {
        "image_id": image_id,
        "lon": coords[0],
        "lat": coords[1],
        "captured_at": data.get("captured_at"),
        "compass_angle": data.get("compass_angle"),
    }

def run():
    with open(INPUT_CSV, "r") as f:
        rows = list(csv.DictReader(f))

    image_ids = [r["image_id"] for r in rows]
    print(f"Fetching coordinates for {len(image_ids)} images...")

    results = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_coords, img_id, TOKEN): img_id for img_id in image_ids}
        for future in as_completed(futures):
            img_id = futures[future]
            completed += 1
            try:
                results[img_id] = future.result()
                if completed % 100 == 0:
                    print(f"  {completed}/{len(image_ids)} done...")
            except Exception as e:
                print(f"  FAILED {img_id}: {e}")
                results[img_id] = {"image_id": img_id, "lon": None, "lat": None, "captured_at": None, "compass_angle": None}

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "lat", "lon", "captured_at", "compass_angle", "sequence_id", "thumb_2048_url"])
        writer.writeheader()
        for row in rows:
            img_id = row["image_id"]
            enriched = results.get(img_id, {})
            writer.writerow({
                "image_id": img_id,
                "lat": enriched.get("lat"),
                "lon": enriched.get("lon"),
                "captured_at": enriched.get("captured_at"),
                "compass_angle": enriched.get("compass_angle"),
                "sequence_id": row.get("sequence_id"),
                "thumb_2048_url": row.get("thumb_2048_url"),
            })

    print(f"\nDone. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run()