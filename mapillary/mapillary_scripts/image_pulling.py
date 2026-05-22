import requests
import pandas as pd
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

df = pd.read_csv("mapillary/mapillary_data/mapillary_lagos_merged.csv")

ACCESS_TOKEN = os.getenv("MAPILLARY_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("MAPILLARY_TOKEN environment variable not set")

output_dir = "mapillary/downloaded_images"
os.makedirs(output_dir, exist_ok=True)

def fetch_and_save_image(args):
    i, total, image_id = args
    image_id = str(image_id)

    out_path = os.path.join(output_dir, f"{image_id}.jpg")
    if os.path.exists(out_path):
        print(f"[{i}/{total}] [{image_id}] Already exists, skipping")
        return "skipped"

    try:
        # Step 1: Get CDN URL
        r = requests.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"access_token": ACCESS_TOKEN, "fields": "thumb_2048_url"},
            timeout=15
        )

        if r.status_code != 200:
            print(f"[{i}/{total}] [{image_id}] Metadata failed: {r.status_code}")
            return "failed"

        img_url = r.json().get("thumb_2048_url")
        if not img_url:
            print(f"[{i}/{total}] [{image_id}] No URL returned")
            return "failed"

        # Step 2: Download image
        img_response = requests.get(img_url, stream=True, timeout=30)
        if img_response.status_code != 200:
            print(f"[{i}/{total}] [{image_id}] Download failed: {img_response.status_code}")
            return "failed"

        # Step 3: Save
        with open(out_path, "wb") as f:
            for chunk in img_response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[{i}/{total}] [{image_id}] Saved")
        return "success"

    except requests.exceptions.Timeout:
        print(f"[{i}/{total}] [{image_id}] Timed out")
        return "failed"
    except Exception as e:
        print(f"[{i}/{total}] [{image_id}] Error: {e}")
        return "failed"


image_ids = df.iloc[:, 0].tolist()
total = len(image_ids)
args_list = [(i, total, image_id) for i, image_id in enumerate(image_ids, 1)]

print(f"Downloading {total} images with 10 threads...")

results = {"success": 0, "skipped": 0, "failed": 0}

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_and_save_image, args): args for args in args_list}
    for future in as_completed(futures):
        outcome = future.result()
        results[outcome] += 1

print(f"\nDone! {results['success']} saved, {results['skipped']} skipped, {results['failed']} failed")