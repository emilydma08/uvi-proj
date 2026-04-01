"""
Downloads Google Street View images for each pano_id in the GSV metadata CSV.
Uses pano_id-based API queries
Safe to re-run: skips rows already marked downloaded=true.
"""

import os
import time
import io
from pathlib import Path

import requests
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import numpy as np


# Config 
CSV_PATH   = Path("gsv_data/gsv_lagos_75M_15000L_0.001SD.csv")
IMG_DIR    = Path("raw/images")
LOG_DIR    = Path("logs")
FAILED_LOG = LOG_DIR / "failed_panos.txt"
DELAY      = 0.5  
BASE_URL   = "https://maps.googleapis.com/maps/api/streetview"
IMG_SIZE   = "640x640"
FOV        = 90
PITCH      = 0
GRAY_STD_THRESHOLD = 8


load_dotenv()
API_KEY = os.getenv("GSV_KEY")
if not API_KEY:
    raise SystemExit("Error: GSV_KEY environment variable is not set.")

IMG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
if "downloaded" not in df.columns:
    df["downloaded"] = False

df["downloaded"] = df["downloaded"].astype(str).str.lower().isin(["true", "1", "yes"])


def is_blank_image(content: bytes) -> bool:
    """Return True if GSV returned a gray 'no imagery' placeholder."""
    try:
        img = Image.open(io.BytesIO(content)).convert("L") 
        return float(np.array(img).std()) < GRAY_STD_THRESHOLD
    except Exception:
        return True  


def build_url(pano_id: str, heading: int) -> str:
    return (
        f"{BASE_URL}?size={IMG_SIZE}&pano={pano_id}"
        f"&fov={FOV}&heading={heading}&pitch={PITCH}&key={API_KEY}"
    )


total_rows   = len(df)
already_done = df["downloaded"].sum()
to_process   = total_rows - already_done
print(f"Total panos: {total_rows} | Already downloaded: {already_done} | Remaining: {to_process}\n")

downloaded_count = 0
failed_count     = 0

for idx, row in df.iterrows():
    if row["downloaded"]:
        continue

    pano_id  = row["pano_id"]
    headings = [int(h.strip()) for h in str(row["headings"]).split(",")]
    pano_failed = False

    for heading in headings:
        url      = build_url(pano_id, heading)
        out_path = IMG_DIR / f"{pano_id}_{heading}.jpg"

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ERROR] {pano_id} h={heading}: {e}")
            pano_failed = True
            time.sleep(DELAY)
            continue

        if is_blank_image(resp.content):
            print(f"  [BLANK] {pano_id} h={heading}")
            pano_failed = True
            time.sleep(DELAY)
            continue

        out_path.write_bytes(resp.content)
        time.sleep(DELAY)

    if pano_failed:
        failed_count += 1
        with open(FAILED_LOG, "a") as f:
            f.write(pano_id + "\n")
    else:
        df.at[idx, "downloaded"] = True
        downloaded_count += 1

    done_so_far = already_done + downloaded_count + failed_count
    print(
        f"  [{done_so_far}/{total_rows}] pano {pano_id} "
        f"{'✓ saved' if not pano_failed else '✗ failed'}"
    )

    df.to_csv(CSV_PATH, index=False)

print(f"\n── Summary ──────────────────────────────")
print(f"  Downloaded : {downloaded_count}")
print(f"  Failed     : {failed_count}  (see {FAILED_LOG})")
print(f"  Skipped    : {already_done}")
print(f"─────────────────────────────────────────")
