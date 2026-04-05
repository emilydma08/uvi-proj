"""
Downloads Google Street View images for each pano_id in the GSV metadata CSV.
Uses pano_id-based API queries.
Safe to re-run: skips rows already marked downloaded=true.
"""

import os
import io
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
BASE_URL   = "https://maps.googleapis.com/maps/api/streetview"
IMG_SIZE   = "640x640"
FOV        = 90
PITCH      = 0
GRAY_STD_THRESHOLD = 8
MAX_WORKERS = 8   # concurrent HTTP requests
BATCH_SAVE  = 50  # save CSV every N completed panos


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

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS * 2,
)
session.mount("https://", adapter)

csv_lock    = threading.Lock()
log_lock    = threading.Lock()
print_lock  = threading.Lock()


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


def download_image(pano_id: str, heading: int) -> tuple[str, int, bool, str | None]:
    """
    Download a single (pano_id, heading) image.
    Returns (pano_id, heading, success, error_msg).
    """
    url      = build_url(pano_id, heading)
    out_path = IMG_DIR / f"{pano_id}_{heading}.jpg"

    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return pano_id, heading, False, str(e)

    if is_blank_image(resp.content):
        return pano_id, heading, False, "blank image"

    out_path.write_bytes(resp.content)
    return pano_id, heading, True, None


total_rows   = len(df)
already_done = int(df["downloaded"].sum())
to_process   = total_rows - already_done
print(f"Total panos: {total_rows} | Already downloaded: {already_done} | Remaining: {to_process}\n")

tasks = []
for idx, row in df.iterrows():
    if row["downloaded"]:
        continue
    pano_id  = row["pano_id"]
    headings = [int(h.strip()) for h in str(row["headings"]).split(",")]
    for heading in headings:
        tasks.append((idx, pano_id, heading))

results: dict[int, dict[int, bool]] = {}
pano_ids: dict[int, str] = {
    idx: row["pano_id"]
    for idx, row in df.iterrows()
    if not row["downloaded"]
}

downloaded_count = 0
failed_count     = 0
pano_done_count  = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(download_image, pano_id, heading): (idx, pano_id, heading)
        for idx, pano_id, heading in tasks
    }

    for future in as_completed(future_map):
        idx, pano_id, heading = future_map[future]
        _, _, success, err = future.result()

        results.setdefault(idx, {})
        results[idx][heading] = success

        if not success:
            with print_lock:
                label = "ERROR" if err != "blank image" else "BLANK"
                print(f"  [{label}] {pano_id} h={heading}: {err or ''}")

        expected = {h for _, pid, h in tasks if _ == idx}
        if expected == results[idx].keys():
            pano_done_count += 1
            pano_failed = not all(results[idx].values())

            if pano_failed:
                failed_count += 1
                with log_lock:
                    with open(FAILED_LOG, "a") as f:
                        f.write(pano_id + "\n")
            else:
                downloaded_count += 1
                with csv_lock:
                    df.at[idx, "downloaded"] = True

            done_so_far = already_done + downloaded_count + failed_count
            with print_lock:
                print(
                    f"  [{done_so_far}/{total_rows}] pano {pano_id} "
                    f"{'saved' if not pano_failed else 'failed'}"
                )

            if pano_done_count % BATCH_SAVE == 0:
                with csv_lock:
                    df.to_csv(CSV_PATH, index=False)

df.to_csv(CSV_PATH, index=False)

print(f"\n── Summary ──────────────────────────────")
print(f"  Downloaded : {downloaded_count}")
print(f"  Failed     : {failed_count}  (see {FAILED_LOG})")
print(f"  Skipped    : {already_done}")
print(f"─────────────────────────────────────────")
