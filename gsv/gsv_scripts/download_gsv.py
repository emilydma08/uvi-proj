"""
Downloads Google Street View images for each image_id listed in
abuja_clusters_gsvmatched.csv (column: image_ids), resolved to URLs via
abuja_merged_metadata.csv (column: image_id -> image_url).

Safe to re-run: skips image files that already exist on disk.
"""

import io
import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
CLUSTERS_CSV  = Path("data/metadata/akwa/akwa_clusters_gsvmatched.csv")
METADATA_CSV  = Path("data/metadata/akwa/akwa_merged_metadata.csv")
IMG_DIR       = Path("data/raw/images/akwa")
LOG_DIR       = Path("logs")
FAILED_LOG    = LOG_DIR / "failed_images.txt"

GRAY_STD_THRESHOLD = 8
MAX_WORKERS        = 10   # concurrent HTTP requests
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("GSV_KEY")          # still needed if URLs contain a placeholder key
if not API_KEY:
    raise SystemExit("Error: GSV_KEY environment variable is not set.")

IMG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load metadata: build image_id -> image_url lookup ────────────────────────
meta_df = pd.read_csv(METADATA_CSV)
required_meta_cols = {"image_id", "image_url"}
if not required_meta_cols.issubset(meta_df.columns):
    raise SystemExit(f"Metadata CSV must contain columns: {required_meta_cols}")

# Replace placeholder API key in URLs with the real one from the environment
meta_df["image_url"] = meta_df["image_url"].str.replace(
    r"key=[^&\"]+", f"key={API_KEY}", regex=True
)

url_lookup: dict[str, str] = dict(zip(meta_df["image_id"], meta_df["image_url"]))
print(f"Metadata loaded: {len(url_lookup):,} image URLs indexed.")

# ── Collect all image_ids to download from the clusters CSV ──────────────────
clusters_df = pd.read_csv(CLUSTERS_CSV)
if "image_ids" not in clusters_df.columns:
    raise SystemExit("Clusters CSV must contain an 'image_ids' column.")

all_image_ids: set[str] = set()
for cell in clusters_df["image_ids"].dropna():
    for img_id in str(cell).split(","):
        img_id = img_id.strip()
        if img_id:
            all_image_ids.add(img_id)

print(f"Unique image IDs in clusters: {len(all_image_ids):,}")

# ── Filter: skip already-downloaded, warn on missing metadata ─────────────────
tasks: list[tuple[str, str]] = []   # (image_id, url)
skipped_existing  = 0
skipped_no_url    = 0
missing_ids: list[str] = []

for img_id in sorted(all_image_ids):
    out_path = IMG_DIR / f"{img_id}.jpg"
    if out_path.exists():
        skipped_existing += 1
        continue
    url = url_lookup.get(img_id)
    if url is None:
        skipped_no_url += 1
        missing_ids.append(img_id)
        continue
    tasks.append((img_id, url))

if missing_ids:
    missing_log = LOG_DIR / "missing_in_metadata.txt"
    missing_log.write_text("\n".join(missing_ids))
    print(f"  WARNING: {skipped_no_url} image IDs not found in metadata "
          f"(logged to {missing_log})")

print(f"Already on disk : {skipped_existing:,}")
print(f"To download     : {len(tasks):,}\n")

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS,
    pool_maxsize=MAX_WORKERS * 2,
)
session.mount("https://", adapter)

log_lock   = threading.Lock()
print_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_blank_image(content: bytes) -> bool:
    """Return True if GSV returned a gray 'no imagery' placeholder."""
    try:
        img = Image.open(io.BytesIO(content)).convert("L")
        return float(np.array(img).std()) < GRAY_STD_THRESHOLD
    except Exception:
        return True


def download_image(image_id: str, url: str) -> tuple[str, bool, str | None]:
    """
    Download a single image by URL and save as {image_id}.jpg.
    Returns (image_id, success, error_msg).
    """
    out_path = IMG_DIR / f"{image_id}.jpg"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return image_id, False, str(e)

    if is_blank_image(resp.content):
        return image_id, False, "blank image"

    out_path.write_bytes(resp.content)
    return image_id, True, None


# ── Download ──────────────────────────────────────────────────────────────────
downloaded_count = 0
failed_count     = 0
total            = len(tasks)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(download_image, img_id, url): img_id
        for img_id, url in tasks
    }

    for i, future in enumerate(as_completed(future_map), 1):
        image_id, success, err = future.result()

        if success:
            downloaded_count += 1
        else:
            failed_count += 1
            with log_lock:
                with open(FAILED_LOG, "a") as f:
                    f.write(image_id + "\n")

        with print_lock:
            status = "saved" if success else f"FAILED ({err})"
            print(f"  [{i}/{total}] {image_id} — {status}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── Summary ──────────────────────────────")
print(f"  Downloaded     : {downloaded_count:,}")
print(f"  Failed         : {failed_count:,}  (see {FAILED_LOG})")
print(f"  Already on disk: {skipped_existing:,}")
print(f"  Missing in meta: {skipped_no_url:,}")
print(f"─────────────────────────────────────────")