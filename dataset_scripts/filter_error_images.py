"""
Scans all GSV images for Google Maps "no imagery" error images (mostly white),
removes matching image IDs from clusters_all_cities.csv, and reports results.
"""

import os
import glob
import csv
from pathlib import Path
from PIL import Image
import numpy as np

IMAGE_DIR = Path("data/raw/images")
CSV_PATH = Path("data/processed/clusters_all_cities.csv")
WHITE_THRESHOLD = 0.80  # flag if >90% of pixels are near-white
WHITE_PIXEL_VALUE = 240  # pixel channel value >= this counts as white

def is_error_image(image_path: Path) -> bool:
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        white_pixels = np.all(arr >= WHITE_PIXEL_VALUE, axis=2).sum()
        ratio = white_pixels / arr.shape[0] / arr.shape[1]
        return ratio > WHITE_THRESHOLD
    except Exception as e:
        print(f"  Warning: could not read {image_path}: {e}")
        return False

def find_error_image_ids() -> set:
    error_ids = set()
    image_paths = list(IMAGE_DIR.glob("*/*.jpg"))
    print(f"Scanning {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(image_paths)} checked, {len(error_ids)} errors so far")
        if is_error_image(path):
            error_ids.add(path.stem)  # stem = filename without .jpg

    print(f"Found {len(error_ids)} error images out of {len(image_paths)}")
    return error_ids

def filter_csv(error_ids: set):
    rows = []
    total_removed = 0

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            ids = [id_.strip() for id_ in row["image_ids"].split(",") if id_.strip()]
            clean_ids = [id_ for id_ in ids if id_ not in error_ids]
            removed = len(ids) - len(clean_ids)
            total_removed += removed
            row["image_ids"] = ",".join(clean_ids)
            row["image_count"] = len(clean_ids)
            rows.append(row)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Removed {total_removed} error image IDs from {CSV_PATH}")

def main():
    error_ids = find_error_image_ids()

    if error_ids:
        print("\nError image IDs:")
        for id_ in sorted(error_ids):
            print(f"  {id_}")

        print(f"\nUpdating {CSV_PATH}...")
        filter_csv(error_ids)
    else:
        print("No error images found — CSV unchanged.")

if __name__ == "__main__":
    main()
