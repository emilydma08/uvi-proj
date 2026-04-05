import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent

reruns = pd.read_csv(data_dir / "mapillary_lagos__reruns_coords.csv")
with_coords = pd.read_csv(data_dir / "mapillary_lagos_with_coords.csv")

merged = pd.concat([reruns, with_coords], ignore_index=True)
merged.drop_duplicates(subset="image_id", inplace=True)

# captured_at is Unix time in milliseconds
merged["captured_at"] = pd.to_datetime(merged["captured_at"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")

out_path = data_dir / "mapillary_lagos_merged.csv"
merged.to_csv(out_path, index=False)
print(f"Merged {len(reruns)} + {len(with_coords)} rows → {len(merged)} unique rows")
print(f"Saved to {out_path}")
