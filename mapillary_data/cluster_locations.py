"""
Clusters mapillary_lagos_locations.csv by rounding lat/lon to 3 decimal places
(~110m grid cells) and averaging coordinates within each cell.
Outputs mapillary_lagos_locations.csv in-place with ~900 rows.
"""
import csv
from collections import defaultdict
from pathlib import Path

INPUT = Path(__file__).parent / "mapillary_lagos_locations.csv"
OUTPUT = Path(__file__).parent / "mapillary_lagos_locations_clustered.csv"

cells = defaultdict(list) 

with open(INPUT) as f:
    reader = csv.DictReader(f)
    for row in reader:
        lat = float(row["lat"])
        lon = float(row["lon"])
        key = (round(lat, 3), round(lon, 3))
        cells[key].append((lat, lon))

rows_out = []
for (_, _), points in sorted(cells.items()):
    avg_lat = sum(p[0] for p in points) / len(points)
    avg_lon = sum(p[1] for p in points) / len(points)
    rows_out.append({"lat": avg_lat, "lon": avg_lon, "image_id": "", "color": "#FF0000"})

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lat", "lon", "image_id", "color"])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Done: {len(rows_out)} rows written to {OUTPUT}")
