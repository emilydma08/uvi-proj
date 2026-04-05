import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent

locations = pd.read_csv(data_dir / "mapillary_lagos_merged.csv")

locations = locations.drop(columns=["captured_at", "compass_angle", "sequence_id", "thumb_2048_url"])

locations = locations[['lat', 'lon', 'image_id']]

locations['color'] = '#FF0000'


out_path = data_dir / "mapillary_lagos_locations.csv"
locations.to_csv(out_path, index=False)
