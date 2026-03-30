import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent

locations = pd.read_csv(data_dir / "gsv_lagos_75M_15000L_0.001SD.csv")

locations = locations.drop(columns=["date", "copyright", "headings", "url_h0", "url_h90", "url_h180", "url_h270"])

locations['color'] = "#588CC7"


out_path = data_dir / "gsv_lagos_locations.csv"
locations.to_csv(out_path, index=False)
