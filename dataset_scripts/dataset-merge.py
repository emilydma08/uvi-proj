import pandas as pd

gsv = pd.read_csv('gsv_enugu_metadata.csv')
#mapillary = pd.read_csv('mapillary/mapillary_data/mapillary_lagos_merged.csv')
clusters = pd.read_csv('dhs_enugu_clusters.csv')

gsv_rows = []

for _, row in gsv.iterrows():
    for heading in [0, 90, 180, 270]:
        gsv_rows.append({
            "image_id": f"{row['pano_id']}_{heading}",
            "source": "gsv",
            "date": row['date'],
            "lat": row["lat"],
            "lon": row["lon"],
            "heading": heading,
            "image_url": row[f"url_h{heading}"]
        })

gsv_df = pd.DataFrame(gsv_rows)
"""
mapillary_df = mapillary.rename(columns={
    "thumb_2048_url": "image_url",
    "compass_angle": "heading",
})

mapillary_df["source"] = "mapillary"

mapillary_df['date'] = pd.to_datetime(
    mapillary_df["captured_at"],
    errors="coerce"
).dt.strftime("%Y-%m")

mapillary_df = mapillary_df[[
    "image_id",
    "source",
    "date",
    "lat",
    "lon",
    "heading",
    "image_url"
]]
"""

all_images = pd.concat([gsv_df], ignore_index=True)
all_images["year"] = pd.to_datetime(all_images["date"]).dt.year
all_images.to_csv('enugu_merged_metadata.csv', index=False)