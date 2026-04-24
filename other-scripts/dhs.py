import pandas as pd
import geopandas as gpd

# GPS: DHSCLUST, LATNUM, LONGNUM

# hv001 - Cluster ID
# hv270 - Wealth index (middle, richer, poorer, richest, poorest)
mapping = {
    "poorest": 1,
    "poorer": 2,
    "middle": 3,
    "richer": 4,
    "richest": 5
}

hr = pd.read_stata("data/raw/labels/NGHR7BFL.DTA")
data = hr[["hv001", "hv270"]].dropna()
data = data.rename(columns={"hv001": "cluster_id", "hv270": "wealth"})

data["wealth_num"] = data["wealth"].cat.codes + 1
print(data["wealth"].cat.categories)

cluster_data = data.groupby("cluster_id")["wealth_num"].mean().reset_index()


gps = gpd.read_file("data/raw/labels/NGGE7BFL - Geo data/NGGE7BFL.shp")

merged = cluster_data.merge(
    gps,
    left_on="cluster_id",
    right_on="DHSCLUST"
)
final_data = merged[["cluster_id", "wealth_num", "LATNUM", "LONGNUM"]]
print(final_data.shape)
print(final_data.isnull().sum())
final_data.to_csv("dhs_clusters.csv", index=False)