import pandas as pd
import geopandas as gpd

# GPS: DHSCLUST, LATNUM, LONGNUM

# hv001 - Cluster ID
# hv025 - Urban vs. rural, maybe filter to only urban? or ill see when i match images
# hv024 - Region
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
#print(gps.columns.tolist())
#print(gps["ADM1NAME"].unique())


merged = cluster_data.merge(
    gps,
    left_on="cluster_id",
    right_on="DHSCLUST"
)
print(merged["ADM1NAME"].unique())

"""
lagos_clusters = merged[merged["ADM1NAME"] == "LAGOS"]

final_data = lagos_clusters[["cluster_id", "wealth_num", "LATNUM", "LONGNUM"]]
print(final_data.shape)
print(final_data.isnull().sum())
print(final_data["wealth_num"].describe())
print(final_data["wealth_num"].value_counts(bins=5))
final_data.to_csv("dhs_clusters.csv", index=False)
"""