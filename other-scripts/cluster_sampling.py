import pandas as pd
from geopy.distance import geodesic

images = pd.read_csv("data/metadata/lagos_merged_metadata.csv")
dhs = pd.read_csv("dhs_clusters.csv")
RADIUS_KM = 2  # Lagos is urban throughout

print(f"DHS clusters: {len(dhs)}")
print(f"Total images: {len(images)}")

results = []
for i, cluster in dhs.iterrows():
    matched_ids = []
    for _, img in images.iterrows():
        dist = geodesic(
            (cluster["LATNUM"], cluster["LONGNUM"]),
            (img["lat"], img["lon"])
        ).km
        if dist <= RADIUS_KM:
            matched_ids.append(img["image_id"])

    results.append({
        "cluster_id": cluster["cluster_id"],
        "wealth_num": cluster["wealth_num"],
        "lat": cluster["LATNUM"],
        "lon": cluster["LONGNUM"],
        "image_count": len(matched_ids),
        "image_ids": ",".join(str(i) for i in matched_ids)
    })

    print(f"[{i+1}/{len(dhs)}] Cluster {cluster['cluster_id']}: {len(matched_ids)} images")

matched_df = pd.DataFrame(results)

# Coverage stats before filtering
print(f"\n--- Coverage Summary ---")
print(f"Clusters with 0 images:   {len(matched_df[matched_df.image_count == 0])}")
print(f"Clusters with <5 images:  {len(matched_df[matched_df.image_count < 5])}")
print(f"Clusters with >=5 images: {len(matched_df[matched_df.image_count >= 5])}")
print(f"\nImage count distribution:")
print(matched_df.image_count.describe())

matched_df.to_csv("clusters_matched_all.csv", index=False)