import pandas as pd
import random
from geopy.distance import geodesic

images = pd.read_csv("kwale_merged_metadata.csv")
#images = images[images['source'] == 'gsv']
dhs = pd.read_csv("dhs_kwale_clusters.csv")
RADIUS_KM = 2
MIN_IMAGES = 99
MAX_IMAGES = 250

def cap_images(image_ids_str, max_n, seed=42):
    ids = image_ids_str.split(",")
    if len(ids) <= max_n:
        return image_ids_str
    random.seed(seed)
    sampled = random.sample(ids, max_n)
    return ",".join(sampled)

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
print(f"\n--- Coverage Summary (before filtering) ---")
print(f"Clusters with 0 images:    {len(matched_df[matched_df.image_count == 0])}")
print(f"Clusters with <{MIN_IMAGES} images:  {len(matched_df[matched_df.image_count < MIN_IMAGES])}")
print(f"Clusters with >={MIN_IMAGES} images: {len(matched_df[matched_df.image_count >= MIN_IMAGES])}")
print(f"\nImage count distribution:")
print(matched_df.image_count.describe())

# Filter clusters below minimum and cap image IDs at maximum
filtered_df = matched_df[matched_df.image_count >= MIN_IMAGES].copy()
filtered_df["image_ids"] = filtered_df["image_ids"].apply(lambda x: cap_images(x, MAX_IMAGES))
filtered_df["image_count"] = filtered_df["image_ids"].apply(lambda x: len(x.split(",")))

print(f"\n--- After Filtering & Capping ---")
print(f"Clusters remaining: {len(filtered_df)}")
print(filtered_df.image_count.describe())

filtered_df.to_csv("kwale_clusters_gsvmatched.csv", index=False)