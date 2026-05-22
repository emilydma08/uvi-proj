import pandas as pd
import random

matched_df=pd.read_csv('clusters_matched_all.csv')

def cap_images(image_ids_str, max_n, seed=42):
    ids = image_ids_str.split(",")
    if len(ids) <= max_n:
        return image_ids_str
    random.seed(seed)
    sampled = random.sample(ids, max_n)
    return ",".join(sampled)

MIN_IMAGES = 100
MAX_IMAGES = 250

filtered_df = matched_df[matched_df.image_count >= MIN_IMAGES].copy()
filtered_df["image_ids"] = filtered_df["image_ids"].apply(lambda x: cap_images(x, MAX_IMAGES))
filtered_df["image_count"] = filtered_df["image_ids"].apply(lambda x: len(x.split(",")))

print(f"Clusters remaining: {len(filtered_df)}")
print(filtered_df.image_count.describe())

filtered_df.to_csv("clusters_matched_filtered.csv", index=False)