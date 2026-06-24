import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIRS = [
    "data/raw/images/abuja",
    "data/raw/images/akwa",
    "data/raw/images/benue",
    "data/raw/images/ebonyi",
    "data/raw/images/edo",
    "data/raw/images/enugu",
    "data/raw/images/kaduna",
    "data/raw/images/kano",
    "data/raw/images/lagos/gsv",
    "data/raw/images/more",
    "data/raw/images/niger",
    "data/raw/images/oyo",
    "data/raw/images/plateau",
    "data/raw/images/rivers",
    "data/raw/images/sokoto"
]
MATCHED_CSV = "data/processed/clusters_cities_pt3.csv"
OUTPUT_DIR = "features_clip_citiespt3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval().to(device)

print("CLIP model loaded")

# Sanity check
test_tensor = torch.randn(1, 3, 224, 224).to(device)
print(f"Model ready on {device}")

# ── Image finder ──────────────────────────────────────────────────────────────
def find_image(image_id):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            # Use vision model directly instead of get_image_features
            outputs = model.vision_model(pixel_values=pixel_values)
            # pooler_output is the [CLS] token — (1, 768)
            features = outputs.pooler_output
        return features.squeeze().cpu().numpy()  # (768,)
    except Exception as e:
        print(f"Failed on {image_path}: {e}")
        return None

# ── Per-cluster aggregation ───────────────────────────────────────────────────
df = pd.read_csv(MATCHED_CSV)

cluster_features = []
cluster_labels = []
cluster_ids = []
skipped_clusters = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Clusters"):
    image_ids = row["image_ids"].split(",")
    wealth = row["wealth_num"]
    cluster_id = row["cluster_id"]

    vectors = []
    for image_id in image_ids:
        image_path = find_image(image_id.strip())
        if image_path is None:
            continue
        feat = extract_features(image_path)
        if feat is not None:
            vectors.append(feat)

    if len(vectors) < 10:
        print(f"Cluster {cluster_id}: only {len(vectors)} images loaded, skipping")
        skipped_clusters.append(cluster_id)
        continue

    cluster_vector = np.mean(vectors, axis=0)  # (768,)
    cluster_features.append(cluster_vector)
    cluster_labels.append(wealth)
    cluster_ids.append(cluster_id)

    print(f"Cluster {cluster_id}: {len(vectors)} images → {cluster_vector.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
X = np.array(cluster_features)
y = np.array(cluster_labels)

np.save(os.path.join(OUTPUT_DIR, "X_clusters.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)
pd.DataFrame({"cluster_id": cluster_ids}).to_csv(
    os.path.join(OUTPUT_DIR, "cluster_ids.csv"), index=False
)

print(f"\n--- Done ---")
print(f"Feature matrix shape: {X.shape}")  # should be (225, 768)
print(f"Labels shape: {y.shape}")
print(f"Skipped clusters: {skipped_clusters}")