import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_DIRS = [
    "data/raw/images/abuja",
    "data/raw/images/edo",
    "data/raw/images/enugu",
    "data/raw/images/kano",
    "data/raw/images/lagos/gsv",
    "data/raw/images/oyo",
    ]
MATCHED_CSV = "data/processed/clusters_all_cities.csv"
OUTPUT_DIR = "gsv-features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = models.resnet50(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC layer
model.eval().to(device)

# ── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor)
        return features.squeeze().cpu().numpy()  # (2048,)
    except Exception as e:
        print(f"Failed on {image_path}: {e}")
        return None

def find_image(image_id):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

# ── Per-cluster aggregation ───────────────────────────────────────────────────
"""test_path = "data/raw/images/gsv/HgVpcmJN08mLdVA_K1CyoA_0.jpg"
test_feat = extract_features(test_path)
print(f"Test feature shape: {test_feat.shape}")  # should print (2048,)"""

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
        image_path = find_image(image_id)
        if image_path is None:
            continue
        feat = extract_features(image_path)
        if feat is not None:
            vectors.append(feat)

    if len(vectors) < 10:  # skip cluster if too few images loaded successfully
        print(f"Cluster {cluster_id}: only {len(vectors)} images loaded, skipping")
        skipped_clusters.append(cluster_id)
        continue

    # Aggregate: mean pooling across all image feature vectors
    cluster_vector = np.mean(vectors, axis=0)  # (2048,)
    cluster_features.append(cluster_vector)
    cluster_labels.append(wealth)
    cluster_ids.append(cluster_id)

    print(f"Cluster {cluster_id}: {len(vectors)} images → feature vector shape {cluster_vector.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
X = np.array(cluster_features)  # (n_clusters, 2048)
y = np.array(cluster_labels)    # (n_clusters,)

np.save(os.path.join(OUTPUT_DIR, "X_clusters.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)
pd.DataFrame({"cluster_id": cluster_ids}).to_csv(os.path.join(OUTPUT_DIR, "cluster_ids.csv"), index=False)

print(f"\n--- Done ---")
print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Skipped clusters: {skipped_clusters}")