import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIRS = [
    "data/raw/images/abuja",
    "data/raw/images/edo",
    "data/raw/images/enugu",
    "data/raw/images/kano",
    "data/raw/images/lagos/gsv",
    "data/raw/images/oyo",
    "data/raw/images/more",
]
MATCHED_CSV = "data/processed/clusters_all_cities.csv"
OUTPUT_DIR = "finetuned_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 30
LR = 3e-5                  # was 1e-4 — smaller steps, less overshoot per update
WEIGHT_DECAY = 1e-3
ACCUM_STEPS = 16           # gradient accumulation: effective batch = 16 clusters
GRAD_CLIP = 1.0            # max gradient norm (caps any single bad window)
N_FOLDS = 5

# Optional: standardize wealth_num to z-scores using TRAIN-FOLD stats only.
# Off by default so numbers stay comparable to your current run. Flip to True
# to test — metrics are still reported in the ORIGINAL units either way.
STANDARDIZE_TARGET = False

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Image transforms ──────────────────────────────────────────────────────────
# Decode + resize + crop ONCE, store as uint8 (~5 GB for 35k @ 224²).
# Cheap float-convert + normalize runs per-access. Valid because the transform
# is deterministic (no random augmentation) — cached tensor is identical each epoch.
cache_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),          # uint8 (3, 224, 224)
])

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def to_model_input(uint8_batch):
    return _normalize(uint8_batch.float() / 255.0)

def find_image(image_id):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

# ── Build image cache (one-time) ──────────────────────────────────────────────
df = pd.read_csv(MATCHED_CSV)

seen = set()
unique_ids = []
for ids in df["image_ids"]:
    for image_id in str(ids).split(","):
        image_id = image_id.strip()
        if image_id and image_id not in seen:
            seen.add(image_id)
            unique_ids.append(image_id)

print(f"Caching {len(unique_ids)} unique images (one-time decode)...")
image_cache = {}
missing = 0
t0 = time.time()
for n, image_id in enumerate(unique_ids, 1):
    path = find_image(image_id)
    if path is None:
        missing += 1
        continue
    try:
        image_cache[image_id] = cache_transform(Image.open(path).convert("RGB"))
    except Exception:
        missing += 1
    if n % 2000 == 0:
        print(f"  {n}/{len(unique_ids)} cached  ({time.time() - t0:.0f}s)")

cache_gb = len(image_cache) * 3 * 224 * 224 / 1e9
print(f"Cached {len(image_cache)} images, {missing} missing/failed, "
      f"{time.time() - t0:.0f}s, ~{cache_gb:.1f} GB in RAM")

def cluster_batch(image_ids_str):
    """Return normalized (N, 3, 224, 224) float tensor for a cluster, or None."""
    tensors = [image_cache[i.strip()] for i in str(image_ids_str).split(",")
               if i.strip() in image_cache]
    if not tensors:
        return None
    return to_model_input(torch.stack(tensors))

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    return model.to(device)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, dataset_df, optimizer, criterion, y_mean=0.0, y_std=1.0):
    model.train()
    total_loss = 0.0
    n_used = 0

    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad()
    accum_count = 0   # clusters accumulated since last optimizer step

    for _, row in dataset_df.iterrows():
        imgs = cluster_batch(row["image_ids"])
        if imgs is None:
            continue
        imgs = imgs.to(device)
        target = (row["wealth_num"] - y_mean) / y_std
        wealth = torch.tensor(target, dtype=torch.float32).to(device)

        preds = model(imgs).squeeze()      # (N,)
        cluster_pred = preds.mean()        # scalar
        # divide by ACCUM_STEPS so the accumulated gradient is the MEAN over the
        # window (what a true batch of this size would produce), not the sum.
        loss = criterion(cluster_pred, wealth) / ACCUM_STEPS
        loss.backward()

        total_loss += loss.item() * ACCUM_STEPS   # undo scaling for logging
        n_used += 1
        accum_count += 1

        if accum_count == ACCUM_STEPS:
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    # flush leftover partial window at end of epoch
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_used, 1)

def evaluate(model, dataset_df, y_mean=0.0, y_std=1.0):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for _, row in dataset_df.iterrows():
            imgs = cluster_batch(row["image_ids"])
            if imgs is None:
                continue
            imgs = imgs.to(device)
            preds = model(imgs).squeeze()
            pred_std = preds.mean().item()
            # un-standardize back to real units so R²/MAE are in original scale
            cluster_pred = pred_std * y_std + y_mean

            y_true.append(row["wealth_num"])
            y_pred.append(cluster_pred)

    return np.array(y_true), np.array(y_pred)

# ── K-Fold cross validation ───────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_best_r2s, fold_final_r2s, fold_best_maes = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────")
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # standardization stats come from the TRAIN fold only (no test leakage)
    if STANDARDIZE_TARGET:
        y_mean = float(train_df["wealth_num"].mean())
        y_std = float(train_df["wealth_num"].std()) or 1.0
        print(f"  standardizing target: mean={y_mean:.3f}, std={y_std:.3f}")
    else:
        y_mean, y_std = 0.0, 1.0

    model = build_model()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_r2 = -999
    best_mae = None
    last_r2 = None
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_df, optimizer, criterion, y_mean, y_std)
        y_true, y_pred = evaluate(model, test_df, y_mean, y_std)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        scheduler.step()
        print(f"Epoch {epoch+1:2d}: loss={loss:.4f}, R²={r2:.3f}, MAE={mae:.3f}")

        last_r2 = r2
        # We still SAVE the best-test-epoch weights (useful as a model artifact),
        # but for REPORTING, prefer the final-epoch R² below — selecting the best
        # epoch on the same test fold you report is optimistically biased.
        if r2 > best_r2:
            best_r2 = r2
            best_mae = mae
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, f"best_fold{fold+1}.pt"))

    fold_best_r2s.append(best_r2)
    fold_final_r2s.append(last_r2)
    fold_best_maes.append(best_mae)
    print(f"Fold {fold+1}: best R²={best_r2:.3f}, final R²={last_r2:.3f}")

print(f"\n── Final Results ─────────────────────────")
print(f"Mean final R²: {np.mean(fold_final_r2s):.3f} ± {np.std(fold_final_r2s):.3f}   <- report this")
print(f"Mean best R²:  {np.mean(fold_best_r2s):.3f} ± {np.std(fold_best_r2s):.3f}   (optimistic)")
print(f"Mean MAE (best epoch): {np.mean(fold_best_maes):.3f} ± {np.std(fold_best_maes):.3f}")