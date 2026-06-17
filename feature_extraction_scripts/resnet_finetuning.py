import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
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
    "data/raw/images/more"
    ]
MATCHED_CSV = "data/processed/clusters_all_cities.csv"
OUTPUT_DIR = "finetuned_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 1  # one cluster at a time
N_FOLDS = 5

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def find_image(image_id):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

class ClusterDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_ids = row["image_ids"].split(",")
        wealth = torch.tensor(row["wealth_num"], dtype=torch.float32)

        images = []
        for image_id in image_ids:
            path = find_image(image_id.strip())
            if path is None:
                continue
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
            except:
                continue

        # Stack all images for this cluster → (N, 3, 224, 224)
        images = torch.stack(images) if images else torch.zeros(1, 3, 224, 224)
        return images, wealth

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace FC with regression head
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    return model.to(device)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, dataset_df, optimizer, criterion):
    model.train()
    total_loss = 0

    # Shuffle clusters
    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

    for _, row in dataset_df.iterrows():
        image_ids = row["image_ids"].split(",")
        wealth = torch.tensor(row["wealth_num"], dtype=torch.float32).to(device)

        images = []
        for image_id in image_ids:
            path = find_image(image_id.strip())
            if path is None:
                continue
            try:
                img = Image.open(path).convert("RGB")
                images.append(transform(img))
            except:
                continue

        if not images:
            continue

        imgs = torch.stack(images).to(device)  # (N, 3, 224, 224)

        optimizer.zero_grad()

        # Forward pass all images, mean pool predictions
        preds = model(imgs).squeeze()          # (N,)
        cluster_pred = preds.mean()            # scalar

        loss = criterion(cluster_pred, wealth)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset_df)

def evaluate(model, dataset_df):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for _, row in dataset_df.iterrows():
            image_ids = row["image_ids"].split(",")
            wealth = row["wealth_num"]

            images = []
            for image_id in image_ids:
                path = find_image(image_id.strip())
                if path is None:
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(transform(img))
                except:
                    continue

            if not images:
                continue

            imgs = torch.stack(images).to(device)
            preds = model(imgs).squeeze()
            cluster_pred = preds.mean().item()

            y_true.append(wealth)
            y_pred.append(cluster_pred)

    return np.array(y_true), np.array(y_pred)

# ── K-Fold cross validation ───────────────────────────────────────────────────
df = pd.read_csv(MATCHED_CSV)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_r2s, fold_maes = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────")
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    model = build_model()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_r2 = -999
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_df, optimizer, criterion)
        y_true, y_pred = evaluate(model, test_df)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        scheduler.step()
        print(f"Epoch {epoch+1:2d}: loss={loss:.4f}, R²={r2:.3f}, MAE={mae:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_fold{fold+1}.pt"))

    fold_r2s.append(best_r2)
    fold_maes.append(mae)
    print(f"Fold {fold+1} best R²: {best_r2:.3f}")

print(f"\n── Final Results ─────────────────────────")
print(f"Mean R²:  {np.mean(fold_r2s):.3f} ± {np.std(fold_r2s):.3f}")
print(f"Mean MAE: {np.mean(fold_maes):.3f} ± {np.std(fold_maes):.3f}")