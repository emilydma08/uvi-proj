import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# ── Load ──────────────────────────────────────────────────────────────────────
X = np.load("resnet-gsv-features/X_clusters.npy")  # (44, 2048)
y = np.load("resnet-gsv-features/y_labels.npy")    # (44,)
cluster_ids = pd.read_csv("resnet-gsv-features/cluster_ids.csv")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Wealth range: {y.min():.2f} - {y.max():.2f}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to reduce dimensionality — 44 samples, 2048 features is very high dimensional
# reduce to n_components < n_samples to avoid overfitting
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained by 30 components: {pca.explained_variance_ratio_.sum():.2%}")

# ── Model: Ridge Regression ───────────────────────────────────────────────────
# Ridge handles high-dimensional, small-sample data well via L2 regularization
# LOO cross-validation is best practice with only 44 samples
loo = LeaveOneOut()
model = Ridge(alpha=1.0)

y_pred = cross_val_predict(model, X_pca, y, cv=loo)

# ── Metrics ───────────────────────────────────────────────────────────────────
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"\n--- Results ---")
print(f"R²:  {r2:.3f}")
print(f"MAE: {mae:.3f}")

# Baseline: always predict mean wealth
baseline_pred = np.full_like(y, y.mean())
baseline_mae = mean_absolute_error(y, baseline_pred)
baseline_r2 = r2_score(y, baseline_pred)
print(f"\n--- Baseline (predict mean) ---")
print(f"R²:  {baseline_r2:.3f}")
print(f"MAE: {baseline_mae:.3f}")

# ── Plot: predicted vs actual ─────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(7, 5))
plt.scatter(y, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual Wealth Score")
plt.ylabel("Predicted Wealth Score")
plt.title(f"Predicted vs Actual Wealth\nR²={r2:.3f}, MAE={mae:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/gsv_resnet_predicted_vs_actual.png", dpi=150)
plt.show()
print("Saved predicted_vs_actual.png")

# ── Try different alpha values ────────────────────────────────────────────────
print("\n--- Ridge alpha sweep ---")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = Ridge(alpha=alpha)
    preds = cross_val_predict(model, X_pca, y, cv=loo)
    print(f"alpha={alpha:6}: R²={r2_score(y, preds):.3f}, MAE={mean_absolute_error(y, preds):.3f}")