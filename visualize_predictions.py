import torch
import matplotlib.pyplot as plt
from models.ReachNet import ReachNet
from data.dataset import ScalarFieldToReachDataset
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import json
import os

# Paths
sf_path = "../g7_v100/sf"
bup_path = "../g7_v100/bup"
model_path = "trained_ReachNet.pth"
split_path = "split_indices.json"

# Load dataset and split
dataset = ScalarFieldToReachDataset(sf_path, bup_path)

# === Load split indices ===
if not os.path.exists(split_path):
    raise FileNotFoundError(f"Missing {split_path}. Make sure you trained and saved the split first.")
with open(split_path, "r") as f:
    indices = json.load(f)

val_indices = indices["val"]
val_ds = Subset(dataset, val_indices)

# Random Split
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# _, val_ds = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

# Load model
sample_sf, sample_bup = val_ds[0]
output_size = sample_bup.shape[0]
model = ReachNet(output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Visualize N samples
N = 4
for i, (sf, true_bup) in enumerate(val_loader):
    if i >= N:
        break

    sf = sf.to(device).float()
    true_bup = true_bup.to(device).float()

    with torch.no_grad():
        pred_bup = model(sf)

    pred_bup = pred_bup.cpu().squeeze().numpy()
    true_bup = true_bup.cpu().squeeze().numpy()

    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(true_bup, label='Ground Truth', linewidth=2)
    plt.plot(pred_bup, label='Prediction', linestyle='-.')
    plt.title(f"Validation Sample {i+1}")
    plt.xlabel("Curve Point Index")
    plt.ylabel("Reach Value")
    plt.legend()
    plt.tight_layout()
    plt.show()