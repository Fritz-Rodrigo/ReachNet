import torch
import matplotlib.pyplot as plt
from models.ReachNet import ReachNet
from data.data_prep import data_prep
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import json
import os

# === PATHS ===
# Save Folder
predictions_dir = 'Predictions'
os.makedirs(predictions_dir, exist_ok=True)

# Scalar Fields
sf_path = "../g7_v100/sf"
# Radii
bup_path = "../g7_v100/bup"
# STLs
stl_path = "../g7_v100/ns"
# Trained Model
model_path = "trained_ReachNet_sin3.pth"
# Split Indices
split_path = "split_indices.json"

# Load dataset
dataset = data_prep(sf_path, bup_path, stl_path)

# Random Split
train_size = int(round(0.8*len(dataset),0))
val_size = len(dataset) - train_size
print(f'Training size: {train_size}')
print(f'Validation size: {val_size}')
# _, val_ds = random_split(dataset, [train_size, val_size])

# Ordered Split: first 80% training, last 20% validation
train_indices = list(range(train_size))
val_indices = list(range(train_size, train_size + val_size))
train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

# === Load split indices ===
# if not os.path.exists(split_path):
#     raise FileNotFoundError(f"Missing {split_path}. Make sure you trained and saved the split first.")
# with open(split_path, "r") as f:
#     indices = json.load(f)
# val_indices = indices["val"]
# val_ds = Subset(dataset, val_indices)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# Load model
sample_sf, sample_bup, *_ = val_ds[0]
output_size = sample_bup.shape[0]
model = ReachNet(output_size)
# model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Visualize N samples
N = int(input(f'How many validation samples to visualize? (Maximum {val_size})\n'))
for i, (sf, true_bup, *_) in enumerate(val_loader):
    if i >= N:
        break

    sf = sf.to(device).float()
    true_bup = true_bup.to(device).float()

    with torch.no_grad():
        pred_bup = model(sf)

    pred_bup = pred_bup.cpu().squeeze().numpy()
    true_bup = true_bup.cpu().squeeze().numpy()

    # Save to file
    np.savetxt(os.path.join(predictions_dir, f"pred_{i+1:02d}.txt"), pred_bup)
    np.savetxt(os.path.join(predictions_dir, f"true_{i+1:02d}.txt"), true_bup)

    # Plot and save es PNG
    plt.figure(figsize=(10, 3))
    plt.plot(true_bup, '.', label='Ground Truth')
    plt.plot(pred_bup, '.', label='Prediction')
    plt.title(f"Validation Sample {i+1}")
    plt.xlabel("Curve Point Index")
    plt.ylabel("Radius")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(predictions_dir, f"plot_{i+1:02d}.png")
    plt.savefig(plot_path)
    plt.close()
    # plt.show()

print(f"Saved {N} predictions, ground truths, and plots to '{predictions_dir}'")