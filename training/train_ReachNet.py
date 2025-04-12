import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import json
from models.ReachNet import ReachNet
from data.dataset import ScalarFieldToReachDataset
import numpy as np
import os

def train_model(data_path_sf, data_path_bup,
                epochs=10, batch_size=2, lr=1e-3,
                val_fraction=0.2):
    # Determine output_size from one bup file
    bup_files = sorted(os.listdir(data_path_bup))
    if not bup_files:
        raise ValueError("No bup files found in the folder")
    sample_bup_path = os.path.join(data_path_bup, bup_files[0])
    sample_bup = np.loadtxt(sample_bup_path)
    output_size = sample_bup.shape[0]
    print("ReachNet will be trained on scalar fields to generate a new blow-up (bup) file")
    print("\nwith optimized radii according to the reach of each singular knot surface\n")
    print(f"Output size (curve points): {output_size}")

    # Full dataset
    full_dataset = ScalarFieldToReachDataset(data_path_sf, data_path_bup)
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_fraction)
    train_size = total_samples - val_size

    # Option 1: Load existing split indices, if any
    train_indices, val_indices = load_split_indices()
    if train_indices is None or val_indices is None:
        # No saved split indices; create a new random split (set a seed for reproducibility)
        torch.manual_seed(42)
        indices = torch.randperm(total_samples).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        save_split_indices(train_indices, val_indices)
    else:
        print(f"Loaded split indices from {SPLIT_INDICES_FILE}")

    train_ds = Subset(full_dataset, train_indices)
    val_ds   = Subset(full_dataset, val_indices)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model & device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReachNet(output_size).to(device)
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for sf, bup in train_loader:
            sf, bup = sf.to(device).float(), bup.to(device).float()
            optimizer.zero_grad()
            preds = model(sf)
            loss  = criterion(preds, bup)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_mse = train_loss / len(train_loader)
        train_rmse = np.sqrt(train_mse)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sf, bup in val_loader:
                sf, bup = sf.to(device).float(), bup.to(device).float()
                preds = model(sf)
                loss  = criterion(preds, bup)
                val_loss += loss.item()

        val_mse  = val_loss / len(val_loader)
        val_rmse = np.sqrt(val_mse)

        print(f"Epoch {epoch+1}/{epochs}  "
              # f"Train MSE: {train_mse:.6e}, RMSE: {train_rmse:.6e}  "
              f"Train RMSE: {train_rmse:.2e}  "
              # f"Val MSE: {val_mse:.6e}, RMSE: {val_rmse:.6e}")
              f"Val RMSE: {val_rmse:.2e}")

    # Save
    torch.save(model.state_dict(), "trained_ReachNet1.pth")
    print("Saved trained_ReachNet1.pth ✅")


SPLIT_INDICES_FILE = "split_indices.json"

def save_split_indices(train_indices, val_indices):
    data = {"train": train_indices, "val": val_indices}
    with open(SPLIT_INDICES_FILE, "w") as f:
        json.dump(data, f)
    print(f"Saved split indices to {SPLIT_INDICES_FILE}")

def load_split_indices():
    if os.path.exists(SPLIT_INDICES_FILE):
        with open(SPLIT_INDICES_FILE, "r") as f:
            data = json.load(f)
        return data["train"], data["val"]
    return None, None