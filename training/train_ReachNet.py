import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.ReachNet import ReachNet
from data.dataset import ScalarFieldToReachDataset
import numpy as np
import os


def train_model(data_path_sf, data_path_bup,
                epochs=10, batch_size=2, lr=1e-3,
                val_fraction=0.2):
    # Determine output_size from one bup file
    bup_files = sorted(os.listdir(data_path_bup))
    sample_bup = np.loadtxt(os.path.join(data_path_bup, bup_files[0]))
    output_size = sample_bup.shape[0]
    print(f"Output size (curve points): {output_size}")

    # Full dataset
    full_dataset = ScalarFieldToReachDataset(data_path_sf, data_path_bup)
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_fraction)
    train_size = total_samples - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"Training on {train_size} samples, validating on {val_size} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model & device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReachNet(output_size).to(device)
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for sf, bup in train_loader:
            sf, bup = sf.to(device), bup.to(device)

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
                sf, bup = sf.to(device), bup.to(device)
                preds = model(sf)
                loss  = criterion(preds, bup)
                val_loss += loss.item()

        val_mse  = val_loss / len(val_loader)
        val_rmse = np.sqrt(val_mse)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Train MSE: {train_mse:.6e}, RMSE: {train_rmse:.6e}  "
              f"Val MSE: {val_mse:.6e}, RMSE: {val_rmse:.6e}")

    # Save
    torch.save(model.state_dict(), "trained_ReachNet.pth")
    print("✅ Saved trained_ReachNet.pth")
