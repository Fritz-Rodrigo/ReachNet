import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from models.ReachNet import ReachNet
from data.data_prep import data_prep
from training.geometric_losses import area_loss

SPLIT_INDICES_FILE = "split_indices.json"

def save_split_indices(train_indices, val_indices):
    data = {"train": train_indices, "val": val_indices}
    with open(SPLIT_INDICES_FILE, "w") as f:
        json.dump(data, f)
    print(f"✅ Saved split indices to {SPLIT_INDICES_FILE}")

def load_split_indices():
    if os.path.exists(SPLIT_INDICES_FILE):
        with open(SPLIT_INDICES_FILE, "r") as f:
            data = json.load(f)
        return data["train"], data["val"]
    return None, None

def custom_collate(batch):
    scalar_fields, bups, meshes, filenames, metadata = zip(*batch)
    return (
        torch.stack(scalar_fields),
        torch.stack(bups),
        list(meshes),         # keep STL mesh tensors as a list
        list(filenames),      # filenames as list of strings
        list(metadata)
    )


def train_model(data_path_sf, data_path_bup, data_path_stl,
                epochs=10, batch_size=2, lr=1e-3,
                val_fraction=0.1, alpha_geom=0.1):

    # Determine output size from a sample bup file
    bup_files = sorted(os.listdir(data_path_bup))
    if not bup_files:
        raise ValueError("No bup files found in the folder.")
    sample_bup = np.loadtxt(os.path.join(data_path_bup, bup_files[0]))
    output_size = sample_bup.shape[0]
    
    now = datetime.now()
    print("Started at:", now)
    print("\nReachNet will be trained on scalar fields and meshes to generate a new blow-up (bup) file.")
    print("Each prediction corresponds to radii approximating the reach of a singular knot surface.")
    print(f"\nOutput size (curve points): {output_size}")

    # Load dataset
    full_dataset = data_prep(data_path_sf, data_path_bup, data_path_stl)
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_fraction)
    train_size = total_samples - val_size

    # Load or create split
    train_indices, val_indices = load_split_indices()
    if train_indices is None or val_indices is None:
        torch.manual_seed(42)
        indices = torch.randperm(total_samples).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        save_split_indices(train_indices, val_indices)
    else:
        print(f"📂 Loaded split indices from {SPLIT_INDICES_FILE}")

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print(f"📊 Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

    # Setup model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReachNet(output_size).to(device)
    print(f"🖥 Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_rmse_list = []
    val_rmse_list = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sf, bup, mesh, fname, metadata in train_loader:
            sf, bup = sf.to(device).float(), bup.to(device).float()
            optimizer.zero_grad()
            preds = model(sf)

            mse_loss = criterion(preds, bup)
            geom_loss = area_loss(mesh, preds.cpu().detach().numpy(), bup.cpu().numpy())
            total_loss = mse_loss + alpha_geom*geom_loss

            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            train_loss += total_loss.item()

        train_mse = train_loss / len(train_loader)
        train_rmse = np.sqrt(train_mse)
        train_rmse_list.append(train_rmse)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sf, bup, mesh, *_ in val_loader:
                sf, bup = sf.to(device).float(), bup.to(device).float()
                preds = model(sf)

                mse_loss = criterion(preds, bup)
                geom_loss = area_loss(mesh, preds.cpu().detach().numpy(), bup.cpu().numpy())
                total_loss = mse_loss + alpha_geom * geom_loss

                val_loss += total_loss.item()

        val_mse = val_loss / len(val_loader)
        val_rmse = np.sqrt(val_mse)
        val_rmse_list.append(val_rmse)

        print(f"📈 Epoch {epoch+1}/{epochs}  "
              f"Train RMSE: {train_rmse:.3e}  "
              f"Val RMSE: {val_rmse:.3e}")

    save_name = "trained_ReachNet.pth"
    torch.save(model.state_dict(), save_name)
    print(f"💾 Saved {save_name} ✅")
    
    now = datetime.now()
    print("Ended at:", now)

    # Save directory
    plot_dir = "./"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, "train_val.png")

    # Plot training and validation RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_rmse_list, label='Train RMSE', color='blue', marker='o')
    plt.plot(range(epochs), val_rmse_list, label='Validation RMSE', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    print(f"📈 Saved RMSE plot to '{plot_filename}'")


# ---------- ARGPARSE ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReachNet with geometric losses")
    parser.add_argument("--sf_dir", required=True, help="Path to scalar field .txt files")
    parser.add_argument("--bup_dir", required=True, help="Path to blow-up (radius) .txt files")
    parser.add_argument("--stl_dir", required=True, help="Path to STL surface meshes")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha_geom", type=float, default=1.0, help="Weight for geometric loss (surface area)")
    args = parser.parse_args()

    train_model(
        data_path_sf=args.sf_dir,
        data_path_bup=args.bup_dir,
        data_path_stl=args.stl_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_geom=args.alpha_geom
    )