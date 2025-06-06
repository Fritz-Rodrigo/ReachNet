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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

    # To use both smoothed and non-smoothed meshes
    if isinstance(meshes[0], dict):
        sm_meshes = [m.get("smoothed") for m in meshes]
        ns_meshes = [m.get("non_smoothed") for m in meshes]
        return (
            torch.stack(scalar_fields),
            torch.stack(bups),
            {"smoothed": sm_meshes, "non_smoothed": ns_meshes},
            list(filenames),
            list(metadata)
        )
    
    # To use just one mesh type
    else:
        return (
            torch.stack(scalar_fields),
            torch.stack(bups),
            list(meshes),      # list of one type of mesh
            list(filenames),
            list(metadata)
        )



def train_model(data_path_sf, data_path_bup, data_path_sm_stl, data_path_ns_stl, mesh_type="ns",
                val_fraction=0.2, epochs=10, batch_size=2, lr=1e-3, alpha_geom=0.1):

    # Determine output size from a sample bup file
    # Filter for valid bup files only
    bup_files = sorted([
        f for f in os.listdir(data_path_bup)
        if f.endswith('_bup.txt')
    ])
    if not bup_files:
        raise ValueError(f"No '_bup.txt' files found in {data_path_bup}.")
    # Try reading the first valid bup file
    for f in bup_files:
        bup_path = os.path.join(data_path_bup, f)
        try:
            sample_bup = np.loadtxt(bup_path)
            break  # Success! No need to check more files.
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    else:
        raise ValueError("All '_bup.txt' files failed to load. Check encoding or file content.")
    output_size = sample_bup.shape[0]
    
    now = datetime.now()
    print("Started at:", now)
    print("\nReachNet is trained with radii profiles & scalar fields to generate a new radii profile (blow-up file).")
    print("Each prediction corresponds to radii approximating the reach of the surfaces.")
    print(f"\nOutput size (curve points): {output_size}")

    # Load dataset
    full_dataset = data_prep(data_path_sf, data_path_bup,
                             sm_stl_dir=data_path_sm_stl if mesh_type in ("sm", "both") else None, 
                             ns_stl_dir=data_path_ns_stl if mesh_type in ("ns", "both") else None,
                             mesh_type=mesh_type)
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

        # --- Training ---
        model.train()
        train_loss = 0.0

        # fname and metadata kept for future logging/debugging:
        # Log specific examples by filename (e.g., for debugging)
        # Analyze patterns in training based on metadata (e.g., does genus affect loss?)
        # Save intermediate predictions with identifiable names
        for sf, bup, mesh, fname, metadata in train_loader:
            sf, bup = sf.to(device).float(), bup.to(device).float()
            optimizer.zero_grad()
            preds = model(sf)

            mse_loss = criterion(preds, bup)
            # This was before separating sm/ns meshes:
            # geom_loss = area_loss(mesh, preds.cpu().detach().numpy(), bup.cpu().numpy())
            if mesh_type == "both":
                sm_meshes = mesh["smoothed"]
                ns_meshes = mesh["non_smoothed"]
                geom_loss = area_loss(sm_meshes, preds.cpu().detach().numpy(), bup.cpu().numpy()) + \
                            area_loss(ns_meshes, preds.cpu().detach().numpy(), bup.cpu().numpy())
                geom_loss /= 2
            else:
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
                # This was before separating sm/ns meshes:
                # geom_loss = area_loss(mesh, preds.cpu().detach().numpy(), bup.cpu().numpy())
                if mesh_type == "both":
                    sm_meshes = mesh["smoothed"]
                    ns_meshes = mesh["non_smoothed"]
                    geom_loss = area_loss(sm_meshes, preds.cpu().detach().numpy(), bup.cpu().numpy()) + \
                                area_loss(ns_meshes, preds.cpu().detach().numpy(), bup.cpu().numpy())
                    geom_loss /= 2
                else:
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
    parser.add_argument("--bup_dir", required=True, help="Path to blow-up radii .txt files")
    parser.add_argument("--sm_stl_dir", required=True, help="Path to smoothed STL surface meshes")
    parser.add_argument("--ns_stl_dir", required=True, help="Path to non-smoothed STL surface meshes")
    parser.add_argument("--mesh_type", type=str, default="both", choices=["sm", "ns", "both"],
                        help="Use smoothed/non-smoothed meshes or both, for training loss")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation Fraction")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha_geom", type=float, default=1.0, help="Weight for geometric loss (surface area)")
    args = parser.parse_args()

    train_model(
        data_path_sf=args.sf_dir,
        data_path_bup=args.bup_dir,
        data_path_sm_stl=args.sm_stl_dir,
        data_path_ns_stl=args.ns_stl_dir,
        mesh_type=args.mesh_type,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_geom=args.alpha_geom
    )