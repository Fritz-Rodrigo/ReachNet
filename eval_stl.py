import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from models.ReachNet import ReachNet
from data.data_prep import data_prep
from utils.bup_to_scalar_field import bup_to_scalar_field
from skimage.measure import marching_cubes
import trimesh
import json

# ----- Config -----
MODEL_PATH = "trained_ReachNet1.pth"
SPLIT_FILE = "split_indices.json"
DATA_DIR = "../g7_v100"
SAVE_TXT_DIR = "Predictions"
SAVE_STL_DIR = "Predictions/STL"
BATCH_SIZE = 1
NUM_EXAMPLES = 2

os.makedirs(SAVE_TXT_DIR, exist_ok=True)
os.makedirs(SAVE_STL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load data -----
print("📂 Loading dataset...")
dataset = data_prep(
    sf_dir=os.path.join(DATA_DIR, "sf"),
    bup_dir=os.path.join(DATA_DIR, "bup"),
    stl_dir=os.path.join(DATA_DIR, "ns")
)

with open(SPLIT_FILE, "r") as f:
    val_indices = json.load(f)["val"]

val_ds = Subset(dataset, val_indices)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ----- Load model -----
print(f"📥 Loading model from {MODEL_PATH}")
output_size = len(np.loadtxt(os.path.join(DATA_DIR, "bup", os.listdir(os.path.join(DATA_DIR, "bup"))[0])))
model = ReachNet(output_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# ----- Evaluate -----
print("🔍 Running inference on validation set...")
for i, (sf, bup, mesh, filenames, metadata) in enumerate(val_loader):
    if i >= NUM_EXAMPLES:
        break

    sf = sf.to(device).float()
    bup = bup.to(device).float()
    pred = model(sf).squeeze().detach().cpu().numpy()
    gt = bup.squeeze().cpu().numpy()

    # ---- Plot prediction vs ground truth ----
    plt.figure(figsize=(10, 4))
    plt.plot(gt, label='Ground Truth', linewidth=2)
    plt.plot(pred, label='Prediction', linestyle='.')
    plt.title(f"Prediction vs Ground Truth (Example {i})")
    plt.xlabel("Curve Index")
    plt.ylabel("Radius")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # ---- Save plot ----
    example_id = os.path.splitext(os.path.basename(filenames[0]))[0].replace("_sf", "")
    plot_filename = os.path.join(SAVE_TXT_DIR, f"{example_id}_pred_vs_gt.png")
    plt.savefig(plot_filename)
    print(f"✅ Saved plot to {plot_filename}")
    plt.show()

    # ---- Save prediction ----
    np.savetxt(os.path.join(SAVE_TXT_DIR, f"{example_id}_pred_bup.txt"), pred)
    print(f"✅ Saved predicted radii to {example_id}_pred_bup.txt")

    # ---- Reconstruct scalar field ----
    scalar_field_pred = bup_to_scalar_field(pred, metadata[0])
    verts, faces, _, _ = marching_cubes(scalar_field_pred, level=0.0)
    mesh_pred = trimesh.Trimesh(vertices=verts, faces=faces)

    # ---- Save mesh ----
    stl_path = os.path.join(SAVE_STL_DIR, f"{example_id}_pred.stl")
    mesh_pred.export(stl_path)
    print(f"🧊 Saved predicted surface to {stl_path}")
