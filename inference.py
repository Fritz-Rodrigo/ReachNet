import torch
import numpy as np
from models.ReachNet import Conv3DAutoencoder
from utils.marching_cubes import apply_mc
from utils.mesh_metrics import compute_volume
import trimesh
import os

def run_inference(sf_path, model_path, output_dir):
    # === Load scalar field ===
    scalar = np.loadtxt(sf_path)
    size = int(np.cbrt(scalar.size))
    scalar = scalar.reshape((size, size, size))

    # Prepare tensor for model
    input_tensor = torch.tensor(scalar, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, D, H, W]

    # === Load model ===
    model = Conv3DAutoencoder().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # === Predict scalar field ===
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze().cpu().numpy()

    # === Save predicted scalar field ===
    base_filename = os.path.basename(sf_path).replace('_sf.txt', '')
    sf_pred_path = os.path.join(output_dir, f"{base_filename}_sf_pred.txt")
    np.savetxt(sf_pred_path, output_tensor.reshape(-1))
    print(f"📄 Saved predicted scalar field: {sf_pred_path}")

    # === Generate & save predicted mesh ===
    verts_pred, faces_pred = apply_mc(output_tensor, level=0.0)
    mesh_pred = trimesh.Trimesh(vertices=verts_pred, faces=faces_pred)
    stl_pred_path = os.path.join(output_dir, f"{base_filename}_pred.stl")
    mesh_pred.export(stl_pred_path)
    print(f"✅ Saved predicted STL: {stl_pred_path}")

    # === Compare to original STL ===
    stl_orig_path = sf_path.replace('/sf/', '/stl/').replace('_sf.txt', '.stl')
    if not os.path.exists(stl_orig_path):
        print(f"⚠️ Original STL not found: {stl_orig_path}")
        return

    mesh_orig = trimesh.load_mesh(stl_orig_path)
    vol_orig = compute_volume(mesh_orig)
    vol_pred = compute_volume(mesh_pred)
    diff = (vol_pred - vol_orig) / vol_orig * 100

    print(f"\n📊 Volume comparison:")
    print(f" - Original volume:  {vol_orig:.4f}")
    print(f" - Predicted volume: {vol_pred:.4f}")
    print(f" - Difference:       {diff:+.2f}%")
