
import trimesh
import torch
import torch.nn.functional as F
from utils.marching_cubes import apply_mc
from utils.mesh_metrics import compute_volume

def geometry_loss(input_field, output_field):
    laplacian = torch.mean((output_field[:, :, :-1] - output_field[:, :, 1:]) ** 2)
    meshes = []
    for f in output_field:
        verts, faces = apply_mc(f.squeeze().detach().cpu().numpy())
        if len(verts) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            meshes.append(mesh)
    if not meshes:
        return torch.tensor(0.0, requires_grad=True).cuda()  # skip if no valid mesh
    print(f"[Info] {len(meshes)} valid meshes in batch")
    vols = [compute_volume(mesh) for mesh in meshes]
    vol_loss = -torch.mean(torch.tensor(vols))
    return laplacian + 0.1 * vol_loss
