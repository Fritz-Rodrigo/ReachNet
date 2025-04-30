import torch
import trimesh
import numpy as np

def area_loss(mesh_paths, pred_radii, gt_radii):
    """
    Computes surface area-based loss as L1 norm of area differences
    between the surfaces obtained from pred_radii and gt_radii.
    
    Args:
        mesh_paths: list of STL file paths (original mesh)
        pred_radii: np.array of shape (B, N) - predicted radii
        gt_radii:   np.array of shape (B, N) - ground truth radii
    
    Returns:
        Scalar torch tensor with surface area difference loss.
    """
    total_loss = 0.0
    batch_size = len(mesh_paths)

    for i in range(batch_size):
        mesh = trimesh.load(mesh_paths[i])
        area_gt = mesh.area  # Using original mesh as proxy

        # Assume we cannot directly rescale; simulate area change:
        # Estimate new area proportional to radius change
        scale_factor = np.mean(pred_radii[i]) / (np.mean(gt_radii[i]) + 1e-8)
        estimated_area = area_gt * (scale_factor ** 2)  # surface area ~ r²

        diff = abs(estimated_area - area_gt)
        total_loss += diff

    return torch.tensor(total_loss / batch_size, requires_grad=True)
