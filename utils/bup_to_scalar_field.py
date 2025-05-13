import numpy as np
from scipy.spatial import cKDTree

def bup_to_scalar_field(pred_radii, metadata, grid_size=90, padding=5):
    """
    Rebuild scalar field from predicted radii and curve metadata.
    Scalar field has +1 outside the tube, -1 inside.

    Args:
        pred_radii (np.ndarray): predicted radius at each curve point [N]
        metadata (dict): must contain 'curve_points' (Nx3 np.ndarray)
        grid_size (int): voxel resolution along each axis (default: 90)
        padding (int): extra grid margin to ensure curve is fully embedded

    Returns:
        np.ndarray: scalar field of shape (grid_size, grid_size, grid_size)
    """
    assert "curve_points" in metadata, "Metadata must contain 'curve_points'"
    curve = metadata["curve_points"]  # shape [N, 3]

    # Normalize curve to fit in unit cube with margin
    curve = np.asarray(curve)
    min_coord = curve.min(axis=0)
    max_coord = curve.max(axis=0)
    scale = max(max_coord - min_coord)
    curve = (curve - min_coord) / scale  # now in [0,1]^3
    curve *= (grid_size - 2*padding - 1)
    curve += padding

    # Create a KDTree for fast nearest neighbor queries
    tree = cKDTree(curve)
    grid = np.indices((grid_size, grid_size, grid_size)).reshape(3,-1).T  # (V, 3)

    # Query closest curve point for each voxel
    dists, indices = tree.query(grid, k=1)
    radii = pred_radii[indices]

    inside_mask = dists < radii
    scalar_field = np.ones(grid.shape[0], dtype=np.float32)
    scalar_field[inside_mask] = -1.0

    # Reshape back to 3D volume
    scalar_field = scalar_field.reshape((grid_size, grid_size, grid_size))
    return scalar_field