from skimage.measure import marching_cubes
import numpy as np

def apply_mc(field_np, level=0.0):
    min_val = np.min(field_np)
    max_val = np.max(field_np)

    if level < min_val or level > max_val:
        # Return dummy values or raise a warning/log instead
        print(f"[Warning] Skipping MC: level={level} not in range [{min_val:.2f}, {max_val:.2f}]")
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)  # empty verts and faces

    verts, faces, _, _ = marching_cubes(field_np, level=level)
    return verts, faces
