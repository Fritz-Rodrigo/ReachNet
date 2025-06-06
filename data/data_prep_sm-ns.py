import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np
import re
import os
import glob

class data_prep(Dataset):

    def __init__(self, sf_dir, bup_dir, sm_stl_dir=None, ns_stl_dir=None, mesh_type="sm"):
        self.sf_dir = sf_dir
        self.bup_dir = bup_dir
        self.sm_stl_dir = sm_stl_dir
        self.ns_stl_dir = ns_stl_dir
        self.mesh_type = mesh_type

        if mesh_type not in ("sm", "ns", "both"):
            raise ValueError(f"Invalid mesh_type: {mesh_type}. Must be 'sm', 'ns', or 'both'.")
        if mesh_type in ("sm", "both") and not sm_stl_dir:
            raise ValueError("mesh_type is 'sm' or 'both' but sm_stl_dir is None.")
        if mesh_type in ("ns", "both") and not ns_stl_dir:
            raise ValueError("mesh_type is 'ns' or 'both' but ns_stl_dir is None.")

        self.files = []
        for f in os.listdir(sf_dir):
            if not f.endswith("_sf.txt"):
                continue
            base_name = f.replace("_sf.txt", "")
            bup_filename = f"{base_name}_bup.txt"

            sm_match = []
            ns_match = []
            if sm_stl_dir:
                sm_match = glob.glob(os.path.join(sm_stl_dir, f"{base_name}_a*_sm.stl"))
            if ns_stl_dir:
                ns_match = glob.glob(os.path.join(ns_stl_dir, f"{base_name}_a*_ns.stl"))

            # Determine if appropriate STL files exist
            if mesh_type == "sm" and not sm_match:
                continue
            if mesh_type == "ns" and not ns_match:
                continue
            if mesh_type == "both" and (not sm_match or not ns_match):
                continue

            # Grab the first match for area extraction
            area_match = sm_match or ns_match
            if not area_match:
                print(f"[WARNING] Could not find STL file for {base_name}")
                continue

            area = re.search(r"_a([0-9.]+)", area_match[0])
            area_str = f"_a{area.group(1)}" if area else ""

            # Construct full file paths
            sf_path = os.path.join(sf_dir, f)
            bup_path = os.path.join(bup_dir, bup_filename)
            sm_stl_path = os.path.join(sm_stl_dir, f"{base_name}{area_str}_sm.stl") if sm_stl_dir else None
            ns_stl_path = os.path.join(ns_stl_dir, f"{base_name}{area_str}_ns.stl") if ns_stl_dir else None

            # Conditional existence checks
            missing = []
            if not os.path.exists(bup_path):
                missing.append("bup")
            if mesh_type in ("sm", "both") and (not sm_stl_path or not os.path.exists(sm_stl_path)):
                missing.append("sm_stl")
            if mesh_type in ("ns", "both") and (not ns_stl_path or not os.path.exists(ns_stl_path)):
                missing.append("ns_stl")

            if missing:
                print(f"[Warning] Missing {', '.join(missing)} for {f}, skipping.")
                continue

            self.files.append((sf_path, bup_path, sm_stl_path, ns_stl_path, base_name))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        sf_path, bup_path, sm_stl_path, ns_stl_path, base_name = self.files[idx]

        # Load scalar field and convert to tensor
        sf = np.loadtxt(sf_path)
        side_len = round(sf.size**(1/3))
        if side_len**3 != sf.size:
            raise ValueError(f"Cannot reshape scalar field {sf_path} to cube.")
        sf = sf.reshape((side_len, side_len, side_len))
        sf_tensor = torch.tensor(sf, dtype=torch.float32).unsqueeze(0)

        # Load blow-up radii and convert to tensor
        radii = np.loadtxt(bup_path)
        radii_tensor = torch.tensor(radii, dtype=torch.float32)

        # Load smoothed & non-smoothed STL meshes and convert them to tensors
        sm_mesh = trimesh.load(sm_stl_path, force='mesh')
        ns_mesh = trimesh.load(ns_stl_path, force='mesh')
        stl_mesh = {"smoothed": sm_mesh, "non_smoothed": ns_mesh}
        
        vertices, faces = self.mesh_to_tensor(stl_mesh)

        # Parse metadata from base name
        metadata = self.parse_metadata(base_name)

        # Package mesh as dictionary
        mesh = {'vertices': vertices, 'faces': faces, 'smoothed': metadata["smoothed"]}

        return sf_tensor, radii_tensor, mesh, base_name, metadata


    @staticmethod # To call parse_phase in the static method for parse_metadata
    def parse_phase(val):
        if val.startswith("pi"):
            if val == "pi":
                return np.pi
            else:
                try:
                    return np.pi / int(val[2:])
                except Exception:
                    return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0


    # Parse data for 'sinusoidal' method
    @staticmethod # To call parse_metadata without a full dataset object
    def parse_metadata(filename):
        """
        Parse metadata from structured filename, e.g.:
        g7_lissajous_3,7,8,0,pi5,0_16_1.1_±0.2_r0.02_0.029_v90
        """
        try:
            parts = filename.split("_")
            genus = int(parts[0][1:])  # 'g7' -> 7
            knot_type = parts[1]

            # Split the 6 comma-separated parameters into freqs and phases
            freq_phase = parts[2].split(",")
            freqs = tuple(map(int, freq_phase[:3]))
            phases = tuple(data_prep.parse_phase(p) for p in freq_phase[3:6])

            sin_freq = int(parts[3])
            sin_offset = float(parts[4])
            sin_amp = parts[5]

            rmin = float(parts[6][1:])  # remove leading 'r'
            rmax = float(parts[7])

            voxels = int(parts[8][1:])  # remove leading 'v'

            return {
                "genus": genus,
                "knot type": knot_type,
                "frequencies": freqs,
                "phases": phases,
                "sinusoidal frequency": sin_freq,
                "sinusoidal offset": sin_offset,
                "sinusoidal amplitude": sin_amp,
                "minimum radius": rmin,
                "maximum radius": rmax,
                "voxels per axis": voxels
            }

        except Exception as e:
            raise ValueError(f"Failed to parse '{filename}': {e}")


    # Parse data for 'minimal', 'manual', 'automatic', and 'voronoi' methods
    @staticmethod # To call parse_metadata without a full dataset object
    def parse_metadata_reach_based(filename):
        """
        Parse metadata from filenames generated with the 'manual' or 'automatic' reach-based methods.
        Example filenames:
            g7_lissajous_1,7,9,0,pi2,0_manual_r0.05_v40_sf.txt
            g7_lissajous_2,3,5,0,0,pi2_automatic_its30_r0.015_0.298_v173_sf.txt
        """
        name = os.path.basename(filename)
        if name.endswith(".stl"):
            smoothed = not name.endswith("_ns.stl")  # Smoothed STL if it does NOT end in _ns.stl

        # Strip extensions and suffixes
        for suffix in ["_ns.stl", "_sm.stl", "_sf.txt", "_bup.txt"]:
            if name.endswith(suffix):
                # name = name.replace(suffix, "")
                name = name[:-len(suffix)]
                break
        parts = name.split("_")
        freq_phase = parts[2].split(",") # Split the 6 comma-separated parameters into freqs and phases

        # Common fields
        metadata = {
            "genus": int(parts[0][1:]),  # remove 'g'
            "knot type": parts[1],
            "frequencies": tuple(map(int, freq_phase[:3])),
            "phases": tuple(data_prep.parse_phase(p) for p in freq_phase[3:6]),
            "method": parts[3],
        }

        method = metadata["method"]

        if method == "minimal" or method == "manual":
            metadata["radius"] = float(parts[4][1:])  # e.g., r0.05
            metadata["voxels"] = int(parts[5][1:])    # e.g., v40 → 40

        elif method == "automatic":
            metadata["iterations"] = int(parts[4][3:])   # e.g., its30 → 30
            # Check if next part starts with 'pts'
            if parts[5].startswith("pts"):
                metadata["points"] = int(parts[5][3:])       # e.g., pts470 → 470
                metadata["min radius"] = float(parts[6][1:]) # remove leading 'r'
                metadata["max radius"] = float(parts[7])
                metadata["voxels"] = int(parts[8][1:])   # remove leading 'v'
            else:
                # e.g., r0.015_0.298 → min and max radius
                metadata["min radius"] = float(parts[5][1:]) # remove leading 'r'
                metadata["max radius"] = float(parts[6])
                metadata["voxels"] = int(parts[7][1:])       # remove leading 'v'
        
        elif method == "voronoi":
            metadata["points"] = int(parts[4][3:])       # e.g., pts100 → 100
            metadata["offset %"] = float(parts[5][3:])   # e.g., off1.5 → 1.5
            # e.g., r0.015_0.298 → min and max radius
            metadata["min radius"] = float(parts[6][1:]) # remove leading 'r'
            metadata["max radius"] = float(parts[7])
            metadata["voxels"] = int(parts[8][1:])       # remove leading 'v'

        else:
            raise ValueError(f"Unknown reach-based method in filename: {metadata['method']}")
        
        # Parse area if present (e.g., a21.77)
        for part in parts:
            if part.startswith("a"):
                try:
                    metadata["surface area"] = float(part[1:]) # remove leading 'a'
                    break
                except ValueError:
                    continue  # Not a valid float after 'a', skip
        
        # Finally add 'smoothed: True/False'
        metadata["smoothed"] = smoothed

        return metadata

            
    def mesh_to_tensor(self, mesh):
        """Convert a Trimesh object to a tensor representation."""
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)  # (V, 3) shape
        faces = torch.tensor(mesh.faces, dtype=torch.int64)  # (F, 3) shape
        return vertices, faces