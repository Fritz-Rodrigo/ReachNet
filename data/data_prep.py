import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np
import re
import os
import glob

class data_prep(Dataset):

    def __init__(self, sf_dir, bup_dir, stl_dir):
        self.sf_dir  = sf_dir
        self.bup_dir = bup_dir
        self.stl_dir = stl_dir

        self.files = []
        for f in os.listdir(sf_dir):
            if not f.endswith("_sf.txt"):
                continue
            base_name = f.replace("_sf.txt", "")
            bup_filename = f"{base_name}_bup.txt"

            # area extraction
            area_match = []
            if stl_dir:
                area_match = glob.glob(os.path.join(stl_dir, f"{base_name}_a*_ns.stl"))
            if not area_match:
                print(f"[WARNING] Could not find STL file for {base_name}")
                continue
            area = re.search(r"_a([0-9.]+)", area_match[0])
            area_str = f"_a{area.group(1)}" if area else ""

            # Construct full file paths
            sf_path = os.path.join(sf_dir, f)
            bup_path = os.path.join(bup_dir, bup_filename)
            stl_path = os.path.join(stl_dir, f"{base_name}{area_str}_ns.stl") if stl_dir else None

            # Conditional existence checks
            missing = []
            if not os.path.exists(bup_path):
                missing.append("bup")
            if not os.path.exists(stl_path):
                missing.append("stl")
            if missing:
                print(f"[Warning] Missing {', '.join(missing)} for {f}, skipping.")
                continue

            self.files.append((sf_path, bup_path, stl_path, base_name))

        # Sorted lists of all files:
        # sf_path = sorted([f for f in os.listdir(sf_dir) if f.endswith('_sf.txt')])
        # bup_path = sorted([f for f in os.listdir(bup_dir) if f.endswith('_bup.txt')])
        # stl_path = sorted([f for f in os.listdir(stl_dir) if f.endswith('.stl')])
        # base_name = sorted([f.replace("_sf.txt", "") for f in os.listdir(sf_dir) if f.endswith('_sf.txt')])
        # self.files.append((sf_path, bup_path, stl_path, base_name))

        if len(self.files) == 0:
            print(f"[ERROR] No valid training samples found in {sf_dir}")
        # else:
        #     sf_path, bup_path, stl_path, base_name = self.files[0]
        #     print(f"[DEBUG] Sample 0:")
        #     print(f"  Scalar field: {sf_path}")
        #     print(f"  BUP file    : {bup_path}")
        #     print(f"  STL file    : {stl_path}")
        #     print(f"  Base Name   : {base_name}")


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        sf_path, bup_path, stl_path, base_name = self.files[idx]

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

        # Load STL mesh and convert to tensor
        stl_mesh = trimesh.load(stl_path)
        
        vertices, faces, smoothed = self.mesh_to_tensor(stl_mesh)

        # Parse SINUSOIDAL metadata from base name
        # metadata = self.parse_metadata(base_name)

        # Parse REACH metadata from base name
        metadata = self.parse_metadata_reach_based(base_name)

        # Package mesh as dictionary
        mesh = {'vertices': vertices, 'faces': faces, 'smoothed': smoothed}

        # Debug
        if any(x is None for x in [sf, radii, mesh, base_name]):
            print(f"[WARNING] None value detected in __getitem__ for idx={idx}, file={base_name}")
            return None
        
        # Verify info of loaded files
        # print(f"[OK] Loaded sample {idx}: SF shape: {sf.shape}, BUP size: {radii.size}, STL vertices: {mesh['vertices'].shape}")

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


    # Parse data for SINUSOIDAL method
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
    @staticmethod # To call parse_metadata_reach_based without a full dataset object
    def parse_metadata_reach_based(filename):
        """
        Parse metadata from filenames generated with the 'manual' or 'automatic' reach-based methods.
        Example filenames:
            g7_lissajous_1,7,9,0,pi2,0_manual_r0.05_v40_sf.txt
            g7_lissajous_2,3,5,0,0,pi2_automatic_its30_r0.015_0.298_v173_bup.txt
            g7_lissajous_3,5,8,0,pi2,0_voronoi_pts180_off0.5_r0.052_0.145_v58_a31.8_ns.stl
        """
        name = os.path.basename(filename)

        # Strip extensions and suffixes
        core_name = name
        for suffix in ["_ns.stl", ".stl", "_sf.txt", "_bup.txt"]:
            if core_name.endswith(suffix):
                # core_name = core_name.replace(suffix, "")
                core_name = core_name[:-len(suffix)]
                break
        parts = core_name.split("_")
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
        
        # Finally add 'smoothed: True/False' to STLs
        smoothed = None # No smoothing status unless it's an STL
        if name.endswith(".stl"):
            smoothed = not name.endswith("_ns.stl")  # Smoothed STL if it does NOT end in _ns.stl
        if smoothed is not None:
            metadata["smoothed"] = smoothed

        return metadata

            
    def mesh_to_tensor(self, mesh):
        """Convert a Trimesh object to a tensor representation."""
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)  # (V, 3) shape
        faces = torch.tensor(mesh.faces, dtype=torch.int64)  # (F, 3) shape
        smoothed = bool(mesh.smoothed)
        return vertices, faces, smoothed