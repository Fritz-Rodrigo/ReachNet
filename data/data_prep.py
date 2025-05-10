import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np
import os

class data_prep(Dataset):
    def __init__(self, sf_dir, bup_dir, stl_dir):
        self.sf_dir = sf_dir
        self.bup_dir = bup_dir
        self.stl_dir = stl_dir

        self.file_triples = []
        for f in os.listdir(sf_dir):
            if not f.endswith("_sf.txt"):
                continue
            base_name = f.replace("_sf.txt", "")
            bup_filename = f"{base_name}_bup.txt"
            stl_filename = f"{base_name}_ns.stl"
            bup_path = os.path.join(bup_dir, bup_filename)
            stl_path = os.path.join(stl_dir, stl_filename)
            sf_path = os.path.join(sf_dir, f)
            if os.path.exists(bup_path) and os.path.exists(stl_path):
                self.file_triples.append((sf_path, bup_path, stl_path, base_name))
            else:
                missing = []
                if not os.path.exists(bup_path):
                    missing.append("bup")
                if not os.path.exists(stl_path):
                    missing.append("stl")
                print(f"[Warning] Missing {', '.join(missing)} for {f}, skipping. → Checked: bup='{bup_path}', stl='{stl_path}'")

    def __len__(self):
        return len(self.file_triples)

    def __getitem__(self, idx):
        sf_path, bup_path, stl_path, base_name = self.file_triples[idx]

        # Load scalar field
        sf = np.loadtxt(sf_path)
        side_len = round(sf.size**(1/3))
        if side_len**3 != sf.size:
            raise ValueError(f"Cannot reshape scalar field {sf_path} to cube.")
        sf = sf.reshape((side_len, side_len, side_len))
        sf_tensor = torch.tensor(sf, dtype=torch.float32).unsqueeze(0)

        # Load reach
        reach = np.loadtxt(bup_path)
        reach_tensor = torch.tensor(reach, dtype=torch.float32)

        # Load mesh and convert to tensor
        stl_mesh = trimesh.load(stl_path, force='mesh')
        vertices, faces = self.mesh_to_tensor(stl_mesh)

        # Parse metadata from base name
        metadata = self.parse_metadata(base_name)
        mesh = {'vertices': vertices, 'faces': faces}
        return sf_tensor, reach_tensor, mesh, base_name, metadata

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

    @staticmethod # To call parse_metadata without needing a full dataset object
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
                "knot_type": knot_type,
                "frequencies": freqs,
                "phases": phases,
                "sin_freq": sin_freq,
                "sin_offset": sin_offset,
                "sin_amp": sin_amp,
                "rmin": rmin,
                "rmax": rmax,
                "voxels": voxels
            }

        except Exception as e:
            print(f"[Warning] Failed to parse metadata from {filename}: {e}")
            return {}

            
    def mesh_to_tensor(self, mesh):
        """Convert a Trimesh object to a tensor representation."""
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)  # (V, 3) shape
        faces = torch.tensor(mesh.faces, dtype=torch.int64)  # (F, 3) shape
        return vertices, faces