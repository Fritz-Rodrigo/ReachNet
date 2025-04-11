# dataset_reach.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ScalarFieldToReachDataset(Dataset):
    def __init__(self, sf_folder, bup_folder):
        self.sf_folder = sf_folder
        self.bup_folder = bup_folder

        # Match only scalar fields that have corresponding _bup.txt
        self.file_pairs = []
        for f in os.listdir(sf_folder):
            if not f.endswith("_sf.txt"):
                continue
            base_name = f.replace("_sf.txt", "")
            bup_filename = f"{base_name}_bup.txt"
            bup_path = os.path.join(bup_folder, bup_filename)
            sf_path = os.path.join(sf_folder, f)
            if os.path.exists(bup_path):
                self.file_pairs.append((sf_path, bup_path))
            else:
                print(f"[Warning] Missing bup for {f}, skipping.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        sf_path, bup_path = self.file_pairs[idx]

        # Load and reshape scalar field
        sf = np.loadtxt(sf_path)
        flat_size = sf.size
        side_len = round(flat_size ** (1/3))
        if side_len**3 != flat_size:
            raise ValueError(f"Cannot reshape scalar field {sf_path} to cube.")
        sf = sf.reshape((side_len, side_len, side_len))
        sf_tensor = torch.tensor(sf, dtype=torch.float32).unsqueeze(0)  # Shape: [1, D, H, W]

        # Load reach (bup file)
        reach = np.loadtxt(bup_path)
        reach_tensor = torch.tensor(reach, dtype=torch.float32)  # Shape: [N]

        return sf_tensor, reach_tensor
