import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data.data_prep import data_prep

def visualize_batch(data_path_sf, data_path_bup, batch_index=0, batch_size=2, train=True):
    dataset = data_prep(data_path_sf, data_path_bup)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    # Convert to list so we can index batches
    all_batches = list(dataloader)
    print(f'Number of batches: {len(all_batches)}')
    
    if batch_index >= len(all_batches):
        raise IndexError(f"Batch index {batch_index} out of range.")
    
    sf_batch, bup_batch = all_batches[batch_index]
    
    for i in range(batch_size):
        sf = sf_batch[i][0].numpy()  # (D, H, W)
        bup = bup_batch[i].numpy()   # (N,)

        # Show a middle slice (for example, slice along depth axis)
        mid_slice = sf[sf.shape[0]//2]
        # mid_slice = sf[sf.shape[0]//4 + 4*i] # Slices other than mid-slice
        
        plt.figure(figsize=(12, 4))
        
        # Scalar field slice
        plt.subplot(1, 2, 1)
        plt.imshow(mid_slice, cmap='coolwarm', origin='lower')
        plt.title(f"Scalar Field Slice (mid-depth) - Sample {i}")
        plt.colorbar()

        # Blow-up curve
        plt.subplot(1, 2, 2)
        plt.plot(bup)
        plt.title(f"Radii - Sample {i}")
        plt.xlabel("Point index along curve")
        plt.ylabel("Radius")

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    visualize_batch("../g7_v100/sf", "../g7_v100/bup", batch_index=0, batch_size=2, train=False)