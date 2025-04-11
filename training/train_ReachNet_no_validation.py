import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ReachNet import ReachNet
from data.dataset import ScalarFieldToReachDataset
import numpy as np
import os

def train_model(data_path_sf, data_path_bup, epochs=10, batch_size=2, lr=1e-3):
    # Determine output_size from one bup file
    bup_files = sorted(os.listdir(data_path_bup))
    if not bup_files:
        raise ValueError("No bup files found in the folder")
    sample_bup_path = os.path.join(data_path_bup, bup_files[0])
    sample_bup = np.loadtxt(sample_bup_path)
    output_size = sample_bup.shape[0]
    print(f"Determined output size (number of curve points): {output_size}")

    # Dataset & DataLoader
    dataset = ScalarFieldToReachDataset(data_path_sf, data_path_bup)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model with the determined output size
    model = ReachNet(output_size)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_batches = len(dataloader)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (sf, bup) in enumerate(dataloader):
            # Move both to the selected device
            sf   = sf.to(device, dtype=torch.float32)   # [B,1,D,H,W] on GPU
            bup  = bup.to(device, dtype=torch.float32)  # [B,output_size] on GPU

            optimizer.zero_grad()
            output = model(sf)

            if output.shape != bup.shape:
                raise ValueError(f"Shape mismatch: output {output.shape}, target {bup.shape}")

            loss = criterion(output, bup)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{total_batches}, Batch Loss: {loss.item():.6f}", end="\r")

        avg_loss = epoch_loss / total_batches
        rmse = np.sqrt(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], MSE Loss: {avg_loss:.6f}, RMSE Loss: {rmse:.6f}")

    # Save model
    torch.save(model.state_dict(), "trained_ReachNet.pth")
    print("✅ Model saved to 'trained_ReachNet.pth'.")

# Example usage (if running as script)
if __name__ == "__main__":
    train_model("../g7_v100/sf", "../g7_v100/bup", epochs=10)
