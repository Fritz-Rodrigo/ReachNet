import torch.nn as nn
import torch.nn.functional as F

class ReachNet(nn.Module):
    def __init__(self, output_size):
        super(ReachNet, self).__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # down to (32, 4, 4, 4)
        self.fc1 = nn.Linear(32 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (1, D, H, W)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)            # (32, 4, 4, 4)
        x = x.view(x.size(0), -1)   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # shape: (batch_size, output_size)
