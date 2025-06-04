import torch
import torch.nn as nn

class MERcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # 0
            nn.ReLU(),                                              # 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 2
            nn.ReLU(),                                              # 3
            nn.MaxPool2d(2, 2),                                     # 4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 5
            nn.ReLU(),                                              # 6
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# 7
            nn.ReLU(),                                              # 8
            nn.MaxPool2d(2, 2),                                     # 9

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# 10
            # nn.ReLU(),                                              # 11
            # nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),# 12
            # nn.ReLU(),                                              # 13
            # nn.MaxPool2d(2, 2),                                     # 14

            nn.Flatten(),                                           # 15
            nn.Linear(256 * 10 * 10, 1024),                         # 16
            nn.ReLU(),                                              # 17
            nn.Linear(1024, 512),                                   # 18
            nn.ReLU(),                                              # 19
            nn.Linear(512, 7)                                       # 20
        )

    def forward(self, x):
        return self.network(x)

# Load model
model = MERcnn()
checkpoint = torch.load("MERcnn.pth", map_location='cpu')

# Try loading with strict=False and print mismatches
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

# Set model to eval
model.eval()
