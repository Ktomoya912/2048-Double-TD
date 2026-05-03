import torch
import torch.nn as nn

# Test CNN_DEEP_MULTI architecture
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        chw0, chw1, chw2, chw3, chw4, chw5 = 64, 128, 232, 256, 256, 256
        self.conv0 = nn.Conv2d(11, chw0, kernel_size=1)
        self.conv1 = nn.Conv2d(chw0, chw1, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(chw1, chw2, kernel_size=2)
        self.conv3 = nn.Conv2d(chw2, chw3, kernel_size=2)
        self.conv4 = nn.Conv2d(chw3, chw4, kernel_size=2)
        self.fc1 = nn.Linear(chw4, chw5)
        self.value_head = nn.Linear(chw5, 1)
        self.policy_head = nn.Linear(chw5, 4)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 11, 3, 3)
        print(f"After reshape: {x.shape}")

        x = torch.relu(self.conv0(x))
        print(f"After conv0: {x.shape}")

        x = torch.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")

        x = torch.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")

        x = torch.relu(self.conv3(x))
        print(f"After conv3: {x.shape}")

        x = torch.relu(self.conv4(x))
        print(f"After conv4: {x.shape}")
        print(f"Conv4 actual flatten would be: {x.view(x.shape[0], -1).shape}")

        x = x.view(-1, 256)
        print(f"After hardcoded flatten to 256: {x.shape}")

        features = torch.relu(self.fc1(x))
        print(f"After fc1: {features.shape}")

        value = self.value_head(features)
        policy = self.policy_head(features)
        print(f"Value head: {value.shape}, Policy head: {policy.shape}")

        return value, policy

model = Model()
x = torch.randn(2, 99)
try:
    value, policy = model(x)
    print("\nTest passed!")
except RuntimeError as e:
    print(f"\nError: {e}")
