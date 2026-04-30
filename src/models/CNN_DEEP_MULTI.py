import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """CNN_DEEP backbone with shared features, value head and policy head.
    Input: 99-dim state (before action). Output: (value [N,1], pi_logits [N,4]).
    """

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 11, 3, 3)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256)
        features = F.relu(self.fc1(x))
        return self.value_head(features), self.policy_head(features)
