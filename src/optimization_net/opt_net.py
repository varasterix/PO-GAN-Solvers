import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizationNet(nn.Module):
    def __init__(self):
        super(OptimizationNet, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)

        params = self.parameters()
        self.optimizer = torch.optim.SGD(params, lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x