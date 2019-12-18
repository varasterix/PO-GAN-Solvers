import torch.nn.functional as F
import torch.nn as nn
import torch
from src.activation.max_over_columns import max_over_columns


# class of the model
class Generator(nn.Module):
    # the definition of the layers
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 64)
        self.fc3 = nn.Linear(64, 100)

        params = self.parameters()
        self.optimizer = torch.optim.SGD(params, lr=0.001)

    # the definition of the activation functions.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = max_over_columns(self.fc3(x))
        return x
