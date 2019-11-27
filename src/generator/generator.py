import torch.nn.functional as F
import torch.nn as nn


# class of the model
class Generator(nn.Module):
    # the definition of the layers
    def __init__(self):
        super(Generator).__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 400)

    # the definition of the activation functions.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x
