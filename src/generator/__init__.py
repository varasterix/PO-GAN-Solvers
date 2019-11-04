import torch.nn.functional as F
import torch.nn as nn
import torch


# class of the model
class Generator(nn.Module):
    # the definition of the layers, with perceptrons
    def __init__(self):
        super(Generator).__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 56)
        self.fc3 = nn.Linear(56, 128)
        self.fc4 = nn.Linear(128, 50)

    # the definition of the activation fuctions.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


generator = Generator()

input = torch.randint(1, 51, (50, ))
output = generator(input)
