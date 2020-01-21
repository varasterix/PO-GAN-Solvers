import torch.nn.functional as F
import torch.nn as nn
import torch


# class of the model
class Generator(nn.Module):
    # the definition of the layers
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 100)

        params = self.parameters()
        self.optimizer = torch.optim.SGD(params, lr=0.01)

    # the definition of the activation functions.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(F.softmax(torch.reshape(self.fc3(x), (10, 10)), dim=1), (100,))
        return x
