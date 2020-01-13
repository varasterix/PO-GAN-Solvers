import torch
import torch.nn as nn
import torch.nn.functional as F
from src.activation.max_over_columns import max_over_columns_opt_net


# The class of the model
class OptimizationNet(nn.Module):
    # the definition of the layers
    def __init__(self):
        super(OptimizationNet, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)

        params = self.parameters()
        self.optimizer = torch.optim.SGD(params, lr=0.001)
        self.loss_function = torch.nn.MSELoss()

    # the definition of the activation functions
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = max_over_columns_opt_net(self.fc2(x))
        return x
