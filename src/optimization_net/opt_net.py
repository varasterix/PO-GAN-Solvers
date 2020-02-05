import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.activation.max_over_columns import max_over_columns_opt_net


# The class of the model
class OptNet(nn.Module):
    # the definition of the layers
    def __init__(self, nb_cities):
        super(OptNet, self).__init__()
        self.name = "OptNet"
        self.nb_cities = nb_cities
        square_nb_cities = nb_cities * nb_cities

        self.fc1 = nn.Linear(square_nb_cities, 2 * square_nb_cities)
        self.fc2 = nn.Linear(2 * square_nb_cities, 2 * square_nb_cities)
        self.fc3 = nn.Linear(2 * square_nb_cities, square_nb_cities)
        self.loss_function = torch.nn.MSELoss()

    # the definition of the activation functions
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(torch.reshape(self.fc3(x), (self.nb_cities, self.nb_cities)), dim=1)  # torch.float32
        # x = max_over_columns_opt_net(self.fc2(x))
        # There are no networks that do ordinary backprop through argmax
        # https://www.reddit.com/r/MachineLearning/comments/4e2get/argmax_differentiable/
        # x = torch.argmax(x, dim=1)  # dtype=torch.int64
        return x
