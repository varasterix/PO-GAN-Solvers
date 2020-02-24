import torch
import torch.nn as nn
import torch.nn.functional as F


# The class of the model
class SegNet(nn.Module):
    # the definition of the layers
    def __init__(self, nb_cities):
        super(SegNet, self).__init__()
        self.name = "SegNet"
        self.nb_cities = nb_cities
        square_nb_cities = nb_cities * nb_cities

        self.fc1 = nn.Linear(square_nb_cities+nb_cities*2, 2 * square_nb_cities)
        self.fc2 = nn.Linear(2 * square_nb_cities, 2 * square_nb_cities)
        self.fc3 = nn.Linear(2 * square_nb_cities, nb_cities)
        self.loss_function = torch.nn.MSELoss()

    # the definition of the activation functions
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
