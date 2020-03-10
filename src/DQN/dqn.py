import torch
import torch.nn as nn
import torch.nn.functional as F

p = 0.9  # dropout


class DQN(nn.Module):

    def __init__(self, distance_matrix, nb_cities):
        super(DQN, self).__init__()
        matrix_size = len(distance_matrix)
        self.fc1 = nn.Linear(matrix_size + nb_cities, 2 * matrix_size)
        self.bn1 = nn.BatchNorm1d(2 * matrix_size)
        self.fc2 = nn.Linear(2 * matrix_size, 2 * matrix_size)
        self.bn2 = nn.BatchNorm1d(2 * matrix_size)
        self.fc3 = nn.Linear(2 * matrix_size, nb_cities)
        self.bn3 = nn.BatchNorm1d(nb_cities)

    def forward(self, x):
        model = torch.nn.Sequential(self.fc1, nn.Dropout(p), nn.ReLU(),
                                    self.fc2, nn.Dropout(p), nn.ReLU(),
                                    self.fc3, nn.Dropout(p), nn.ReLU())
        return model(x)


class Environment:

    def __init__(self, distance_matrix, nb_cities):
        self.__nb_cities = nb_cities
        self.__distance_matrix = distance_matrix
        self.__current_city = 0
        self.__visited_cities = [1.] + [0. for i in range(self.__nb_cities - 1)]

    def get_nb_cities(self):
        return self.__nb_cities

    def get_distance_matrix(self):
        return self.__distance_matrix

    def get_visited_cities(self):
        return self.__visited_cities

    def get_current_city(self):
        return self.__current_city

    def set_next_city(self, city):
        self.__visited_cities[city] += 1.
        self.__current_city = city

    def step(self, action):
        done = False
        if self.get_visited_cities()[action] >= 1.:
            reward = -1
        else:
            reward = 1 / self.get_distance_matrix()[self.get_current_city() * self.get_nb_cities() + action]
        if sum(self.get_visited_cities()) == self.get_nb_cities():
            done = True
        return reward, done
