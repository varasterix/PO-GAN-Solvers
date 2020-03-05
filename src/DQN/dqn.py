import torch
import torch.nn as nn

# import torch.nn.functional as F

p = 0.9  # dropout


class DQN(nn.module):

    def __init__(self, weight_matrix):
        super(DQN, self).__init__()
        matrix_length = len(weight_matrix)
        matrix_size = matrix_length ** 2
        self.fc1 = nn.Linear(matrix_size, 2 * matrix_size)
        self.bn1 = nn.BatchNorm1d(2 * matrix_size)
        self.fc2 = nn.Linear(2 * matrix_size, 2 * matrix_size)
        self.bn2 = nn.BatchNorm1d(2 * matrix_size)
        self.fc3 = nn.Linear(2 * matrix_size, matrix_length)
        self.bn3 = nn.BatchNorm1d(matrix_length)

    def forward(self, x):
        model = torch.nn.Sequential(self.fc1, self.bn1, nn.Dropout(p), nn.ReLU(),
                                    self.fc2, self.bn2, nn.Dropout(p), nn.ReLU(),
                                    self.fc3, self.bn3, nn.Dropout(p), nn.ReLU())
        return model(x)
