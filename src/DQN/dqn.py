import torch
import torch.nn as nn
import torch.nn.functional as F


p = 0.6  # dropout


class DQN(nn.module):

    def __init__(self, weight_matrix):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(100, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        model = torch.nn.Sequential(self.fc1, self.bn1, F.dropout(p), F.relu(),
                                    self.fc2, self.bn2, F.dropout(p), F.relu(),
                                    self.fc3, self.bn3, F.dropout(p), F.relu())
        return model(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.bn3(self.fc3(x)))
        # return x
