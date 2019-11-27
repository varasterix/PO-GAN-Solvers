import torch.nn.functional as F
import torch.nn as nn


# class of the model
class Generator(nn.Module):
    # the definition of the layers
    def __init__(self):
        super(Generator,self).__init__()
        #self.fc1 = nn.Linear(1, 12)
        #self.fc2 = nn.Linear(12, 64)
        self.fc1 = nn.Linear(111, 256) #distance matrice length=100 + visited cities length=10 + current city=1   ===111
        self.fc2 = nn.Linear(256, 400)
        self.fc3 = nn.Linear(400, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 10)

    # the definition of the activation functions.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x

    def init_weights(m):
        >> > print(m)

    >> > if type(m) == nn.Linear:
        >> > m.weight.data.fill_(1.0)
    >> > print(m.weight)