import numpy as np
import torch
import torch.nn.functional as F
from src.generator.generator import Generator
from src.discriminator.discriminator import Discriminator
from src.discriminator.checker import nb_cycles
from src.objects import neighboursBinaryMatrix, neighbours
from src.activation.max_over_columns import max_over_columns


class GAN:
    def __init__(self):
        self.input_shape = 10

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.loss_function = torch.nn.MSELoss()

    def train(self, epochs=1, batch_size=128):
        weightmatrix = np.random.rand(10, 10)
        for epoch in range(epochs):
            input = torch.randn(1)
            output = self.generator(input)
            nbm = neighboursBinaryMatrix.NeighboursBinaryMatrix(np.array(output.detach(),
                                                                dtype=int).reshape((10, 10)).transpose(),
                                                                weightmatrix)
            neighbor_candidate = nbm.to_neighbours().get_candidate()
            nb_cycle_pred = torch.tensor([nb_cycles(neighbor_candidate)], dtype=torch.float)
            nb_cycle = torch.tensor([1], dtype=torch.float)

            loss = self.loss_function(nb_cycle_pred, nb_cycle)

            self.generator.optimizer.zero_grad()
            loss.backward()
            self.generator.optimizer.step()


gan = GAN()
gan.train()
