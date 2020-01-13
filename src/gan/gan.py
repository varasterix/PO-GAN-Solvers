import numpy as np
import torch
import torch.nn.functional as F
from src.generator.generator import Generator
from src.discriminator.discriminator import Discriminator
from src.discriminator.checker import nb_cycles
from src.objects import neighboursBinaryMatrix, neighbours
from src.activation.max_over_columns import max_over_columns
from src.database import databaseTools


class GAN:
    def __init__(self):
        self.input_shape = 10

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.loss_function = torch.nn.CrossEntropyLoss(

        )

    def train(self, epochs=1000, batch_size=128):
        dataset = []
        for i in range(2000):
            dataset.append(databaseTools.read_tsp_heuristic_solution_file(10, i))

        for epoch in range(epochs):
            avg_loss = 0
            for iter in range(batch_size):
                if iter % 2 == 0:

                    input = torch.randn(1, requires_grad=True)
                    output = self.generator(input)
                    nbm = neighboursBinaryMatrix.NeighboursBinaryMatrix(np.array(output.detach(),
                                                                        dtype=int).reshape((10, 10)).transpose(),
                                                                        dataset)
                    nb_cycle_pred = torch.tensor([nbm.get_nb_cycles()], dtype=torch.float, requires_grad=True)
                    nb_cycle = torch.tensor([1], dtype=torch.float, requires_grad=True)

                    loss = self.loss_function(nb_cycle_pred, nb_cycle)
                    avg_loss += loss

                    self.generator.optimizer.zero_grad()
                    loss.backward()
                    self.generator.optimizer.step()
            avg_loss /= batch_size
            print("average loss at epoch " + str(epoch) + ": " + str(avg_loss))
        return


gan = GAN()
gan.train()
