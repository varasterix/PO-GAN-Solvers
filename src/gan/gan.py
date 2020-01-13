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

    def train(self, epochs=1000, batch_size=100):
        dataset = []
        for i in range(2000):
            dataset.append(databaseTools.read_tsp_heuristic_solution_file(10, i))

        for epoch in range(epochs):
            wm = []  # weight matrix
            can = []  # candidate
            for j in range(20):
                wm += [dataset[k][0].get_weight_matrix() for k in range(j*batch_size, (j+1)*batch_size)]
                can += [dataset[k][0].get_candidate() for k in range(j*batch_size, (j+1)*batch_size)]


gan = GAN()
gan.train()
