import numpy as np
import torch
import torch.nn.functional as F
from src.generator.generator import Generator
from src.discriminator.discriminator import Discriminator
from src.database import databaseTools
import random


def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


class GAN:
    def __init__(self):
        self.input_shape = 10

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.loss_function = torch.nn.MSELoss()

    def train(self, epochs=1000, batch_size=100):
        dataset = []
        for i in range(2000):
            dataset.append(databaseTools.read_tsp_heuristic_solution_file(10, i))

        b = 0
        for epoch in range(epochs):
            batch = dataset[b*batch_size:(b+1)*batch_size]
            for data in batch:

                # Discriminator training
                self.discriminator.optimizer.zero_grad()

                wm = data[0].get_weight_matrix().reshape(100)
                wm = torch.tensor(wm, dtype=torch.float, requires_grad=True)

                can_solver = data[0].get_candidate()
                binary_can_solver = [0 for k in range(100)]
                for k in range(len(can_solver)):
                    binary_can_solver[k*10+can_solver[k]] = 1
                label = [0]
                binary_can_solver = torch.tensor(binary_can_solver, dtype=torch.float, requires_grad=True)

                input_d = torch.tensor(torch.cat((binary_can_solver, wm), 0), dtype=torch.float, requires_grad=True)
                output_d = torch.tensor(label, dtype=torch.float, requires_grad=True)
                predicted_output_d = self.discriminator(input_d)

                d_loss = self.loss_function(output_d, predicted_output_d)

                d_loss.backward()
                self.discriminator.optimizer.step()

                self.discriminator.optimizer.zero_grad()
                can_gen = self.generator(wm)
                label = [1]

                input_d = torch.tensor(torch.cat((can_gen, wm), 0), dtype=torch.float, requires_grad=True)
                output_d = torch.tensor(label, dtype=torch.float, requires_grad=True)
                predicted_output_d = self.discriminator(input_d)

                d_loss = self.loss_function(output_d, predicted_output_d)

                d_loss.backward()
                self.discriminator.optimizer.step()

                # Generator training
                self.generator.optimizer.zero_grad()
                input_g = torch.cat((can_gen, wm), 0)
                output_g = torch.tensor(label, dtype=torch.float, requires_grad=True)
                predicted_output_g = self.discriminator(input_g)

                g_loss = self.loss_function(output_g, predicted_output_g)

                g_loss.backward()
                self.generator.optimizer.step()

                if epoch == epochs:
                    b = 0
                else:
                    b += 1


gan = GAN()
gan.train()
