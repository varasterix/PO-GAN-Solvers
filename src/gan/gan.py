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


generator = Generator()
discriminator = Discriminator()

epochs = 100
dataset = []
for i in range(2000):
    dataset.append(databaseTools.read_tsp_heuristic_solution_file(10, i))

d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.001)
g_optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)

for epoch in range(epochs):
    avg_g_loss = 0.
    avg_d_loss = 0.
    avg_fake_d_loss = 0.
    batch = dataset[:]
    for data in batch:

        # Discriminator training
        d_optimizer.zero_grad()

        wm = data[0].get_weight_matrix().reshape(100)
        wm = torch.tensor(wm, dtype=torch.float, requires_grad=True)

        can_solver = data[0].get_candidate()
        binary_can_solver = [0 for k in range(100)]
        for k in range(len(can_solver)):
            binary_can_solver[k*10+can_solver[k]] = 1
        label = [0]
        binary_can_solver = torch.tensor(binary_can_solver, dtype=torch.float, requires_grad=True)

        input_d = binary_can_solver
        output_d = torch.tensor(label, dtype=torch.float, requires_grad=False)
        predicted_output_d = discriminator(input_d)

        valid_d_loss = F.binary_cross_entropy(predicted_output_d, output_d)

        can_gen = generator(wm)
        label = [1]

        input_d = can_gen
        output_d = torch.tensor(label, dtype=torch.float, requires_grad=False)
        predicted_output_d = discriminator(input_d)

        fake_d_loss = F.binary_cross_entropy(predicted_output_d, output_d)
        d_loss = (valid_d_loss + fake_d_loss) / 2
        avg_d_loss += d_loss.item()

        d_loss.backward()
        d_optimizer.step()

        # Generator training
        label = [0]
        g_optimizer.zero_grad()
        input_g = can_gen
        output_g = torch.tensor(label, dtype=torch.float, requires_grad=False)
        predicted_output_g = discriminator(torch.reshape(torch.tensor(F.one_hot(torch.argmax(
            torch.reshape(input_g.detach(), (10, 10)), dim=1), num_classes=10), dtype=torch.float), (100,)))

        g_loss = F.binary_cross_entropy(predicted_output_g, output_g)
        avg_g_loss += g_loss.item()

        g_loss.backward()
        g_optimizer.step()
    avg_d_loss /= 2000
    avg_g_loss /= 2000
    print("epoch: " + str(epoch + 1) + ": average loss of generator: " + str(avg_g_loss))
    print("epoch: " + str(epoch + 1) + ": average loss of discriminator: " + str(avg_d_loss))
