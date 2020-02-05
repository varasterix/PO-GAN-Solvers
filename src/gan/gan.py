import numpy as np
import torch
import torch.nn.functional as F
from src import constants
from src.generator.generator import Generator
from src.discriminator.discriminator import Discriminator
from src.database import databaseTools
import random


def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    epochs = 100
    dataset = []
    for i in range(2000):
        dataset.append(databaseTools.read_tsp_choco_solution_file(10, i, path="../../" +
                                                                              constants.PARAMETER_TSP_CHOCO_DATA_FILES))

    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.001)
    g_optimizer = torch.optim.SGD(generator.parameters(), lr=0.001)

    first_g_loss, last_g_loss = None, None
    first_d_loss, last_d_loss = None, None
    net_switch = False
    for epoch in range(epochs):
        avg_g_loss = 0.
        avg_d_loss = 0.
        avg_fake_d_loss = 0.
        batch = dataset[:]
        trained_net = 0
        nb_g_loss = 0
        nb_d_loss = 0
        for data in batch:
            wm = data[0].get_weight_matrix().reshape(100)
            wm = torch.tensor(wm, dtype=torch.float, requires_grad=True)

            if trained_net == 0:
                # Generator training
                label = [0]
                g_optimizer.zero_grad()
                input_g = generator(wm)
                output_g = torch.tensor(label, dtype=torch.float, requires_grad=False)
                predicted_output_g = discriminator(input_g)

                g_loss = F.binary_cross_entropy(predicted_output_g, output_g)
                avg_g_loss += g_loss.item()
                nb_g_loss += 1

                if net_switch or first_g_loss is None:
                    first_g_loss = g_loss.item()
                    if last_g_loss is None:
                        last_g_loss = first_g_loss
                else:
                    last_g_loss = g_loss.item()
                net_switch = last_g_loss < 0.995 * first_g_loss
                if net_switch:
                    print("switch")
                    trained_net = 1 - trained_net

                g_loss.backward()
                g_optimizer.step()

            else:
                # Discriminator training
                d_optimizer.zero_grad()

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
                predicted_output_d = discriminator(input_d.detach())

                fake_d_loss = F.binary_cross_entropy(predicted_output_d, output_d)
                d_loss = (valid_d_loss + fake_d_loss) / 2
                avg_d_loss += d_loss.item()
                nb_d_loss += 1

                if net_switch or first_d_loss is None:
                    first_d_loss = d_loss.item()
                    if last_d_loss is None:
                        last_d_loss = first_d_loss
                else:
                    last_d_loss = d_loss.item()
                net_switch = last_d_loss < 0.99 * first_d_loss
                if net_switch:
                    print("switch")
                    trained_net = 1 - trained_net

                d_loss.backward()
                d_optimizer.step()

        avg_d_loss /= max(1, nb_d_loss)
        avg_g_loss /= max(1, nb_g_loss)
        print("epoch: " + str(epoch + 1) + ": average loss of generator: " + str(avg_g_loss))
        print("epoch: " + str(epoch + 1) + ": average loss of discriminator: " + str(avg_d_loss))
