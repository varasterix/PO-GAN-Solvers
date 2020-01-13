import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def max_over_columns(input):
    nb_columns = int(np.sqrt(input.detach().numpy().shape[0]))
    for column in range(nb_columns):
        index_min = column * nb_columns
        index_max = (column + 1) * nb_columns
        index_max_value = input.detach().numpy()[index_min:index_max].argmax() + index_min
        for index in range(index_min, index_max):
            if index == index_max_value:
                input.detach().numpy()[index] = 1
                print("yo")
            else:
                input.detach().numpy()[index] = 0
    return input


def max_over_columns_opt_net(input):
    nb_columns = int(np.sqrt(input.detach().numpy().shape[1]))
    for column in range(nb_columns):
        index_min = column * nb_columns
        index_max = (column + 1) * nb_columns
        index_max_value = input.detach().numpy()[0][index_min:index_max].argmax() + index_min
        for index in range(index_min, index_max):
            if index == index_max_value:
                input.detach().numpy()[0][index] = 1
            else:
                input.detach().numpy()[0][index] = 0
    return input


if __name__ == '__main__':
    input = torch.randn(1)
    x = F.relu(nn.Linear(1, 25)(input))
    print(x.detach().numpy())
    max_over_columns(x)
    print(x.detach().numpy())
