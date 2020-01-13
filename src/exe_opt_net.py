import os
import random
import numpy as np
import torch
from torch.autograd import Variable
# import torchvision
import torch.nn.functional as F
# from src.fashion import FashionMNIST
# import torchvision.transforms as transforms
# from torch import nn
# from torch import optim
from src.database.databaseTools import read_tsp_heuristic_solution_file
from src.optimization_net.opt_net import OptimizationNet
from src.objects.orderedPathBinaryMatrix import OrderedPathBinaryMatrix
from src.objects.objectsTools import normalize_weight_matrix


# Parameters
tsp_database_path = "../data/tsp_files/"
train_proportion = 0.8
valid_proportion = 0.1
test_proportion = 0.1

# Execution script
tsp_database_files = [file_name for file_name in os.listdir(tsp_database_path)
                      if file_name.split('.')[1] == 'heuristic']
random.shuffle(tsp_database_files)
tsp_database_size = len(tsp_database_files)

bound_1 = round(train_proportion * tsp_database_size)
bound_2 = round((train_proportion + valid_proportion) * tsp_database_size)
train_data = tsp_database_files[0:bound_1]
valid_data = tsp_database_files[bound_1:bound_2]
test_data = tsp_database_files[bound_2:]


def train(model, train_set):
    model.train()
    for index, data_file in enumerate(train_set):
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        input_data = Variable(torch.tensor(normalize_weight_matrix(weight_matrix).reshape((1, nb_cities*nb_cities)),
                                           dtype=torch.float))
        model.optimizer.zero_grad()
        output = model(input_data)  # calls the forward function
        candidate = OrderedPathBinaryMatrix(np.array(output.detach(), dtype=int)
                                            .reshape((nb_cities, nb_cities)).transpose(), weight_matrix)
        result = torch.tensor([int(candidate.is_solution())], dtype=torch.float, requires_grad=True)
        target = torch.tensor([1], dtype=torch.float, requires_grad=True)

        loss = model.loss_function(result, target)

        # model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    return model


def valid(model, valid_set):
    model.eval()
    valid_loss = 0
    correct = 0
    for index, data_file in enumerate(valid_set):
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        input_data = Variable(torch.tensor(normalize_weight_matrix(weight_matrix).reshape((1, nb_cities * nb_cities)),
                                           dtype=torch.float))
        output = model(input_data)
        candidate = OrderedPathBinaryMatrix(np.array(output.detach(), dtype=int)
                                            .reshape((nb_cities, nb_cities)).transpose(), weight_matrix)
        result = Variable(torch.tensor([int(candidate.is_solution())], dtype=torch.float, requires_grad=True))
        target = Variable(torch.tensor([1], dtype=torch.float, requires_grad=True))
        valid_loss += model.loss_function(result, target)  # sum up batch loss
        pred = result.data.max(0, keepdim=True)[1].type(torch.float)  # get the index of the max log-probability
        correct += (pred.eq(target.data.view_as(pred))).cpu().sum()

    valid_set_size = len(valid_set)
    valid_loss /= valid_set_size
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, valid_set_size, 100. * correct / valid_set_size))
    return correct / valid_set_size


def test(model, test_set):
    model.eval()
    test_loss = 0
    correct = 0
    for index, data_file in enumerate(test_set):
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        input_data = Variable(torch.tensor(normalize_weight_matrix(weight_matrix).reshape((1, nb_cities * nb_cities)),
                                           dtype=torch.float))
        output = model(input_data)
        candidate = OrderedPathBinaryMatrix(np.array(output.detach(), dtype=int)
                                            .reshape((nb_cities, nb_cities)).transpose(), weight_matrix)
        result = torch.tensor([int(candidate.is_solution())], dtype=torch.float, requires_grad=True)
        target = torch.tensor([1], dtype=torch.float, requires_grad=True)

        test_loss += model.loss_function(result, target)  # sum up batch loss
        pred = result.data.max(0, keepdim=True)[1].type(torch.float)  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_set_size = len(test_set)
    test_loss /= test_set_size
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_set_size, 100. * correct / test_set_size))


def experiment(model, epochs=50, lr=0.001):
    best_precision = 0
    best_model = model
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model = train(model, train_data)  # optimizer)
        precision = valid(model, valid_data)

        if precision > best_precision:
            best_precision = precision
            best_model = model
    return best_model, best_precision


Models = [OptimizationNet()]
best_precision_ = 0
best_model_ = Models[0]
for model_ in Models:  # add your models in the list
    model_, precision_ = experiment(model_)
    if precision_ > best_precision_:
        best_precision_ = precision_
        best_model_ = model_

test(best_model_, test_data)

"""
train_data = FashionMNIST('../data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
valid_data = FashionMNIST('../data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))

train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)

train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

mask = np.ones(60000)
mask[train_idx] = 0

valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]

print(len(train_data))
print(train_data)
print(train_data[0])

batch_size = 100
test_batch_size = 100
train_loader_ = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader_ = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader_ = torch.utils.data.DataLoader(FashionMNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=test_batch_size, shuffle=True)


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return model


def valid_(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return correct / len(valid_loader.dataset)


def test_(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def experiment_(model, epochs=50, lr=0.001):
    best_precision = 0
    best_model = model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model = train(model, train_loader_, optimizer)
        precision = valid(model, valid_loader_)

        if precision > best_precision:
            best_precision = precision
            best_model = model
    return best_model, best_precision


Models = [OptimizationNet()]
best_precision_ = 0
best_model_ = Models[0]
for model_ in Models:  # add your models in the list
    model_, precision_ = experiment(model_)
    if precision_ > best_precision_:
        best_precision_ = precision_
        best_model_ = model_

test(best_model_, test_loader_)
"""
