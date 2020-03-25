import os
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import optim
from src import constants
from src.utils import custom_formatwarning
from src.database.databaseTools import read_tsp_choco_solution_file
from src.optimization_net.opt_net import OptNet
from src.objects.orderedPath import OrderedPath
from src.objects.objectsTools import normalize_weight_matrix


def build_target_from_output(output, nb_cities):
    np_output = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
    order_of_exploration = list(range(nb_cities))
    cities = [i for i in range(nb_cities)]
    random.shuffle(order_of_exploration)
    target = np.zeros((nb_cities, nb_cities), dtype=int)
    for i in order_of_exploration:
        city_output = np_output[i]
        if city_output in cities:
            cities.remove(city_output)
            target[city_output, i] = 1
    random.shuffle(cities)
    c = 0
    for j in range(nb_cities):
        if sum(target[:, j]) == 0:
            c += 1
            target[cities.pop(0), j] = 1
    return Variable(torch.tensor(target, dtype=torch.float), requires_grad=True)


def train(model, train_set, optimizer, tsp_database_path):
    model.train()
    sum_duplicates = 0
    sum_solution = 0
    random.shuffle(train_set)
    for data_file in train_set:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        input_data = Variable(torch.tensor(normalize_weight_matrix(weight_matrix).reshape((1, nb_cities * nb_cities)),
                                           dtype=torch.float), requires_grad=True)
        optimizer.zero_grad()
        output = model(input_data)  # calls the forward function

        target = build_target_from_output(output, nb_cities)
        candidate = OrderedPath(np.array(torch.argmax(output.detach(), dim=1), dtype=int), weight_matrix)
        sum_duplicates += candidate.get_nb_duplicates()
        sum_solution += int(candidate.is_solution())

        loss = model.loss_function(output, target)
        loss.backward()
        optimizer.step()

    train_set_size = len(train_set)
    print('Training indicators : Solution: {}, Nb duplicates: {}'.format(sum_solution / train_set_size,
                                                                         sum_duplicates / train_set_size))
    return model


def valid(model, valid_set, epoch, tsp_database_path):
    model.eval()
    valid_loss = 0
    correct = 0
    sum_duplicates = 0
    sum_solution = 0
    random.shuffle(valid_set)
    for data_file in valid_set:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        input_data = Variable(torch.tensor(normalize_weight_matrix(weight_matrix).reshape((1, nb_cities * nb_cities)),
                                           dtype=torch.float))
        output = model(input_data)

        target = build_target_from_output(output, nb_cities)
        candidate = OrderedPath(np.array(torch.argmax(output.detach(), dim=1), dtype=int).transpose(), weight_matrix)

        valid_loss += model.loss_function(output, target)  # sum up batch loss
        sum_duplicates += candidate.get_nb_duplicates()
        sum_solution += int(candidate.is_solution())
        correct += int(candidate.is_solution())

    valid_set_size = len(valid_set)
    sum_solution /= valid_set_size
    sum_duplicates /= valid_set_size
    print('Valid phase indicators : Solution: {}, Nb duplicates: {}'.format(sum_solution,
                                                                            sum_duplicates))
    valid_loss /= valid_set_size
    print('Epoch {} - valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, valid_loss, correct, valid_set_size, 100. * correct / valid_set_size))
    return correct / valid_set_size, valid_loss, sum_duplicates


def test(model, test_set, tsp_database_path):
    model.eval()
    test_loss = 0
    correct = 0
    random.shuffle(test_set)
    for data_file in test_set:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        input_data = Variable(torch.tensor(normalize_weight_matrix(weight_matrix).reshape((1, nb_cities * nb_cities)),
                                           dtype=torch.float))
        output = model(input_data)

        target = build_target_from_output(output, nb_cities)
        candidate = OrderedPath(np.array(torch.argmax(output.detach(), dim=1), dtype=int).transpose(), weight_matrix)

        test_loss += model.loss_function(output, target)  # sum up batch loss
        correct += int(candidate.is_solution())

    test_set_size = len(test_set)
    test_loss /= test_set_size
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_set_size, 100. * correct / test_set_size))


def experiment(model, epochs, lr, train_set, valid_set, tsp_database_path):
    best_precision = 0
    best_model = model
    accuracy_by_epochs = []
    valid_loss_by_epochs = []
    duplicates_by_epochs = []
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model = train(model, train_set, optimizer, tsp_database_path)
        precision, loss, duplicates = valid(model, valid_set, epoch, tsp_database_path)
        accuracy_by_epochs.append(precision)
        valid_loss_by_epochs.append(loss)
        duplicates_by_epochs.append(duplicates)
        if precision > best_precision:
            best_precision = precision
            best_model = model

    return best_model, best_precision, accuracy_by_epochs, valid_loss_by_epochs, duplicates_by_epochs


if __name__ == '__main__':
    # Parameters
    tsp_instances_database_path = "../../" + constants.PARAMETER_TSP_CHOCO_DATA_FILES
    number_cities = 10
    train_proportion = 0.8
    valid_proportion = 0.1
    test_proportion = 0.1
    over_fit_one_instance = False
    Models = [OptNet(number_cities)]  # add your models in the list
    nb_epochs = 20
    learning_rate = 0.001

    # Preparation of the TSP dataSet
    tsp_database_files = [file_name for file_name in os.listdir(tsp_instances_database_path)]
    random.shuffle(tsp_database_files)
    tsp_database_size = len(tsp_database_files)
    if over_fit_one_instance:
        tsp_database_files = [tsp_database_files[0]] * tsp_database_size

    bound_1 = round(train_proportion * tsp_database_size)
    bound_2 = round((train_proportion + valid_proportion) * tsp_database_size)
    train_data = tsp_database_files[0:bound_1]
    valid_data = tsp_database_files[bound_1:bound_2]
    test_data = tsp_database_files[bound_2:]

    # Experimentation of the different models on the TSP dataSet
    best_precision_ = 0
    best_model_ = Models[0]
    results = []
    for model_ in Models:
        model_, precision_, accuracy_by_epochs_, valid_loss_by_epochs_, duplicates_by_epochs_ = \
            experiment(model_, epochs=nb_epochs, lr=learning_rate, train_set=train_data,
                       valid_set=valid_data, tsp_database_path=tsp_instances_database_path)
        results.append((model_.name, accuracy_by_epochs_, valid_loss_by_epochs_, duplicates_by_epochs_))
        if precision_ > best_precision_:
            best_precision_ = precision_
            best_model_ = model_

    test(best_model_, test_data, tsp_instances_database_path)

    # PLOT SECTION
    # Warning section about the plot
    line_styles = ['-', '--', ':', '-.']
    nb_line_styles = len(line_styles)
    nb_results = len(results)
    if nb_results > nb_line_styles:
        warning_message = "[PLOT] Line styles conflict: The number of results to plot (" + str(nb_results) + \
                          ") is higher than the number of line styles used (" + str(nb_line_styles) + ")"
        warnings.formatwarning = custom_formatwarning
        warnings.warn(warning_message, Warning)

    # Validation results plot section
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epochs")
    plt.title("Validation loss, accuracy and TSP metric by epochs and models")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
    ax3.spines["right"].set_position(("axes", 1.2))  # insert a spine for the third y-axis
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    ax3.set_ylabel("TSP Metric : Number of duplicates")

    epochs_ = range(1, len(results[0][1]) + 1)
    models_name = ""
    lines = []
    for index, res in enumerate(results):
        models_name += '_' + res[0]
        # Accuracy curve
        p1, = ax1.plot(epochs_, res[1], linestyle=line_styles[index % len(line_styles)], color='b',
                       label="model" + res[0])
        lines.append(p1)
        # Valid loss curve
        p2, = ax2.plot(epochs_, res[2], linestyle=line_styles[index % len(line_styles)], color='r')
        # Duplicates curve
        p3, = ax3.plot(epochs_, res[3], linestyle=line_styles[index % len(line_styles)], color='g')

        if index == 0:  # Set color for each y-axis and their label
            ax1.yaxis.label.set_color(p1.get_color())
            ax2.yaxis.label.set_color(p2.get_color())
            ax3.yaxis.label.set_color(p3.get_color())
            ax1.tick_params(axis='y', colors=p1.get_color())
            ax2.tick_params(axis='y', colors=p2.get_color())
            ax3.tick_params(axis='y', colors=p3.get_color())

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Shrink current axis's height by 15% on the bottom + Put a legend below current axis
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=nb_results)

    plt.savefig("../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "structureLearning" + models_name + "_"
                + str(learning_rate).replace('.', 'x'))
    plt.show()
