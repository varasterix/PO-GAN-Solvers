import os
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import optim
from src.utils import custom_formatwarning
from src.database.databaseTools import read_tsp_heuristic_solution_file
from src.objects.objectsTools import normalize_weight_matrix
from src import constants
from src.segmented_net.seg_net import SegNet
from src.segmented_net.seg_net2 import SegNet2


def train(model, train_set, optimizer, database_path):
    model.train()
    random.shuffle(train_set)
    sum_predictions = 0
    for data_file in train_set:
        nb_cities, instance_id = data_file[0], data_file[1]
        visited_cities, current_city_one_hot, target_one_hot = data_file[2], data_file[3], data_file[4]
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        input_data = Variable(torch.tensor(
            np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
            dtype=torch.float), requires_grad=True)
        target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
        optimizer.zero_grad()
        output = model(input_data)  # calls the forward function
        loss = model.loss_function(output, target)
        loss.backward()
        optimizer.step()

        expected_next_city = int(torch.argmax(target, dim=1))
        predicted_next_city = int(torch.argmax(output, dim=1))
        sum_predictions += int(predicted_next_city == expected_next_city)

    train_set_size = len(train_set)
    print('Training indicators : ' +
          'Accuracy (predictions): {:.4f}%'.format((sum_predictions / train_set_size) * 100.))
    return model


def valid(model, valid_set, epoch, database_path):
    model.eval()
    random.shuffle(valid_set)
    valid_loss = 0
    sum_predictions = 0
    for data_file in valid_set:
        nb_cities, instance_id = data_file[0], data_file[1]
        visited_cities, current_city_one_hot, target_one_hot = data_file[2], data_file[3], data_file[4]
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        input_data = Variable(torch.tensor(
            np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
            dtype=torch.float), requires_grad=True)
        target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
        output = model(input_data)  # calls the forward function
        valid_loss += model.loss_function(output, target)

        expected_next_city = int(torch.argmax(target, dim=1))
        predicted_next_city = int(torch.argmax(output, dim=1))
        sum_predictions += int(predicted_next_city == expected_next_city)

    valid_set_size = len(valid_set)
    accuracy = sum_predictions / valid_set_size
    valid_loss /= valid_set_size
    print('Epoch {} - valid set: Average loss: {:.4f}, Accuracy (predictions): {}/{} ({:.0f}%)\n'.format(
        epoch, valid_loss, sum_predictions, valid_set_size, 100. * accuracy))
    return accuracy, valid_loss


def test(model, test_set, database_path):
    model.eval()
    random.shuffle(test_set)
    sum_predictions = 0
    test_loss = 0
    for data_file in test_set:
        nb_cities, instance_id = data_file[0], data_file[1]
        visited_cities, current_city_one_hot, target_one_hot = data_file[2], data_file[3], data_file[4]
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        input_data = Variable(torch.tensor(
            np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
            dtype=torch.float), requires_grad=True)
        target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
        output = model(input_data)  # calls the forward function
        test_loss += model.loss_function(output, target)
        expected_next_city = int(torch.argmax(target, dim=1))
        predicted_next_city = int(torch.argmax(output, dim=1))
        sum_predictions += int(predicted_next_city == expected_next_city)

    test_set_size = len(test_set)
    test_loss /= test_set_size
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, sum_predictions, test_set_size, 100. * sum_predictions / test_set_size))
    return sum_predictions / test_set_size


def experiment(model, epochs, lr, train_set, valid_set, database_path):
    best_precision = 0
    best_model = model
    accuracy_by_epochs = []
    valid_loss_by_epochs = []
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model = train(model, train_set, optimizer, database_path)
        precision, loss = valid(model, valid_set, epoch, database_path)

        accuracy_by_epochs.append(precision)
        valid_loss_by_epochs.append(loss)
        if precision > best_precision:
            best_precision = precision
            best_model = model

    return best_model, best_precision, accuracy_by_epochs, valid_loss_by_epochs


if __name__ == '__main__':
    # Parameters
    tsp_database_path = "../../" + constants.PARAMETER_TSP_NNH_TWO_OPT_DATA_FILES
    number_cities = 10
    train_proportion = 0.8
    valid_proportion = 0.1
    test_proportion = 0.1
    over_fit_one_instance = False
    # Models = [SegNet(number_cities), SegNet2(number_cities)]  # add your models in the list
    Models = [SegNet2(number_cities)]  # add your models in the list
    nb_epochs = 100
    learning_rate = 0.01

    # Preparation of the TSP dataSet
    tsp_database_files = [file_name for file_name in os.listdir(tsp_database_path)]

    database_reformatted = []
    for tsp_data_file in tsp_database_files:
        details = tsp_data_file.split('.')[0].split('_')
        tsp_nb_cities, tsp_instance_id = int(details[1]), int(details[2])
        tsp_ordered_path, _ = read_tsp_heuristic_solution_file(tsp_nb_cities, tsp_instance_id, tsp_database_path)
        candidate_target = tsp_ordered_path.get_candidate()
        tsp_visited_cities = np.zeros((1, tsp_nb_cities))
        for i in range(tsp_nb_cities - 1):
            # first arg : number of cities, second arg : instance id
            instance_reformatted = [tsp_nb_cities, tsp_instance_id]
            # third arg : visited cities
            tsp_visited_cities[0, candidate_target[i]] = 1
            instance_reformatted.append(tsp_visited_cities)
            # fourth arg : current city (one hot)
            current_city = np.zeros((1, tsp_nb_cities))
            current_city[0, candidate_target[i]] = 1
            instance_reformatted.append(current_city)
            # fifth arg : target (one hot)
            target_city = np.zeros((1, tsp_nb_cities))
            target_city[0, candidate_target[i + 1]] = 1
            instance_reformatted.append(target_city)
            # Integration of the instance to the database
            database_reformatted.append(instance_reformatted)

    random.shuffle(database_reformatted)
    database_reformatted_size = len(database_reformatted)
    if over_fit_one_instance:
        database_reformatted = [database_reformatted[0]] * database_reformatted_size

    bound_1 = round(train_proportion * database_reformatted_size)
    bound_2 = round((train_proportion + valid_proportion) * database_reformatted_size)
    train_data = database_reformatted[0:bound_1]
    valid_data = database_reformatted[bound_1:bound_2]
    test_data = database_reformatted[bound_2:]

    # Experimentation of the different models on the TSP dataSet
    best_precision_ = 0
    best_model_ = Models[0]
    results = []
    for model_ in Models:
        model_, precision_, accuracy_by_epochs_, valid_loss_by_epochs_ = \
            experiment(model_, epochs=nb_epochs, lr=learning_rate, train_set=train_data, valid_set=valid_data,
                       database_path=tsp_database_path)
        results.append((model_.name, accuracy_by_epochs_, valid_loss_by_epochs_))
        if precision_ > best_precision_:
            best_precision_ = precision_
            best_model_ = model_

    test_accuracy = test(best_model_, test_data, tsp_database_path)

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
    plt.title("Validation loss, accuracy, TSP metric by epochs (test acc {:.0f}%)"
              .format(test_accuracy * 100.))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax3.spines["right"].set_position(("axes", 1.2))  # insert a spine for the third y-axis
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")

    epochs_ = range(1, len(results[0][1]) + 1)
    models_name = ""
    lines = []
    for index, res in enumerate(results):
        models_name += '_' + res[0]
        # Accuracy curve
        p1, = ax1.plot(epochs_, res[1], linestyle=line_styles[index % len(line_styles)], color='b',
                       label="model" + res[0])
        # Valid loss curve
        p2, = ax2.plot(epochs_, res[2], linestyle=line_styles[index % len(line_styles)], color='r')

        if index == 0:  # Set color for each y-axis and their label
            ax1.yaxis.label.set_color(p1.get_color())
            ax2.yaxis.label.set_color(p2.get_color())
            ax1.tick_params(axis='y', colors=p1.get_color())
            ax2.tick_params(axis='y', colors=p2.get_color())

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Shrink current axis's height by 15% on the bottom + Put a legend below current axis
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=nb_results)

    plt.savefig("../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "segmentedLearning" + models_name + "_"
                + str(learning_rate).replace('.', 'x'))
    plt.show()
