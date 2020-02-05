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
from src.segmented_sol.seg_net import SegNet
import src.objects.orderedPath as oP


def train(model, train_set, optimizer, tsp_database_path):
    model.train()
    random.shuffle(train_set)
    sum_predictions = 0
    sum_predicted_distances = 0
    sum_target_distances = 0
    instances_nb_cities = int(train_set[0].split('.')[0].split('_')[1])
    for data_file in train_set:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        candidate_target = ordered_path.get_candidate()
        sum_target_distances += int(ordered_path.distance())
        path = np.zeros(nb_cities, dtype=int)
        path[0] = candidate_target[0]
        for i in range(nb_cities - 1):
            # Building the array of the visited cities
            visited_cities = np.zeros((1, nb_cities))
            for k in range(i):
                visited_cities[0, candidate_target[k]] = 1
            # Building the one-hot array of the current city
            current_city_one_hot = np.zeros((1, nb_cities))
            current_city_one_hot[0, candidate_target[i]] = 1
            # Building the input of the neural network
            input_data = Variable(torch.tensor(
                np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
                dtype=torch.float), requires_grad=True)
            # Building the expected target
            target_one_hot = np.zeros((1, nb_cities))
            target_one_hot[0, candidate_target[i + 1]] = 1
            target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
            # Training number (i+1)/nb_cities for the considered TSP instance
            optimizer.zero_grad()
            output = model(input_data)  # calls the forward function
            loss = model.loss_function(output, target)
            loss.backward()
            optimizer.step()

            predicted_next_city = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
            sum_predictions += int(predicted_next_city == candidate_target[i + 1])
            # Building the array of the predicted visited cities
            next_city = 0
            maxi = 0
            np_output = output.detach().numpy()[0]
            for city in range(nb_cities):
                if not(city in path[:i+1]):
                    if np_output[city] > maxi:
                        maxi = np_output[city]
                        next_city = city
            path[i+1] = next_city

        candidate_predicted = oP.OrderedPath(path, ordered_path.get_weight_matrix())
        sum_predicted_distances += int(candidate_predicted.distance())

    train_set_size = len(train_set)
    total_predictions = train_set_size * (instances_nb_cities - 1)
    print('Training indicators : ' +
          'Accuracy (predictions): {:.2f}%, Distance average target: {:.2f}, Distance average predicted: {:.2f}'.format(
              (sum_predictions / total_predictions), sum_target_distances / train_set_size,
              sum_predicted_distances / train_set_size))
    return model


def valid(model, valid_set, epoch, tsp_database_path):
    model.eval()
    random.shuffle(valid_set)
    valid_loss = 0
    sum_predictions = 0
    instances_nb_cities = int(valid_set[0].split('.')[0].split('_')[1])
    for data_file in valid_set:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        candidate_target = ordered_path.get_candidate()
        for i in range(nb_cities - 1):
            # Building the array of the visited cities
            visited_cities = np.zeros((1, nb_cities))
            for k in range(i):
                visited_cities[0, candidate_target[k]] = 1
            # Building the one-hot array of the current city
            current_city_one_hot = np.zeros((1, nb_cities))
            current_city_one_hot[0, candidate_target[i]] = 1
            # Building the input of the neural network
            input_data = Variable(torch.tensor(
                np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
                dtype=torch.float), requires_grad=True)
            # Building the expected target
            target_one_hot = np.zeros((1, nb_cities))
            target_one_hot[0, candidate_target[i + 1]] = 1
            target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)

            output = model(input_data)  # calls the forward function
            valid_loss += model.loss_function(output, target)

            predicted_next_city = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
            sum_predictions += int(predicted_next_city == candidate_target[i + 1])

    valid_set_size = len(valid_set)
    total_predictions = valid_set_size * (instances_nb_cities - 1)
    valid_loss /= total_predictions
    print('Epoch {} - valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, valid_loss, sum_predictions, total_predictions, 100. * sum_predictions / total_predictions))
    return sum_predictions / total_predictions, valid_loss


def test(model, test_set, tsp_database_path):
    model.eval()
    random.shuffle(test_set)
    sum_predictions = 0
    test_loss = 0
    instances_nb_cities = int(test_set[0].split('.')[0].split('_')[1])
    for data_file in test_set:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        candidate_target = ordered_path.get_candidate()
        for i in range(nb_cities - 1):
            # Building the array of the visited cities
            visited_cities = np.zeros((1, nb_cities))
            for k in range(i):
                visited_cities[0, candidate_target[k]] = 1
            # Building the one-hot array of the current city
            current_city_one_hot = np.zeros((1, nb_cities))
            current_city_one_hot[0, candidate_target[i]] = 1
            # Building the input of the neural network
            input_data = Variable(torch.tensor(
                np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
                dtype=torch.float), requires_grad=True)
            # Building the expected target
            target_one_hot = np.zeros((1, nb_cities))
            target_one_hot[0, candidate_target[i + 1]] = 1
            target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)

            output = model(input_data)  # calls the forward function
            test_loss += model.loss_function(output, target)

            predicted_next_city = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
            sum_predictions += int(predicted_next_city == candidate_target[i + 1])

    total_predictions = len(test_set) * (instances_nb_cities - 1)
    test_loss /= total_predictions
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, sum_predictions, total_predictions, 100. * sum_predictions / total_predictions))


def experiment(model, epochs, lr, train_set, valid_set, tsp_database_path):
    best_precision = 0
    best_model = model
    accuracy_by_epochs = []
    valid_loss_by_epochs = []
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model = train(model, train_set, optimizer, tsp_database_path)
        precision, loss = valid(model, valid_set, epoch, tsp_database_path)
        accuracy_by_epochs.append(precision)
        valid_loss_by_epochs.append(loss)
        if precision > best_precision:
            best_precision = precision
            best_model = model

    return best_model, best_precision, accuracy_by_epochs, valid_loss_by_epochs


if __name__ == '__main__':
    # Parameters
    tsp_heuristic_database_path = "../../" + constants.PARAMETER_TSP_DATA_FILES
    number_cities = 10
    train_proportion = 0.8
    valid_proportion = 0.1
    test_proportion = 0.1
    over_fit_one_instance = False
    Models = [SegNet(number_cities)]  # add your models in the list
    nb_epochs = 200
    learning_rate = 0.001

    # Preparation of the TSP dataSet
    tsp_database_files = [file_name for file_name in os.listdir(tsp_heuristic_database_path)
                          if file_name.split('.')[1] == 'heuristic']
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
        model_, precision_, accuracy_by_epochs_, valid_loss_by_epochs_ = \
            experiment(model_, epochs=nb_epochs, lr=learning_rate, train_set=train_data, valid_set=valid_data,
                       tsp_database_path=tsp_heuristic_database_path)
        results.append((model_.name, accuracy_by_epochs_, valid_loss_by_epochs_))
        if precision_ > best_precision_:
            best_precision_ = precision_
            best_model_ = model_

    test(best_model_, test_data, tsp_heuristic_database_path)

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
    # ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
    # ax3.spines["right"].set_position(("axes", 1.2))  # insert a spine for the third y-axis
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    # ax3.set_ylabel("TSP Metric : Number of duplicates")

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
        # p3, = ax3.plot(epochs_, res[3], linestyle=line_styles[index % len(line_styles)], color='g')

        if index == 0:  # Set color for each y-axis and their label
            ax1.yaxis.label.set_color(p1.get_color())
            ax2.yaxis.label.set_color(p2.get_color())
            # ax3.yaxis.label.set_color(p3.get_color())
            ax1.tick_params(axis='y', colors=p1.get_color())
            ax2.tick_params(axis='y', colors=p2.get_color())
            # ax3.tick_params(axis='y', colors=p3.get_color())

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Shrink current axis's height by 15% on the bottom + Put a legend below current axis
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=nb_results)

    plt.savefig("../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "SegmentedLearning" + models_name)
    plt.show()