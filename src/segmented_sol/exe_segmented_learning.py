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
from src.segmented_sol.seg_net2 import SegNet2
import src.objects.orderedPath as oP


def train(model, train_set, optimizer, tsp_database_path):
    model.train()
    random.shuffle(train_set)
    sum_predictions = 0
    for data_file in train_set:
        visited_cities = data_file[0]
        current_city_one_hot = data_file[1]
        target_one_hot = data_file[2]
        instance_id = data_file[3]
        nb_cities = 10
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        input_data = Variable(torch.tensor(
            np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
            dtype=torch.float), requires_grad=True)
        target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
        city_target = np.array(torch.argmax(target.detach(), dim=1), dtype=int)
        optimizer.zero_grad()
        output = model(input_data)  # calls the forward function
        loss = model.loss_function(output, target)
        loss.backward()
        optimizer.step()

        predicted_next_city = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
        sum_predictions += int(predicted_next_city == city_target)


    train_set_size = len(train_set)
    total_predictions = train_set_size
    print('Training indicators : ' +
          'Accuracy (predictions): {:.2f}%'.format(
              (sum_predictions / total_predictions)))
    return model


def valid(model, valid_set, epoch, tsp_database_path):
    model.eval()
    random.shuffle(valid_set)
    valid_loss = 0
    sum_predictions = 0
    for data_file in valid_set:
        visited_cities = data_file[0]
        current_city_one_hot = data_file[1]
        target_one_hot = data_file[2]
        instance_id = data_file[3]
        nb_cities = 10
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        input_data = Variable(torch.tensor(
            np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
            dtype=torch.float), requires_grad=True)
        target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
        city_target = np.array(torch.argmax(target.detach(), dim=1), dtype=int)
        output = model(input_data)  # calls the forward function
        valid_loss += model.loss_function(output, target)

        predicted_next_city = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
        sum_predictions += int(predicted_next_city == city_target)

    total_predictions = len(valid_set)
    accuracy = sum_predictions/total_predictions
    valid_loss /= total_predictions
    print('Epoch {} - valid set: Accuracy (predictions): {:.2f}%, loss: {:.2f}'.format(
              epoch, accuracy, valid_loss))
    return accuracy, valid_loss


def test(model, test_set, tsp_database_path):
    model.eval()
    random.shuffle(test_set)
    sum_predictions = 0
    test_loss = 0
    for data_file in test_set:
        visited_cities = data_file[0]
        current_city_one_hot = data_file[1]
        target_one_hot = data_file[2]
        instance_id = data_file[3]
        nb_cities = 10
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_database_path)
        formatted_weight_matrix = \
            normalize_weight_matrix(ordered_path.get_weight_matrix()).reshape((1, nb_cities * nb_cities))
        input_data = Variable(torch.tensor(
            np.concatenate((formatted_weight_matrix, visited_cities, current_city_one_hot), axis=1),
            dtype=torch.float), requires_grad=True)
        target = Variable(torch.tensor(target_one_hot, dtype=torch.float), requires_grad=True)
        city_target = np.array(torch.argmax(target.detach(), dim=1), dtype=int)
        output = model(input_data)  # calls the forward function
        test_loss += model.loss_function(output, target)
        predicted_next_city = np.array(torch.argmax(output.detach(), dim=1), dtype=int)
        sum_predictions += int(predicted_next_city == city_target)

    total_predictions = len(test_set)
    test_loss /= total_predictions
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, sum_predictions, total_predictions, 100. * sum_predictions / total_predictions))


def experiment(model, epochs, lr, train_set, valid_set, tsp_database_path):
    best_precision = 0
    best_model = model
    accuracy_by_epochs = []
    valid_loss_by_epochs = []
    predicted_distances_by_epochs = []
    target_distances_by_epochs = []
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
    tsp_heuristic_database_path = "../../" + constants.PARAMETER_TSP_NNH_TWO_OPT_DATA_FILES
    number_cities = 10
    train_proportion = 0.8
    valid_proportion = 0.1
    test_proportion = 0.1
    over_fit_one_instance = False
    #Models = [SegNet(number_cities), SegNet2(number_cities)]  # add your models in the list
    Models = [SegNet2(number_cities)]  # add your models in the list
    nb_epochs = 200
    learning_rate = 0.001

    # Preparation of the TSP dataSet
    tsp_database_files = [file_name for file_name in os.listdir(tsp_heuristic_database_path)
                          if file_name.split('.')[1] == 'heuristic']
    tsp_city_nextcity = []
    for data_file in tsp_database_files:
        details = data_file.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_heuristic_solution_file(nb_cities, instance_id, tsp_heuristic_database_path)
        candidate_target = ordered_path.get_candidate()
        visited_cities = np.zeros((1, nb_cities))
        for i in range (nb_cities-1):
            city_nextcity = []
            # fst arg : visited cities
            visited_cities[0, candidate_target[i]] = 1
            city_nextcity.append(visited_cities)
            # snd arg : current city
            current_city_one_hot = np.zeros((1, nb_cities))
            current_city_one_hot[0, candidate_target[i]] = 1
            city_nextcity.append(current_city_one_hot)
            # thrd arg : target
            target_one_hot = np.zeros((1, nb_cities))
            target_one_hot[0, candidate_target[i + 1]] = 1
            city_nextcity.append(target_one_hot)
            # frth arg : instance id
            city_nextcity.append(instance_id)
            tsp_city_nextcity.append(city_nextcity)

    random.shuffle(tsp_city_nextcity)
    tsp_city_nextcity_size = len(tsp_city_nextcity)
    if over_fit_one_instance:
        tsp_city_nextcity = [tsp_city_nextcity[0]] * tsp_city_nextcity_size

    bound_1 = round(train_proportion * tsp_city_nextcity_size)
    bound_2 = round((train_proportion + valid_proportion) * tsp_city_nextcity_size)
    train_data = tsp_city_nextcity[0:bound_1]
    valid_data = tsp_city_nextcity[bound_1:bound_2]
    test_data = tsp_city_nextcity[bound_2:]

    # Experimentation of the different models on the TSP dataSet
    best_precision = 0
    best_model_ = Models[0]
    results = []
    for model_ in Models:
        model_, precision, accuracy_by_epochs_, valid_loss_by_epochs_ = \
            experiment(model_, epochs=nb_epochs, lr=learning_rate, train_set=train_data, valid_set=valid_data,
                       tsp_database_path=tsp_heuristic_database_path)
        results.append((model_.name, accuracy_by_epochs_, valid_loss_by_epochs_))
        if precision > best_precision:
            best_precision = precision
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
    #ax3.spines["right"].set_position(("axes", 1.2))  # insert a spine for the third y-axis
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")

    epochs_ = range(1, len(results[0][1]) + 1)
    models_name = ""
    lines = []
    for index, res in enumerate(results):
        models_name += '_' + res[0]
        # Accuracy curve
        p1, = ax1.plot(epochs_, res[1], linestyle=line_styles[index % len(line_styles)], color='b', label="model" + res[0])

        # Valid loss curve
        p2, = ax2.plot(epochs_, res[2], linestyle=line_styles[index % len(line_styles)], color='r')
        # prediction distance curve
        #p3, = ax3.plot(epochs_, res[3], linestyle=line_styles[index % len(line_styles)], color='g')
        #lines.append(p3)
        # target distance curve
        #p4, = ax3.plot(epochs_, res[4], linestyle=line_styles[index % len(line_styles)], color='y')


        if index == 0:  # Set color for each y-axis and their label
            ax1.yaxis.label.set_color(p1.get_color())
            ax2.yaxis.label.set_color(p2.get_color())
            #ax3.yaxis.label.set_color(p3.get_color())
            #ax4.yaxis.label.set_color(p4.get_color())
            ax1.tick_params(axis='y', colors=p1.get_color())
            ax2.tick_params(axis='y', colors=p2.get_color())
            #ax3.tick_params(axis='y', colors=p3.get_color())
            #ax4.tick_params(axis='y', colors=p4.get_color())

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Shrink current axis's height by 15% on the bottom + Put a legend below current axis
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=nb_results)

    plt.savefig("../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "segmentedLearning" + models_name)
    plt.show()
