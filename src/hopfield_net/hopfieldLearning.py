import os
import random
import matplotlib.pyplot as plt
from src.database.databaseTools import read_tsp_choco_solution_file
from src.hopfield_net.hopfield_TSP_network import HopfieldTSPNetwork
import src.constants as constants


if __name__ == '__main__':
    # Parameters
    tsp_database_path = "../../" + constants.PARAMETER_TSP_CHOCO_DATA_FILES
    number_of_examples = 5
    A, B, C, D = 5000, 5000, 5000, 0
    MAX_ITERATIONS = 500
    parameters = {'A': A, 'B': B, 'C': C, 'D': D}

    # Loading the chosen TSP dataSet
    tsp_database_files = [file_name for file_name in os.listdir(tsp_database_path)]
    random.shuffle(tsp_database_files)
    tsp_database_size = len(tsp_database_files)
    number_of_examples = tsp_database_size if (number_of_examples > tsp_database_size) else number_of_examples

    # Experimentation on the TSP dataSet
    print("Beginning of the experimentation...")
    number_of_iterations = 0
    number_of_solutions = 0
    relative_gap = 0
    for data_file in tsp_database_files[:number_of_examples]:
        # Loading one TSP dataSet
        details = data_file.split('.')[0].split('_')
        print("...Computing " + data_file + "...")
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        cartesian_coordinates = ordered_path.get_cartesian_coordinates()
        # Creation of Hopfield TSP Network Model
        network = HopfieldTSPNetwork(nb_cities=nb_cities, distance_matrix=weight_matrix,
                                     cartesian_coordinates=cartesian_coordinates, penalty_parameters=parameters)
        iteration, energy_by_iterations = network.run_until_stable(max_iterations=MAX_ITERATIONS)
        candidate = network.get_ordered_path_binary_matrix()
        if candidate.is_solution():
            print("-> Solution found!")
            number_of_solutions += 1
            relative_gap += (candidate.distance() - ordered_path.distance()) / ordered_path.distance()
        number_of_iterations += iteration
        # plt.figure()
        # plt.plot(range(len(energy_by_iterations)), energy_by_iterations)
        # plt.show()
    number_of_solutions /= number_of_examples
    number_of_iterations /= number_of_examples
    relative_gap /= number_of_examples
    print("Experimentation finished!")
    print("Report of the experimentation:")
    print("Percentage of solutions found : {:.1f}%, Mean number of iterations : {:.2f}"
          .format(number_of_solutions * 100, number_of_iterations))
    print("For the solutions found, Mean of the relative gap with Choco solutions : {:.4f}".format(relative_gap))
