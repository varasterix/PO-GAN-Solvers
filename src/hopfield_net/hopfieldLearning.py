import os
import random
import time
import matplotlib.pyplot as plt
from src.database.databaseTools import read_tsp_choco_solution_file
from src.hopfield_net.hopfield_TSP_network import HopfieldTSPNetwork
import src.constants as constants
from src.utils import plot_tsp_candidates


if __name__ == '__main__':
    # Parameters
    tsp_database_path = "../../" + constants.PARAMETER_TSP_CHOCO_DATA_FILES
    number_of_examples = 1
    number_of_restarts = 10
    A, B, C, D = 5000, 5000, 2000, 0.8
    MAX_ITERATIONS = 100
    parameters = {'A': A, 'B': B, 'C': C, 'D': D}

    # Loading the chosen TSP dataSet
    tsp_database_files = [file_name for file_name in os.listdir(tsp_database_path)]
    random.shuffle(tsp_database_files)
    tsp_database_size = len(tsp_database_files)
    number_of_examples = tsp_database_size if (number_of_examples > tsp_database_size) else number_of_examples

    # Experimentation on the TSP dataSet
    print("Beginning of the experimentation...")
    sum_iterations = 0
    sum_solutions = 0
    sum_solving_times = 0
    sum_relative_gaps = 0
    for data_file in tsp_database_files[:number_of_examples]:
        # Loading one TSP dataSet
        details = data_file.split('.')[0].split('_')
        print("...Computing " + data_file + "...")
        nb_cities, instance_id = int(details[1]), int(details[2])
        ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, tsp_database_path)
        weight_matrix = ordered_path.get_weight_matrix()
        cartesian_coordinates = ordered_path.get_cartesian_coordinates()

        # Creation of + Running the Hopfield TSP Network Model
        start_time = time.time()
        best_candidate = None
        for i in range(number_of_restarts):
            print('...Restart {}'.format(i + 1))
            network = HopfieldTSPNetwork(nb_cities=nb_cities, distance_matrix=weight_matrix,
                                         cartesian_coordinates=cartesian_coordinates, penalty_parameters=parameters)
            iteration, energy_by_iterations = network.run_until_stable(max_iterations=MAX_ITERATIONS,
                                                                       stop_at_local_min=False)
            candidate = network.get_best_ordered_path_binary_matrix()
            if candidate.is_solution():
                if best_candidate is None or best_candidate.distance() > candidate.distance():
                    print(str(candidate.distance()))
                    best_candidate = candidate.__copy__()
            sum_iterations += iteration
            # plt.figure()
            # plt.plot(range(len(energy_by_iterations)), energy_by_iterations)
            # plt.show()
        if best_candidate is not None:
            print("-> Solution found!")
            sum_solutions += 1
            sum_relative_gaps += (best_candidate.distance() - ordered_path.distance()) / ordered_path.distance()
            plot_tsp_candidates([ordered_path, best_candidate])
        sum_solving_times = time.time() - start_time
    sum_solutions /= number_of_examples
    sum_iterations /= number_of_examples * number_of_restarts
    sum_solving_times /= number_of_examples
    sum_relative_gaps /= number_of_examples
    print("Experimentation finished!")
    print("Report of the experimentation:")
    print("Percentage of solutions found : {:.1f}%, Mean number of iterations : {:.2f} in {:.2f} seconds"
          .format(sum_solutions * 100, sum_iterations, sum_solving_times))
    print("For the solutions found, Mean of the relative gap with Choco solutions : {:.2f}%"
          .format(sum_relative_gaps * 100))
