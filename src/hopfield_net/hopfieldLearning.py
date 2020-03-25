import os
import random
import time
import matplotlib.pyplot as plt
from src.database.databaseTools import read_tsp_choco_solution_file, read_tsp_nnh_solution_file, \
    read_tsp_heuristic_solution_file
from src.hopfield_net.hopfield_TSP_network import HopfieldTSPNetwork
import src.constants as constants
from src.utils import plot_tsp_candidates

if __name__ == '__main__':
    # Parameters
    tsp_choco_database_path = "../../" + constants.PARAMETER_TSP_CHOCO_DATA_FILES_FOR_HOPFIELD
    tsp_nnh_database_path = "../../" + constants.PARAMETER_TSP_NNH_DATA_FILES_FOR_HOPFIELD
    tsp_nnh_two_opt_database_path = "../../" + constants.PARAMETER_TSP_NNH_TWO_OPT_DATA_FILES_FOR_HOPFIELD
    number_of_examples = 10
    number_of_restarts = 5

    # TEST OF THE PARAMETERS A, B, C, D
    A, B = 5000, 5000
    C_set = [3000]  # [4000, 3000, 5000, 2000, 1000]
    D_set = [0.7]  # [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    MAX_ITERATIONS = 1000

    # Computing iterations by instances size
    size_set = [5, 10, 15, 20, 25]
    iterations_by_nb_cities = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0}
    counter_sol_by_nb_cities = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0}

    for C in C_set:
        for D in D_set:
            parameters = {'A': A, 'B': B, 'C': C, 'D': D}

            # Loading the chosen TSP dataSet
            str_parameters = 'A{}_B{}_C{}_D{}'.format(A, B, C, str(D).replace('.', '-'))
            tsp_database_files = [file_name for file_name in os.listdir(tsp_choco_database_path)]
            random.shuffle(tsp_database_files)
            tsp_database_size = len(tsp_database_files)
            number_of_examples = tsp_database_size if (number_of_examples > tsp_database_size) else number_of_examples

            # Experimentation on the TSP dataSet
            print("Beginning of the experimentation...")
            sum_iterations = 0
            sum_solutions = 0
            sum_solving_times = 0
            sum_choco_relative_gaps, sum_nnh_relative_gaps, sum_heuristic_relative_gaps = 0, 0, 0
            for data_file in tsp_database_files[:number_of_examples]:
                # Loading one TSP dataSet
                details = data_file.split('.')[0].split('_')
                print("...Computing " + data_file + "...")
                nb_cities, inst_id = int(details[1]), int(details[2])

                choco_ordered_path, choco_distance = \
                    read_tsp_choco_solution_file(nb_cities, inst_id, tsp_choco_database_path)
                nnh_ordered_path, nnh_distance = \
                    read_tsp_nnh_solution_file(nb_cities, inst_id, tsp_nnh_database_path)
                heuristic_ordered_path, heuristic_distance = \
                    read_tsp_heuristic_solution_file(nb_cities, inst_id, tsp_nnh_two_opt_database_path)
                weight_matrix = choco_ordered_path.get_weight_matrix()
                cartesian_coordinates = choco_ordered_path.get_cartesian_coordinates()

                # Creation of + Running the Hopfield TSP Network Model
                start_time = time.time()
                best_candidate = None
                for i in range(number_of_restarts):
                    print('...Restart {}'.format(i + 1))
                    network = HopfieldTSPNetwork(nb_cities=nb_cities, distance_matrix=weight_matrix,
                                                 cartesian_coordinates=cartesian_coordinates,
                                                 penalty_parameters=parameters)
                    iteration, energy_by_iterations = network.run_until_stable(max_iterations=MAX_ITERATIONS,
                                                                               stop_at_local_min=False)
                    candidate = network.get_best_ordered_path_binary_matrix()
                    if candidate.is_solution():
                        iterations_by_nb_cities[nb_cities] += iteration
                        counter_sol_by_nb_cities[nb_cities] += 1
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
                    sum_choco_relative_gaps += (best_candidate.distance() - choco_distance) / choco_distance
                    sum_nnh_relative_gaps += (best_candidate.distance() - nnh_distance) / nnh_distance
                    sum_heuristic_relative_gaps += (best_candidate.distance() - heuristic_distance) / heuristic_distance
                    plot_tsp_candidates([choco_ordered_path, best_candidate, nnh_ordered_path, heuristic_ordered_path],
                                        inst_id, ['Choco Solver', 'Hopfield TSP Network', 'Nearest Neighbor Heuristic',
                                                  'NNH + 2-Opt'],
                                        "../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "hopfieldLearning_" +
                                        str(nb_cities) + "_" + str(inst_id) + '_' + str_parameters, True)
                sum_solving_times = time.time() - start_time

            # Normalisations
            sum_solutions /= number_of_examples
            sum_iterations /= number_of_examples * number_of_restarts
            sum_solving_times /= number_of_examples
            sum_choco_relative_gaps /= number_of_examples
            sum_nnh_relative_gaps /= number_of_examples
            sum_heuristic_relative_gaps /= number_of_examples
            for n in size_set:
                iterations_by_nb_cities[n] /= (1 if counter_sol_by_nb_cities[n] == 0 else counter_sol_by_nb_cities[n])
            print("Experimentation finished!")
            print("Report of the experimentation:")
            print("Percentage of solutions found : {:.1f}%, Mean number of iterations : {:.2f} in {:.2f} seconds"
                  .format(sum_solutions * 100, sum_iterations, sum_solving_times))
            print("For the solutions found, Mean of the relative gap with Choco solutions : {:.2f}%, "
                  "with NNH solutions : {:.2f}%, with NNH+2opt solutions : {:.2f}%".format(
                    sum_choco_relative_gaps * 100, sum_nnh_relative_gaps * 100, sum_heuristic_relative_gaps * 100))

            # plt.figure()
            # plt.title("Number of the Hopfield network dynamic iterations by instances size")
            # plt.plot([0] + list(iterations_by_nb_cities.keys()), [0] + list(iterations_by_nb_cities.values()))
            # plt.xlabel("Instance sizes, number of cities n")
            # plt.ylabel("Number of iterations")
            # plt.savefig("../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "hopfieldLearning_iterations_" +
            #             str_parameters + "_maxIter_" + str(MAX_ITERATIONS))
            # plt.show()
