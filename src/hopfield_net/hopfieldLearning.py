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

    """
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

    plt.savefig("../../" + constants.PARAMETER_FIGURE_RESULTS_PATH + "hopfieldLearning" + models_name)
    plt.show()
    """
