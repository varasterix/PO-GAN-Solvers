import numpy as np
import random
from shutil import copyfile
from src import constants
import src.objects.orderedPath as oP
from src.heuristicsTSP import nearestNeighborHeuristic as nNH, twoOptImprovementHeuristic as tOpt

"""
Considering an instance of the TSP, it is defined with its weight/distance matrix <M> of size (n x n) with n, the
number of cities of this considered instance.
Thus, a file in the "data/tsp_files/", which is called "dataSet_<n>_<instance_id>.tsp" is written in this format :
"   <instance_id>   (with <instance_id> between 0 to the number of instances with <n> cities studied)
    <n>             (the number of cities)
    <M>[0,0]   <M>[0,1]    ...  <M>[0,n-1] (with <M>[i,j] the weight/distance from the city i to the city j)
    <M>[1,0]   <M>[1,1]    ...  <M>[1,n-1]
    ...        ...         ...  ...
    <M>[n-1,0] <M>[n-1,1]  ...  <M>[n-1,n-1]    "
Note: For all i in [|0, n-1|], <M>[i,i] = 0

Besides, a file in the same repository, called "dataSet_<n>_<instance_id>.heuristic", which also contains a solution 
to the considered instance of the TSP (in the "dataSet_<n>_<instance_id>.tsp" file), is written in this format :
"   <instance_id>
    <n>
    <M>[0,0]   <M>[0,1]    ...  <M>[0,n-1] (with <M>[i,j] the weight/distance from the city i to the city j)
    <M>[1,0]   <M>[1,1]    ...  <M>[1,n-1]
    ...        ...         ...  ...
    <M>[n-1,0] <M>[n-1,1]  ...  <M>[n-1,n-1]
    <S>[0]  <S>[1]  ... <S>[n-1] (with <S>[i] the (i+1)th city visited in the solution)
    <total_weigh> (the total cost/weight/distance of the solution S for the considered instance)    "
"""


def read_tsp_file(nb_cities, instance_id, path=constants.PARAMETER_TSP_DATA_FILES):
    """
    Imports the weight/distance matrix corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.tsp"
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied
    :param path: the path from the project root of the TSP dataSet file considered
    :return: the weight/distance matrix corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.tsp" considered
    """
    tsp_file = open(path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".tsp", 'r')
    weight_matrix = np.zeros((nb_cities, nb_cities), dtype=int)
    for i, line in enumerate(tsp_file):
        if not (i == 0 or i == 1):
            for j, w_ij in enumerate(line[:-1].split('\t')):
                weight_matrix[i - 2, j] = int(w_ij)
    tsp_file.close()
    return weight_matrix


def generate_tsp_file(nb_cities, instance_id, path=constants.PARAMETER_TSP_DATA_FILES,
                      highest_weight=100, symmetric=False):
    """
    Generates the TSP dataSet file "dataSet_<n>_<instance_id>.tsp" for an instance with a given number of cities
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered (int)
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied (int)
    :param path: the path from the project root of the TSP dataSet file considered (str)
    :param highest_weight: highest integer weight which can be generated in the TSP dataSet file (int)
    :param symmetric: boolean True if the weight/distance matrix has to be symmetric, False otherwise
    :return: the weight/distance matrix corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.tsp" considered
    """
    tsp_file = open(path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".tsp", 'w+')
    tsp_file.write(str(instance_id) + '\n')
    tsp_file.write(str(nb_cities) + '\n')
    memory_weight_matrix = None
    if symmetric:
        # Only the upper strict triangular matrix will be filled
        memory_weight_matrix = np.zeros((nb_cities, nb_cities), dtype=int)
    for i in range(nb_cities):
        for j in range(nb_cities):
            if i == j:
                w = 0
            elif symmetric and i > j:
                w = memory_weight_matrix[j, i]
            else:
                w = random.randint(0, highest_weight)  # Return a random integer w such that 0 <= w <= highest_weight
                if symmetric:
                    memory_weight_matrix[i, j] = w
            if j == (nb_cities - 1):
                tsp_file.write(str(w) + "\n")
            else:
                tsp_file.write(str(w) + "\t")
    tsp_file.close()
    return None


def compute_tsp_heuristic_solution(nb_cities, instance_id, path=constants.PARAMETER_TSP_DATA_FILES, time_limit=1):
    """
    Computes the TSP dataSet heuristic solution file "dataSet_<n>_<instance_id>.heuristic" corresponding to the
    instance of the TSP given by the file "dataSet_<n>_<instance_id>.tsp"
    WARNING: The weight/distance matrix from this instance has to SYMMETRIC, otherwise an Exception is raised
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered (int)
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied (int)
    :param path: the path from the project root of the TSP dataSet file considered (str)
    :param time_limit: time limit to improve the solution with the 2-opt heuristic (unit = seconds, default = 1s) (int)
    """
    tsp_weight_matrix = read_tsp_file(nb_cities, instance_id, path)
    heuristic_path = copyfile(path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".tsp",
                              path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".heuristic")
    heuristic_file = open(heuristic_path, 'a')
    nnh_solution = nNH.NearestNeighborHeuristic(tsp_weight_matrix)
    two_opt_solution = tOpt.TwoOptImprovementHeuristic(nnh_solution.get_ordered_path(), time_limit)
    for i, city in enumerate(two_opt_solution.get_ordered_path().get_candidate()):
        if i == (nb_cities - 1):
            heuristic_file.write(str(city) + "\n")
        else:
            heuristic_file.write(str(city) + "\t")
    heuristic_file.write(str(two_opt_solution.get_total_weight()) + '\n')
    heuristic_file.close()
    return None


def read_tsp_nnh_solution_file(nb_cities, instance_id, path=constants.PARAMETER_TSP_NNH_DATA_FILES):
    """
    Imports a solution (object OrderedPath containing the weight/distance matrix) and the total cost/weight/distance of
    this solution corresponding to the TSP dataSet heuristic solution file "dataSet_<n>_<instance_id>.nnh"
    :param nb_cities: the number of cities of the instance of the TSP dataSet heuristic solution file considered
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied
    :param path: the path from the project root of the TSP dataSet heuristic solution file considered
    :return: a solution (object OrderedPath containing the weight/distance matrix) corresponding to the TSP dataSet
    heuristic solution file "dataSet_<n>_<instance_id>.nnh" considered, and its total cost/weight/distance
    """
    return read_tsp_extension_solution_file(nb_cities, instance_id, path, "nnh")


def read_tsp_heuristic_solution_file(nb_cities, instance_id, path=constants.PARAMETER_TSP_NNH_TWO_OPT_DATA_FILES):
    """
    Imports a solution (object OrderedPath containing the weight/distance matrix) and the total cost/weight/distance of
    this solution corresponding to the TSP dataSet heuristic solution file "dataSet_<n>_<instance_id>.heuristic"
    :param nb_cities: the number of cities of the instance of the TSP dataSet heuristic solution file considered
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied
    :param path: the path from the project root of the TSP dataSet heuristic solution file considered
    :return: a solution (object OrderedPath containing the weight/distance matrix) corresponding to the TSP dataSet
    heuristic solution file "dataSet_<n>_<instance_id>.heuristic" considered, and its total cost/weight/distance
    """
    return read_tsp_extension_solution_file(nb_cities, instance_id, path, "heuristic")


def read_tsp_choco_solution_file(nb_cities, instance_id, path=constants.PARAMETER_TSP_CHOCO_DATA_FILES):
    """
    Imports the Choco solution (object OrderedPath containing the weight/distance matrix and the cartesian coordinates)
    and its total cost/weight/distance corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.choco"
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied
    :param path: the path from the project root of the TSP dataSet file considered
    :return: the Choco solution (object OrderedPath containing the weight/distance matrix) corresponding to the TSP
    dataSet file "dataSet_<n>_<instance_id>.choco" considered, and its total cost/weight/distance
    """
    return read_tsp_extension_solution_file(nb_cities, instance_id, path, "choco")


def read_tsp_extension_solution_file(nb_cities, instance_id, path, extension):
    """
    Imports the solution (object OrderedPath containing the weight/distance matrix and the cartesian coordinates)
    and its total cost/weight/distance corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.<extension>"
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered (str)
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied (str)
    :param path: the path from the project root of the TSP dataSet file considered (str)
    :param extension: the extension of the TSP dataSet file considered (str)
    :return: the solution (object OrderedPath containing the weight/distance matrix and the cartesian coordinates)
    corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.<extension>" considered, and its total
    cost/weight/distance matrix
    """
    solution_file = open(path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + "." + extension, 'r')
    weight_matrix = np.zeros((nb_cities, nb_cities), dtype=int)
    ordered_path, total_weight = [], 0
    cartesian_coordinates = np.zeros((nb_cities, 2), dtype=int)
    contain_cartesian_coordinates = False
    for i, line in enumerate(solution_file):
        if 2 <= i < (2 + nb_cities):
            for j, w_ij in enumerate(line[:-1].split('\t')):
                weight_matrix[i - 2, j] = int(w_ij)
        elif i == (2 + nb_cities):
            ordered_path = [int(city) for index, city in enumerate(line[:-1].split('\t')) if index < nb_cities]
        elif i == (2 + nb_cities + 1):
            total_weight = int(line[:-1])
        elif (2 + nb_cities + 1) < i:
            contain_cartesian_coordinates = True
            for j, x_ij in enumerate(line[:-1].split('\t')):
                cartesian_coordinates[i - nb_cities - 4, j] = int(x_ij)
    solution_file.close()
    cartesian_coordinates = None if not contain_cartesian_coordinates else cartesian_coordinates
    return oP.OrderedPath(np.array(ordered_path, dtype=int), weight_matrix, cartesian_coordinates), total_weight


def compute_tsp_nnh_solution_from_choco_database(nb_cities, instance_id,
                                                 path=constants.PARAMETER_TSP_NNH_DATA_FILES,
                                                 choco_path=constants.PARAMETER_TSP_CHOCO_DATA_FILES):
    """
    Computes the TSP dataSet nearest neighbor heuristic solution file "dataSet_<n>_<instance_id>.nnh" corresponding to
    the instance of the TSP given by the file "dataSet_<n>_<instance_id>.choco"
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered (int)
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied (int)
    :param path: the path from the project root to store TSP nearest neighbor heuristic solution (str)
    :param choco_path: the path from the project root of the TSP choco dataSet file considered (str)
    """
    choco_ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, choco_path)
    nnh_path = path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".nnh"
    nnh_file = open(nnh_path, 'w+')
    nnh_file.write(str(instance_id) + '\n')
    nnh_file.write(str(nb_cities) + '\n')
    nnh_solution = nNH.NearestNeighborHeuristic(choco_ordered_path.get_weight_matrix(),
                                                choco_ordered_path.get_cartesian_coordinates())
    for i in range(nb_cities):
        for j in range(nb_cities):
            if j == (nb_cities - 1):
                nnh_file.write(str(nnh_solution.get_weight_matrix()[i, j]) + "\n")
            else:
                nnh_file.write(str(nnh_solution.get_weight_matrix()[i, j]) + "\t")

    for i, city in enumerate(nnh_solution.get_ordered_path().get_candidate()):
        op = "\n" if (i == (nb_cities - 1)) else "\t"
        nnh_file.write(str(city) + op)
    nnh_file.write(str(nnh_solution.get_total_weight()) + '\n')

    if choco_ordered_path.get_cartesian_coordinates() is not None:
        for i in range(nb_cities):
            for j in range(2):
                op = "\n" if (j == 1) else "\t"
                nnh_file.write(str(nnh_solution.get_ordered_path().get_cartesian_coordinates()[i, j]) + op)

    nnh_file.close()
    return None


def compute_tsp_nnh_two_opt_solution_from_choco_database(nb_cities, instance_id,
                                                         path=constants.PARAMETER_TSP_NNH_TWO_OPT_DATA_FILES,
                                                         choco_path=constants.PARAMETER_TSP_CHOCO_DATA_FILES,
                                                         time_limit=1):
    """
    Computes the TSP dataSet heuristic solution file "dataSet_<n>_<instance_id>.heuristic" corresponding to the
    instance of the TSP given by the file "dataSet_<n>_<instance_id>.choco"
    WARNING: The weight/distance matrix from this instance has to SYMMETRIC, otherwise an Exception is raised
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered (int)
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied (int)
    :param path: the path from the project root to store TSP nearest neighbor + 2-opt heuristic solution (str)
    :param choco_path: the path from the project root of the TSP choco dataSet file considered (str)
    :param time_limit: time limit to improve the solution with the 2-opt heuristic (unit = seconds, default = 1s) (int)
    """
    choco_ordered_path, total_weight = read_tsp_choco_solution_file(nb_cities, instance_id, choco_path)
    heuristic_path = path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".heuristic"
    heuristic_file = open(heuristic_path, 'w+')
    heuristic_file.write(str(instance_id) + '\n')
    heuristic_file.write(str(nb_cities) + '\n')
    nnh_solution = nNH.NearestNeighborHeuristic(choco_ordered_path.get_weight_matrix(),
                                                choco_ordered_path.get_cartesian_coordinates())
    two_opt_solution = tOpt.TwoOptImprovementHeuristic(nnh_solution.get_ordered_path(), time_limit)

    for i in range(nb_cities):
        for j in range(nb_cities):
            if j == (nb_cities - 1):
                heuristic_file.write(str(two_opt_solution.get_weight_matrix()[i, j]) + "\n")
            else:
                heuristic_file.write(str(two_opt_solution.get_weight_matrix()[i, j]) + "\t")

    for i, city in enumerate(two_opt_solution.get_ordered_path().get_candidate()):
        op = "\n" if (i == (nb_cities - 1)) else "\t"
        heuristic_file.write(str(city) + op)
    heuristic_file.write(str(two_opt_solution.get_total_weight()) + '\n')

    if choco_ordered_path.get_cartesian_coordinates() is not None:
        for i in range(nb_cities):
            for j in range(2):
                op = "\n" if (j == 1) else "\t"
                heuristic_file.write(str(two_opt_solution.get_ordered_path().get_cartesian_coordinates()[i, j]) + op)

    heuristic_file.close()
    return None
