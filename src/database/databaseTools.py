import numpy as np
import random
from shutil import copyfile
import src.objects.orderedPath as oP
from src.heuristicsTSP import nearestNeighborHeuristic as nNH, twoOptImprovementHeuristic as tOpt

"""
Considering an instance of the TSP, it is defined with its weight/distance matrix <M> of size (n x n) with n, the
number of cities of this considered instance.
Thus, a file in the "test/tsp_database/", which is called "dataSet_<n>_<instance_id>.tsp" is written in this format :
"   <instance_id>   (with <instance_id> between 0 to the number of instances with <n> cities studied)
    <n>             (the number of cities)
    <M>[0,0]   <M>[0,1]    ...  <M>[0,n-1] (with <M>[i,j] the weight/distance from the city i to the city j)
    <M>[1,0]   <M>[1,1]    ...  <M>[1,n-1]
    ...        ...         ...  ...
    <M>[n-1,0] <M>[n-1,1]  ...  <M>[n-1,n-1]    "
Note: For all i in [|0, n-1|], <M>[i,i] = 0

Besides, a file in the same repository", called "dataSet_<n>_<instance_id>.heuristic", which also contains a solution 
to the considered instance of the TSP (in the "dataSet_<n>_<instance_id>.tsp" file), is written in this format :
"   <instance_id>
    <n>
    <M>[0,0]   <M>[0,1]    ...  <M>[0,n-1] (with <M>[i,j] the weight/distance from the city i to the city j)
    <M>[1,0]   <M>[1,1]    ...  <M>[1,n-1]
    ...        ...         ...  ...
    <M>[n-1,0] <M>[n-1,1]  ...  <M>[n-1,n-1]
    <S>[0]  <S>[1]  ... <O>[n-1] (with <S>[i] the (i+1)th city visited in the solution)
    <total_weigh> (the total cost/weight/distance of the solution S for the considered instance)    "
"""


def read_tsp_file(nb_cities, instance_id, path="test/tsp_database/"):
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


def generate_tsp_file(nb_cities, instance_id, path="test/tsp_database/", highest_weight=100, symmetric=False):
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


def compute_tsp_heuristic_solution(nb_cities, instance_id, path="test/tsp_database/", time_limit=1):
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


def read_tsp_heuristic_solution_file(nb_cities, instance_id, path="test/tsp_database/"):
    """
    Imports a solution (object OrderedPath containing the weight/distance matrix) and the total cost/weight/distance of
    this solution corresponding to the TSP dataSet heuristic solution file "dataSet_<n>_<instance_id>.heuristic"
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied
    :param path: the path from the project root of the TSP dataSet file considered
    :return: a solution (object OrderedPath containing the weight/distance matrix) corresponding to the TSP dataSet
    file "dataSet_<n>_<instance_id>.heuristic" considered, and the total cost/weight/distance of this solution
    """
    heuristic_file = open(path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".heuristic", 'r')
    weight_matrix = np.zeros((nb_cities, nb_cities), dtype=int)
    ordered_path, total_weight = [], 0
    for i, line in enumerate(heuristic_file):
        if 2 <= i < (2 + nb_cities):
            for j, w_ij in enumerate(line[:-1].split('\t')):
                weight_matrix[i - 2, j] = int(w_ij)
        if i == (2 + nb_cities):
            ordered_path = [int(city) for city in line[:-1].split('\t')]
        if i == (2 + nb_cities + 1):
            total_weight = int(line[:-1])
    heuristic_file.close()
    return oP.OrderedPath(np.array(ordered_path, dtype=int), weight_matrix), total_weight
