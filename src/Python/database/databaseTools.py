import numpy as np
import random

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


def generate_tsp_file(nb_cities, instance_id, path="test/tsp_database/", highest_weight=100):
    """
    Generates the TSP dataSet file "dataSet_<n>_<instance_id>.tsp" for an instance with a given number of cities
    :param nb_cities: the number of cities of the instance of the TSP dataSet file considered (int)
    :param instance_id: the instance id for the instances of the TSP with "nb_cities" studied (int)
    :param path: the path from the project root of the TSP dataSet file considered (str)
    :param highest_weight: highest integer weight which can be generated in the TSP dataSet file (int)
    :return: the weight/distance matrix corresponding to the TSP dataSet file "dataSet_<n>_<instance_id>.tsp" considered
    """
    tsp_file = open(path + "dataSet_" + str(nb_cities) + "_" + str(instance_id) + ".tsp", 'w+')
    tsp_file.write(str(instance_id) + '\n')
    tsp_file.write(str(nb_cities) + '\n')
    for i in range(nb_cities):
        for j in range(nb_cities):
            if i == j:
                w = 0
            else:
                w = random.randint(0, highest_weight)  # Return a random integer w such that 0 <= w <= highest_weight
            if j == (nb_cities - 1):
                tsp_file.write(str(w) + "\n")
            else:
                tsp_file.write(str(w) + "\t")
    tsp_file.close()
    return None
