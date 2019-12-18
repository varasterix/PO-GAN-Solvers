import numpy as np
from src.Python.objects import objectsTools
import src.Python.objects.orderedPath as oP


def solve_with_nearest_neighbors_heuristic(weight_matrix):
    if not objectsTools.is_weight_matrix_valid_structure(weight_matrix):
        raise Exception('The structure of the weight matrix in input is not valid')
    else:
        total_weight = 0
        nb_cities = len(weight_matrix)
        first_city = 0  # The first city visited is 0
        current_city = first_city
        ordered_path = [current_city]
        unvisited_cities = [i for i in range(1, nb_cities)]  # composed of all cities except the city 0
        while len(unvisited_cities) != 0:
            next_city, city_weight = get_nearest_unvisited_neighbor(weight_matrix, current_city, unvisited_cities)
            total_weight += city_weight
            ordered_path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city
        # The cycle is closed
        total_weight += weight_matrix[current_city, first_city]
        return oP.OrderedPath(np.array(ordered_path, dtype=int), weight_matrix), total_weight


def get_nearest_unvisited_neighbor(weight_matrix, current_city, unvisited_cities):
    next_city = 0
    best_weight = max(weight_matrix[current_city, :])
    for city in unvisited_cities:
        city_weight = weight_matrix[current_city, city]
        if city_weight <= best_weight:
            next_city = city
            best_weight = city_weight
    return next_city, best_weight


class NearestNeighborHeuristic:
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    This class is offering functions which permit give a solution to an instance of the TSP problem with the heuristic
    of the nearest neighbors. The solution can be of any class which extends the abstract class "CandidateTSP"
    """

    def __init__(self, weight_matrix):
        ordered_path, total_weight = solve_with_nearest_neighbors_heuristic(weight_matrix)
        self.__ordered_path = ordered_path
        self.__total_weight = total_weight

    def __eq__(self, other):
        """
        :param other: an object whose class is NearestNeighborHeuristic
        :return: true if the attributes of the object are equal to the attributes of the other object
        """
        if not isinstance(other, NearestNeighborHeuristic):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            return self.__total_weight == other.__total_weight and self.__ordered_path == other.__ordered_path

    def get_weight_matrix(self):
        return self.__ordered_path.get_weight_matrix()

    def get_nb_cities(self):
        return self.__ordered_path.get_nb_cities()

    def get_total_weight(self):
        return self.__total_weight

    def get_ordered_path(self):
        return self.__ordered_path

    def get_ordered_path_binary_matrix(self):
        return self.get_ordered_path().to_ordered_path_binary_matrix()

    def get_neighbours(self):
        return self.get_ordered_path().to_neighbours()

    def get_neighbours_binary_matrix(self):
        return self.get_ordered_path().to_neighbours_binary_matrix()
