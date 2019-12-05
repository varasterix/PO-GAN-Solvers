from src.objects.candidateTSP import CandidateTSP
from src.objects import objectsTools
import src.objects.neighbours as n
import src.objects.orderedPath as oP
import src.objects.orderedPathBinaryMatrix as oPBM
import numpy as np


class NeighboursBinaryMatrix(CandidateTSP):
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    Structure: Binary matrix (composed of 0/1) whose jth column has to contain one 1 (and otherwise 0s) at the ith
    index (and the city i is the city visited after the city j).
    """

    def __init__(self, candidate, distance_matrix):
        self.__nb_cities = len(candidate)
        self.__binary_matrix = candidate
        self.__distance_matrix = distance_matrix
        self.__is_valid_structure = self.is_valid_structure()

    def __eq__(self, other):
        """
        Note: The eq function is only comparing two solutions of the TSP
        :param other: an object whose class is NeighboursBinaryMatrix
        :return: true if the attributes of the object are equal to the attributes of the other object,
        and an exception is one of the objects considered are not a solution to the TSP
        """
        if not isinstance(other, NeighboursBinaryMatrix):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if not other.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if self.__nb_cities != other.__nb_cities:
                return False
            else:
                is_neighbours_binary_matrix_equal, k = True, 0
                while is_neighbours_binary_matrix_equal and k < self.__nb_cities:
                    is_neighbours_binary_matrix_equal = np.where(self.__binary_matrix[:, k] == 1)[0][0] == \
                                                        np.where(other.__binary_matrix[:, k] == 1)[0][0]
                    k += 1
                is_distance_matrix_equal, i = True, 0
                while is_neighbours_binary_matrix_equal and is_distance_matrix_equal and i < self.__nb_cities:
                    j = 0
                    while is_distance_matrix_equal and j < self.__nb_cities:
                        is_distance_matrix_equal = self.__distance_matrix[i][j] == other.__distance_matrix[i][j]
                        j += 1
                    i += 1
                return is_neighbours_binary_matrix_equal and is_distance_matrix_equal

    def get_nb_cities(self):
        return self.__nb_cities

    def get_candidate(self):
        return self.__binary_matrix

    def get_weight_matrix(self):
        return self.__distance_matrix

    def is_solution(self):
        """
        :return True if the binary matrix (composed of 0/1) is a solution of the TSP, False otherwise
        """
        if not self.__is_valid_structure:
            return False
        else:
            visited = [0] * self.__nb_cities
            current_city = 0
            for j in range(self.__nb_cities):  # There is only one cycle
                neighbour_city = np.where(self.__binary_matrix[:, current_city] == 1)[0][0]
                if visited[neighbour_city] != 0:
                    # The jth neighbour is a city which has been already visited
                    return False
                current_city = neighbour_city
                visited[neighbour_city] = 1
            return True

    def distance(self):
        """
        Computes the cost/objective function of the considered of the instance of a TSP
        :return: the total distance of the candidate of the instance of a TSP if the candidate is a solution,
        an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            return sum([self.__distance_matrix[i, neighbour_i] for i, neighbour_i in
                        enumerate([np.where(self.__binary_matrix[:, j] == 1)[0][0] for j in range(self.__nb_cities)])])

    def to_neighbours(self):
        """
        :return: the Neighbours corresponding to the NeighboursBinaryMatrix object if it corresponds to a solution,
        an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            return n.Neighbours(np.array([np.where(self.__binary_matrix[:, j] == 1)[0][0]
                                          for j in range(self.__nb_cities)], dtype=int), self.__distance_matrix)

    def to_ordered_path(self):
        """
        :return: the OrderedPath corresponding to the NeighboursBinaryMatrix object if it corresponds to a solution,
        an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            ordered_path = []
            current_city = 0
            for i in range(self.__nb_cities):
                neighbour_city = np.where(self.__binary_matrix[:, current_city] == 1)[0][0]
                ordered_path.append(neighbour_city)
                current_city = neighbour_city
            return oP.OrderedPath(np.array(ordered_path, dtype=int), self.__distance_matrix)

    def to_ordered_path_binary_matrix(self):
        """
        :return: the OrderedPathBinaryMatrix corresponding to the NeighboursBinaryMatrix object if it corresponds to a
        solution, an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            path_binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            current_city = 0
            for i in range(self.__nb_cities):
                neighbour_city = np.where(self.__binary_matrix[:, current_city] == 1)[0][0]
                path_binary_matrix[neighbour_city, i] = 1
                current_city = neighbour_city
            return oPBM.OrderedPathBinaryMatrix(path_binary_matrix, self.__distance_matrix)

    def is_valid_structure(self):
        """
        The structure of the neighbours binary matrix is valid if each column is composed of n integers whose (n-1)
        elements equal to 0 and one equals to 1.
        :return: True if the structure of the neighbours binary matrix is valid, False otherwise
        """
        is_valid_structure = (type(self.__binary_matrix) == np.ndarray and self.__binary_matrix.dtype == int and
                              objectsTools.__is_weight_matrix_valid_structure(self.__distance_matrix) and
                              len(self.__distance_matrix) == self.__nb_cities and
                              self.__binary_matrix.shape[1] == self.__nb_cities and
                              self.__binary_matrix.shape[0] == self.__nb_cities)
        if is_valid_structure:
            j = 0
            while is_valid_structure and j < self.__nb_cities:  # verify if there is one 1 and (n-1) 0 by column
                city_bin_j = self.__binary_matrix[:, j]
                is_valid_structure = (len(np.where(city_bin_j == 1)[0]) == 1 and
                                      len(np.where(city_bin_j == 0)[0]) == (self.__nb_cities - 1))
                j += 1
        return is_valid_structure
