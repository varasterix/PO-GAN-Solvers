from src.objects.candidateTSP import CandidateTSP
from src.objects import objectsTools
import src.objects.neighbours as n
import src.objects.orderedPath as oP
import src.objects.neighboursBinaryMatrix as nBM
import numpy as np


class OrderedPathBinaryMatrix(CandidateTSP):
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    Structure: Binary matrix (composed of 0/1) whose jth column has to contain one 1 (and otherwise 0s) at the ith
    index (and the city i is the jth city visited; the order of the columns in the array is significant).
    That is why this structure is called "ordered path binary matrix".
    And, the fact that 2 ordered path are equal is independent of the beginning of the ordered paths
    """

    def __init__(self, candidate, distance_matrix):
        self.__nb_cities = len(candidate)
        self.__binary_matrix = candidate
        self.__distance_matrix = distance_matrix
        self.__is_valid_structure = self.is_valid_structure()

    def __eq__(self, other):
        """
        Note1: The eq function is only comparing two solutions of the TSP
        Note2: The eq function is independent of the beginning of the ordered path of cities
        :param other: an object whose class is OrderedPathBinaryMatrix
        :return: true if the attributes of the object are equal to the attributes of the other object,
        and an exception is one of the objects considered are not a solution to the TSP
        """
        if not isinstance(other, OrderedPathBinaryMatrix):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if not other.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if self.__nb_cities != other.__nb_cities:
                return False
            else:
                is_ordered_path_binary_matrix_equal, h = True, 0
                k = np.where(self.__binary_matrix[0, :] == 1)[0][0]
                m = np.where(other.__binary_matrix[0, :] == 1)[0][0]
                while is_ordered_path_binary_matrix_equal and h < self.__nb_cities:
                    is_ordered_path_binary_matrix_equal = \
                        np.where(self.__binary_matrix[:, k % self.__nb_cities] == 1)[0][0] == \
                        np.where(other.__binary_matrix[:, m % other.__nb_cities] == 1)[0][0]
                    h += 1
                    k += 1
                    m += 1
                is_distance_matrix_equal, i = True, 0
                while is_ordered_path_binary_matrix_equal and is_distance_matrix_equal and i < self.__nb_cities:
                    j = 0
                    while is_distance_matrix_equal and j < self.__nb_cities:
                        is_distance_matrix_equal = self.__distance_matrix[i, j] == other.__distance_matrix[i, j]
                        j += 1
                    i += 1
                return is_ordered_path_binary_matrix_equal and is_distance_matrix_equal

    def __copy__(self):
        return OrderedPathBinaryMatrix(np.copy(self.__binary_matrix), np.copy(self.__distance_matrix))

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
            for j in range(self.__nb_cities):
                neighbour_j = np.where(self.__binary_matrix[:, j] == 1)[0][0]
                if visited[neighbour_j] != 0:
                    # The jth element is a city which has been already visited
                    return False
                visited[neighbour_j] = 1
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
            return sum([self.__distance_matrix[np.where(self.__binary_matrix[:, i] == 1)[0][0],
                                               np.where(self.__binary_matrix[:, (i + 1) % self.__nb_cities] == 1)[0][0]]
                        for i in range(self.__nb_cities)])

    def to_neighbours(self):
        """
        :return: the Neighbours corresponding to the OrderedPathBinaryMatrix object if it corresponds to a solution of
        the TSP, an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            neighbours_array = np.zeros(self.__nb_cities, dtype=int)
            for i in range(self.__nb_cities):
                neighbours_array[np.where(self.__binary_matrix[:, i] == 1)[0][0]] = \
                    np.where(self.__binary_matrix[:, (i + 1) % self.__nb_cities] == 1)[0][0]
            return n.Neighbours(neighbours_array, self.__distance_matrix)

    def to_ordered_path(self):
        """
        :return: the OrderedPath corresponding to the OrderedPathBinaryMatrix object if it has a valid structure,
        an exception otherwise
        """
        if not self.is_valid_structure():
            raise Exception('The candidate has not a valid structure')
        else:
            return oP.OrderedPath(np.array([np.where(self.__binary_matrix[:, i] == 1)[0][0]
                                            for i in range(self.__nb_cities)], dtype=int), self.__distance_matrix)

    def to_neighbours_binary_matrix(self):
        """
        :return: the NeighboursBinaryMatrix corresponding to the OrderedPathBinaryMatrix object if it corresponds to a
        solution of the TSP, an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            neighbours_binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            for i in range(self.__nb_cities):
                neighbours_binary_matrix[np.where(self.__binary_matrix[:, (i + 1) % self.__nb_cities] == 1)[0][0],
                                         np.where(self.__binary_matrix[:, i] == 1)[0][0]] = 1
            return nBM.NeighboursBinaryMatrix(neighbours_binary_matrix, self.__distance_matrix)

    def is_valid_structure(self):
        """
        The structure of the ordered path binary matrix is valid if each column is composed of n integers whose (n-1)
        elements equal to 0 and one equals to 1.
        :return: True if the structure of the ordered path binary matrix is valid, False otherwise
        """
        is_valid_structure = (type(self.__binary_matrix) == np.ndarray and self.__binary_matrix.dtype == int and
                              objectsTools.is_weight_matrix_valid_structure(self.__distance_matrix) and
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
