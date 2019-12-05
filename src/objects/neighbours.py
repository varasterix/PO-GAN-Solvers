from src.objects.candidateTSP import CandidateTSP
from src.objects import objectsTools
import src.objects.neighboursBinaryMatrix as nBM
import src.objects.orderedPath as oP
import src.objects.orderedPathBinaryMatrix as oPBM
import numpy as np


class Neighbours(CandidateTSP):
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    Structure: Array of integers whose ith element is the city visited after the ith one.
    That is why this structure is called array of neighbours.
    """

    def __init__(self, candidate, distance_matrix):
        self.__nb_cities = len(candidate)
        self.__neighbours_array = candidate
        self.__distance_matrix = distance_matrix
        self.__is_valid_structure = self.is_valid_structure()

    def __eq__(self, other):
        """
        Note: The eq function is only comparing two solutions of the TSP
        :param other: an object whose class is Neighbours
        :return: true if the attributes of the object are equal to the attributes of the other object,
        and an exception is one of the objects considered are not a solution to the TSP
        """
        if not isinstance(other, Neighbours):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if not other.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if self.__nb_cities != other.__nb_cities:
                return False
            else:
                is_neighbours_array_equal, k = True, 0
                while is_neighbours_array_equal and k < self.__nb_cities:
                    is_neighbours_array_equal = self.__neighbours_array[k] == other.__neighbours_array[k]
                    k += 1
                is_distance_matrix_equal, i = True, 0
                while is_neighbours_array_equal and is_distance_matrix_equal and i < self.__nb_cities:
                    j = 0
                    while is_distance_matrix_equal and j < self.__nb_cities:
                        is_distance_matrix_equal = self.__distance_matrix[i][j] == other.__distance_matrix[i][j]
                        j += 1
                    i += 1
                return is_neighbours_array_equal and is_distance_matrix_equal

    def get_nb_cities(self):
        return self.__nb_cities

    def get_candidate(self):
        return self.__neighbours_array

    def get_weight_matrix(self):
        return self.__distance_matrix

    def is_solution(self):
        """
        :return True if the array of integers is a solution of the TSP, False otherwise
        """
        if not self.__is_valid_structure:
            return False
        else:
            visited = [0] * self.__nb_cities
            current_city = 0
            for i in range(self.__nb_cities):  # There is only one cycle
                neighbour_city = self.__neighbours_array[current_city]
                if visited[neighbour_city] != 0:
                    # The ith neighbour is a city which has been already visited
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
            return sum([self.__distance_matrix[i, neighbour_i] for i, neighbour_i in enumerate(self.__neighbours_array)])

    def to_neighbours_binary_matrix(self):
        """
        :return: the NeighboursBinaryMatrix corresponding to the Neighbours object if it corresponds to a solution,
        an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            for i in range(self.__nb_cities):
                binary_matrix[self.__neighbours_array[i], i] = 1
            return nBM.NeighboursBinaryMatrix(binary_matrix, self.__distance_matrix)

    def to_ordered_path(self):
        """
        :return: the OrderedPath corresponding to the Neighbours object if it corresponds to a solution,
        an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            ordered_path = []
            current_city = 0
            for i in range(self.__nb_cities):
                ordered_path.append(self.__neighbours_array[current_city])
                current_city = self.__neighbours_array[current_city]
            return oP.OrderedPath(np.array(ordered_path, dtype=int), self.__distance_matrix)

    def to_ordered_path_binary_matrix(self):
        """
        :return: the OrderedPathBinaryMatrix corresponding to the Neighbours object if it corresponds to a solution,
        an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            path_binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            current_city = 0
            for i in range(self.__nb_cities):
                path_binary_matrix[self.__neighbours_array[current_city], i] = 1
                current_city = self.__neighbours_array[current_city]
            return oPBM.OrderedPathBinaryMatrix(path_binary_matrix, self.__distance_matrix)

    def is_valid_structure(self):
        """
        The structure of the neighbours array is valid if each element of the array is an integer between [0, (n-1)]
        and if the distance matrix of integers is of size n x n
        :return: True if the structure of the neighbours array object is valid, False otherwise
        """
        is_valid_structure = (type(self.__neighbours_array) == np.ndarray and self.__neighbours_array.dtype == int and
                              objectsTools.__is_weight_matrix_valid_structure(self.__distance_matrix) and
                              len(self.__distance_matrix) == self.__nb_cities)
        if is_valid_structure:
            i = 0
            while is_valid_structure and i < self.__nb_cities:
                neighbour_i = self.__neighbours_array[i]
                if neighbour_i >= self.__nb_cities or neighbour_i < 0:
                    # The ith neighbour does not represent a city
                    is_valid_structure = False
                i += 1
        return is_valid_structure
