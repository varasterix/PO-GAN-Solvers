import numpy as np
import matplotlib.pyplot as plt
from src.objects.candidateTSP import CandidateTSP
from src.objects import objectsTools
import src.objects.neighboursBinaryMatrix as nBM
import src.objects.neighbours as n
import src.objects.orderedPathBinaryMatrix as oPBM


class OrderedPath(CandidateTSP):
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    Structure: Array of integers whose ith element is the ith city visited (the order in the array is significant).
    That is why this structure is called "ordered path".
    And, the fact that 2 ordered paths are equal is independent of the beginning of the ordered paths
    """

    def __init__(self, candidate, distance_matrix, cartesian_coordinates=None):
        self.__nb_cities = len(candidate)
        self.__ordered_path = candidate
        self.__distance_matrix = distance_matrix
        self.__cartesian_coordinates = cartesian_coordinates
        self.__is_valid_structure = self.is_valid_structure()

    def __eq__(self, other):
        """
        Note1: The eq function is only comparing two solutions of the TSP
        Note2: The eq function is independent of the beginning of the ordered array of cities
        :param other: an object whose class is OrderedPath
        :return: true if the attributes of the object (except the cartesian coordinates) are equal to the attributes of
        the other object, and an exception is one of the objects considered are not a solution to the TSP
        """
        if not isinstance(other, OrderedPath):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if not other.is_solution():
                raise Exception('The eq function is only comparing two solutions of the TSP')
            if self.__nb_cities != other.__nb_cities:
                return False
            else:
                is_ordered_path_equal, h = True, 0
                k = np.where(self.__ordered_path == 0)[0][0]
                m = np.where(other.__ordered_path == 0)[0][0]
                while is_ordered_path_equal and h < self.__nb_cities:
                    is_ordered_path_equal = \
                        self.__ordered_path[k % self.__nb_cities] == other.__ordered_path[m % other.__nb_cities]
                    h += 1
                    k += 1
                    m += 1
                is_distance_matrix_equal, i = True, 0
                while is_ordered_path_equal and is_distance_matrix_equal and i < self.__nb_cities:
                    j = 0
                    while is_distance_matrix_equal and j < self.__nb_cities:
                        is_distance_matrix_equal = self.__distance_matrix[i, j] == other.__distance_matrix[i, j]
                        j += 1
                    i += 1
                return is_ordered_path_equal and is_distance_matrix_equal

    def __copy__(self):
        return OrderedPath(np.copy(self.__ordered_path), np.copy(self.__distance_matrix),
                           None if self.__cartesian_coordinates is None else np.copy(self.__cartesian_coordinates))

    def to_ordered_path(self):
        return self.__copy__()

    def get_nb_cities(self):
        return self.__nb_cities

    def get_candidate(self):
        return self.__ordered_path

    def get_weight_matrix(self):
        return self.__distance_matrix

    def get_cartesian_coordinates(self):
        return self.__cartesian_coordinates

    def is_solution(self):
        """
        :return True if the array of integers is a solution of the TSP, False otherwise
        """
        if not self.__is_valid_structure:
            return False
        else:
            visited = [0] * self.__nb_cities
            for neighbour_i in self.__ordered_path:
                if visited[neighbour_i] != 0:
                    # The ith element is a city which has been already visited
                    return False
                visited[neighbour_i] = 1
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
            return int(
                sum([self.__distance_matrix[self.__ordered_path[i], self.__ordered_path[(i + 1) % self.__nb_cities]]
                     for i in range(self.__nb_cities)]))

    def to_neighbours(self):
        """
        :return: the Neighbours object corresponding to the OrderedPath object if it corresponds to a solution of the
        TSP, an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            neighbours_array = np.zeros(self.__nb_cities, dtype=int)
            for i in range(self.__nb_cities):
                neighbours_array[self.__ordered_path[i]] = self.__ordered_path[(i + 1) % self.__nb_cities]
            return n.Neighbours(neighbours_array, self.__distance_matrix, self.__cartesian_coordinates)

    def to_neighbours_binary_matrix(self):
        """
        :return: the NeighboursBinaryMatrix corresponding to the OrderedPath object if it corresponds to a solution of
        the TSP, an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            for i in range(self.__nb_cities):
                binary_matrix[self.__ordered_path[(i + 1) % self.__nb_cities], self.__ordered_path[i]] = 1
            return nBM.NeighboursBinaryMatrix(binary_matrix, self.__distance_matrix, self.__cartesian_coordinates)

    def to_ordered_path_binary_matrix(self):
        """
        :return: the OrderedPathBinaryMatrix corresponding to the OrderedPath object if it has a valid structure,
        an exception otherwise
        """
        if not self.is_valid_structure():
            raise Exception('The candidate has not a valid structure')
        else:
            path_binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            for i in range(self.__nb_cities):
                path_binary_matrix[self.__ordered_path[i], i] = 1
            return oPBM.OrderedPathBinaryMatrix(path_binary_matrix, self.__distance_matrix,
                                                self.__cartesian_coordinates)

    def is_valid_structure(self):
        """
        The structure of the ordered path is valid if each element of the array is an integer between [0, (n-1)]
        and if the distance matrix of integers is of size n x n
        :return: True if the structure of the ordered path object is valid, False otherwise
        """
        is_valid_structure = (type(self.__ordered_path) == np.ndarray and self.__ordered_path.dtype == int and
                              objectsTools.is_weight_matrix_valid_structure(self.__distance_matrix) and
                              len(self.__distance_matrix) == self.__nb_cities and
                              (self.__cartesian_coordinates is None or
                              (objectsTools.is_cartesian_coordinates_valid_structure(self.__cartesian_coordinates) and
                               self.__cartesian_coordinates.shape[0] == self.__nb_cities)))
        if is_valid_structure:
            i = 0
            while is_valid_structure and i < self.__nb_cities:
                neighbour_i = self.__ordered_path[i]
                if neighbour_i >= self.__nb_cities or neighbour_i < 0:
                    # The ith element does not represent a city
                    is_valid_structure = False
                i += 1
        return is_valid_structure

    def get_nb_duplicates(self):
        done = [0] * self.__nb_cities
        nb_duplicates = 0
        if not self.is_valid_structure():
            raise Exception('The candidate has not a valid structure')
        else:
            for i in range(self.__nb_cities):
                city_i = self.__ordered_path[i]
                if done[city_i] == 0:
                    done[city_i] = 1
                else:
                    nb_duplicates += 1
            return nb_duplicates

    def plot(self):
        if not self.is_valid_structure():
            raise Exception('The candidate has not a valid structure')
        elif self.get_cartesian_coordinates() is None:
            raise Exception('There are no cartesian coordinates for this object')
        else:
            annotation_gap = 10
            label = "Not a TSP solution" if not self.is_solution() else "Solution, D=" + str(self.distance())
            plt.figure("TSP candidate figure")
            plt.title("TSP candidate - Representation of the cycle")
            for i, (x, y) in enumerate(self.get_cartesian_coordinates()):
                plt.plot(x, y, "ok")
                plt.annotate(i, (x + annotation_gap, y + annotation_gap))
            x_seq, y_seq = [], []
            for city in self.get_candidate():
                x_seq.append(self.get_cartesian_coordinates()[city, 0])
                y_seq.append(self.get_cartesian_coordinates()[city, 1])
            x_seq.append(self.get_cartesian_coordinates()[self.get_candidate()[0], 0])
            y_seq.append(self.get_cartesian_coordinates()[self.get_candidate()[0], 1])
            plt.plot(x_seq, y_seq, '-b', label=label)
            plt.legend()
            plt.show()
