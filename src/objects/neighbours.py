import numpy as np
import matplotlib.pyplot as plt
from src.objects.candidateTSP import CandidateTSP
from src.objects import objectsTools
import src.objects.neighboursBinaryMatrix as nBM
import src.objects.orderedPath as oP
import src.objects.orderedPathBinaryMatrix as oPBM


class Neighbours(CandidateTSP):
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    Structure: Array of integers whose ith element is the city visited after the ith one.
    That is why this structure is called array of neighbours.
    """

    def __init__(self, candidate, distance_matrix, cartesian_coordinates=None):
        self.__nb_cities = len(candidate)
        self.__neighbours_array = candidate
        self.__distance_matrix = distance_matrix
        self.__cartesian_coordinates = cartesian_coordinates
        self.__is_valid_structure = self.is_valid_structure()

    def __eq__(self, other):
        """
        Note: The eq function is only comparing two solutions of the TSP
        :param other: an object whose class is Neighbours
        :return: true if the attributes of the object (except the cartesian coordinates) are equal to the attributes of
        the other object, and an exception is one of the objects considered are not a solution to the TSP
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

    def __copy__(self):
        return Neighbours(np.copy(self.__neighbours_array), np.copy(self.__distance_matrix),
                          None if self.__cartesian_coordinates is None else np.copy(self.__cartesian_coordinates))

    def to_neighbours(self):
        return self.__copy__()

    def get_nb_cities(self):
        return self.__nb_cities

    def get_candidate(self):
        return self.__neighbours_array

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
            return int(sum([self.__distance_matrix[i, neighbor_i]
                            for i, neighbor_i in enumerate(self.__neighbours_array)]))

    def to_neighbours_binary_matrix(self):
        """
        :return: the NeighboursBinaryMatrix corresponding to the Neighbours object if it has a valid structure,
        an exception otherwise
        """
        if not self.is_valid_structure():
            raise Exception('The candidate has not a valid structure')
        else:
            binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            for i in range(self.__nb_cities):
                binary_matrix[self.__neighbours_array[i], i] = 1
            return nBM.NeighboursBinaryMatrix(binary_matrix, self.__distance_matrix,
                                              self.__cartesian_coordinates)

    def to_ordered_path(self):
        """
        :return: the OrderedPath corresponding to the Neighbours object if it corresponds to a solution of the TSP,
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
            return oP.OrderedPath(np.array(ordered_path, dtype=int), self.__distance_matrix,
                                  self.__cartesian_coordinates)

    def to_ordered_path_binary_matrix(self):
        """
        :return: the OrderedPathBinaryMatrix corresponding to the Neighbours object if it corresponds to a solution of
        the TSP, an exception otherwise
        """
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            path_binary_matrix = np.zeros((self.__nb_cities, self.__nb_cities), dtype=int)
            current_city = 0
            for i in range(self.__nb_cities):
                path_binary_matrix[self.__neighbours_array[current_city], i] = 1
                current_city = self.__neighbours_array[current_city]
            return oPBM.OrderedPathBinaryMatrix(path_binary_matrix, self.__distance_matrix,
                                                self.__cartesian_coordinates)

    def is_valid_structure(self):
        """
        The structure of the neighbours array is valid if each element of the array is an integer between [0, (n-1)]
        and if the distance matrix of integers is of size n x n
        :return: True if the structure of the neighbours array object is valid, False otherwise
        """
        is_valid_structure = (type(self.__neighbours_array) == np.ndarray and self.__neighbours_array.dtype == int and
                              objectsTools.is_weight_matrix_valid_structure(self.__distance_matrix) and
                              len(self.__distance_matrix) == self.__nb_cities and
                              ((objectsTools.is_cartesian_coordinates_valid_structure(self.__cartesian_coordinates) and
                                self.__cartesian_coordinates.shape[0] == self.__nb_cities)
                               or self.__cartesian_coordinates is None))
        if is_valid_structure:
            i = 0
            while is_valid_structure and i < self.__nb_cities:
                neighbour_i = self.__neighbours_array[i]
                if neighbour_i >= self.__nb_cities or neighbour_i < 0:
                    # The ith neighbour does not represent a city
                    is_valid_structure = False
                i += 1
        return is_valid_structure

    def get_nb_cycles(self):
        """
        Counts the number of cycles in the considered object. If the ith neighbor is i, it is counted as a cycle.
        :return the number of cycles in the considered object if it has a valid structure, an exception otherwise
        """
        if not self.is_valid_structure():
            raise Exception('The candidate has not a valid structure')
        else:
            nb_cities = self.get_nb_cities()
            visited = [0] * nb_cities
            nb_cycles = 0
            for i in range(nb_cities):
                j = i
                if visited[j] == 0:
                    nb_cycles += 1
                    # first_index = j
                    while visited[j] < 1:
                        visited[j] += 1
                        j = self.__neighbours_array[j]
                    # if j != first_index:
                        # return 0
            return nb_cycles

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
            for city_i in range(self.get_nb_cities()):
                x_seq = [self.get_cartesian_coordinates()[city_i, 0],
                         self.get_cartesian_coordinates()[self.__neighbours_array[city_i], 0]]
                y_seq = [self.get_cartesian_coordinates()[city_i, 1],
                         self.get_cartesian_coordinates()[self.__neighbours_array[city_i], 1]]
                if city_i == 0:
                    plt.plot(x_seq, y_seq, '-b', label=label)
                else:
                    plt.plot(x_seq, y_seq, '-b')
            plt.legend()
            plt.show()
