import numpy as np
import time
import src.Python.objects.orderedPath as oP
import src.Python.objects.neighbours as n
import src.Python.objects.neighboursBinaryMatrix as nBM
import src.Python.objects.orderedPathBinaryMatrix as oPBM


def solve_with_two_opt_improvement_heuristic(initial_ordered_path, time_limit):
    if not isinstance(initial_ordered_path, oP.OrderedPath):
        raise NotImplemented
    else:
        if not initial_ordered_path.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            # Time initialisation
            start_time = time.time()
            spent_time = 0
            # Variables initialisation
            nb_cities = initial_ordered_path.get_nb_cities()
            ordered_path_copy = initial_ordered_path.__copy__()
            weight_matrix = ordered_path_copy.get_weight_matrix()
            memory_ordered_path_array = ordered_path_copy.get_candidate()
            memory_total_weight = ordered_path_copy.distance()

            while spent_time < time_limit:
                current_o_p_array = np.copy(memory_ordered_path_array)
                current_total_weight = memory_total_weight
                for i in range(1, nb_cities - 1):  # One city is fixed
                    for j in range(i + 1, nb_cities):
                        new_ordered_path_array = np.copy(current_o_p_array)
                        two_opt_swap(new_ordered_path_array, i, j)
                        new_total_weight = (current_total_weight
                                            - weight_matrix[current_o_p_array[i - 1], current_o_p_array[i]]
                                            - weight_matrix[current_o_p_array[j], current_o_p_array[(j+1) % nb_cities]]
                                            + weight_matrix[current_o_p_array[i - 1], current_o_p_array[j]]
                                            + weight_matrix[current_o_p_array[i], current_o_p_array[(j+1) % nb_cities]])
                        if new_total_weight < current_total_weight:
                            current_total_weight = new_total_weight
                            current_o_p_array = new_ordered_path_array

                if memory_total_weight == current_total_weight:
                    spent_time = time_limit  # The end of the function is forced
                else:
                    spent_time = time.time() - start_time
                    memory_total_weight = current_total_weight
                    memory_ordered_path_array = np.copy(current_o_p_array)

            return oP.OrderedPath(memory_ordered_path_array, weight_matrix), memory_total_weight


def two_opt_swap(ordered_path_array, i, j):
    """
    Swaps the segment of elements between the ith and the jth positions included in the given array "ordered_path_array"
    :param ordered_path_array: array of integers whose ith element is the ith city visited (size n)
    :param i: index of the first city in the segment to swap (i<j)
    :param j: index of the last city in the segment to swap (j<n)
    :return: Side effect: the array ordered_path_array has its segment between ith and jth elements included swapped,
    if the i < j, an exception otherwise
    """
    if i < j:
        cities_mem = np.copy(ordered_path_array)[i:(j + 1)]  # select ith to jth elements included (j+1-i elements)
        for h, city_mem in enumerate(cities_mem):
            ordered_path_array[j - h] = city_mem
        return None
    else:
        raise Exception('The indices i and j of the two swap operation have to be such as i<j<n')


class TwoOptImprovementHeuristic:
    """
    Note: Considering an instance of the TSP, "n" represents the number of cities of this instance.
    To solve the instance of the TSP, each cities (represented by an integer between 0 and (n-1)) has to be visited.

    This class is offering functions which permit give a better solution to an instance of the TSP problem given an
    initial solution to the instance of the TSP problem with the 2-opt improvement heuristic (given a time limit).
    The initial and improved solution can be of any class which extends the abstract class "CandidateTSP"
    """

    def __init__(self, initial_solution, time_limit=1):
        """
        Computes an improved solution of a given initial solution of an instance of the TSP with the 2-opt improvement
        heuristic in a given time limit
        :param initial_solution: solution of an instance of the TSP problem (can be of any class which extends the
        abstract class "CandidateTSP")
        :param time_limit: time limit to compute the improved solution (unit = seconds, default time_limit = 1s)
        """
        if isinstance(initial_solution, oP.OrderedPath):
            initial_ordered_path = initial_solution
        elif isinstance(initial_solution, oPBM.OrderedPathBinaryMatrix):
            initial_ordered_path = initial_solution.to_ordered_path()
        elif isinstance(initial_solution, n.Neighbours):
            initial_ordered_path = initial_solution.to_ordered_path()
        elif isinstance(initial_solution, nBM.NeighboursBinaryMatrix):
            initial_ordered_path = initial_solution.to_ordered_path()
        else:
            raise Exception('The initial_solution has to be from any class extending the abstract class CandidateTSP')
        ordered_path, total_weight = solve_with_two_opt_improvement_heuristic(initial_ordered_path, time_limit)
        self.__ordered_path = ordered_path
        self.__total_weight = total_weight

    def __eq__(self, other):
        """
        :param other: an object whose class is TwoOptImprovementHeuristic
        :return: true if the attributes of the object are equal to the attributes of the other object
        """
        if not isinstance(other, TwoOptImprovementHeuristic):  # don't attempt to compare against unrelated types
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
