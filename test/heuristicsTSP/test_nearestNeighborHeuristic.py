import unittest
import numpy as np
from src.heuristicsTSP.nearestNeighborHeuristic import NearestNeighborHeuristic
from src.objects.orderedPath import OrderedPath


class TestNearestNeighborHeuristicMethods(unittest.TestCase):
    def test_nearest_neighbor_heuristic_case_1(self):
        # 1 _ _ 4 _ 3 _ 0 _ _ _ 2 ( "_" = 1 unit of distance) (symmetric case)
        distance_matrix_case_1 = np.array([[0, 4, 3, 1, 2],
                                           [4, 0, 7, 3, 2],
                                           [3, 7, 0, 4, 5],
                                           [1, 3, 4, 0, 1],
                                           [2, 2, 5, 1, 0]], dtype=int)
        ordered_path_case_1 = OrderedPath(np.array([0, 3, 4, 1, 2], dtype=int), distance_matrix_case_1)
        expected_total_weight_case_1 = 14  # d = 1+1+2+7+3 = 14
        nearest_neighbor_case_1 = NearestNeighborHeuristic(distance_matrix_case_1)
        self.assertEqual(ordered_path_case_1.distance(), expected_total_weight_case_1)
        self.assertEqual(nearest_neighbor_case_1.get_ordered_path(), ordered_path_case_1)
        self.assertEqual(nearest_neighbor_case_1.get_total_weight(), expected_total_weight_case_1)


if __name__ == '__main__':
    unittest.main()
