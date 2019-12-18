import unittest
import numpy as np
from src.Python.heuristicsTSP import TwoOptImprovementHeuristic, two_opt_swap
from src.objects.orderedPath import OrderedPath

distance_matrix_5 = np.array([[0, 5, 1, 10, 1],
                              [1, 0, 1, 8, 1],
                              [1, 15, 0, 4, 6],
                              [8, 1, 2, 0, 2],
                              [9, 7, 0, 1, 0]], dtype=int)
ordered_path_1 = OrderedPath(np.array([0, 1, 2, 3, 4]), distance_matrix_5)  # solution d = 5+1+4+2+9 = 21
ordered_path_2 = OrderedPath(np.array([0, 1, 0, 1, 4]), distance_matrix_5)


class TestTwoOptImprovementHeuristicMethods(unittest.TestCase):
    def test_two_opt_swap(self):
        ordered_path_array = np.array([4, 5, 2, 3, 0, 1], dtype=int)
        expected_path_array = np.array([4, 0, 3, 2, 5, 1], dtype=int)
        self.assertFalse((ordered_path_array == expected_path_array).all())
        two_opt_swap(ordered_path_array, 1, 4)
        self.assertTrue((ordered_path_array == expected_path_array).all())
        ordered_path_array_1 = ordered_path_1.__copy__().get_candidate()
        self.assertTrue((ordered_path_1.get_candidate() == ordered_path_array_1).all())
        two_opt_swap(ordered_path_array_1, 0, 3)
        self.assertFalse((ordered_path_1.get_candidate() == ordered_path_array_1).all())

    def test_two_opt_improvement_heuristic_case_1(self):
        # 1 _ _ 4 _ 3 _ 0 _ _ _ 2 ( "_" = 1 unit of distance) (symmetric case)
        time_limit_case_1 = 0
        distance_matrix_case_1 = np.array([[0, 4, 3, 1, 2],
                                           [4, 0, 7, 3, 2],
                                           [3, 7, 0, 4, 5],
                                           [1, 3, 4, 0, 1],
                                           [2, 2, 5, 1, 0]], dtype=int)
        ordered_path_case_1 = OrderedPath(np.array([0, 3, 4, 1, 2], dtype=int), distance_matrix_case_1)  # d = 14
        case_1a = TwoOptImprovementHeuristic(ordered_path_case_1, time_limit_case_1)
        case_1b = TwoOptImprovementHeuristic(ordered_path_case_1.to_neighbours_binary_matrix(), time_limit_case_1)
        case_1c = TwoOptImprovementHeuristic(ordered_path_case_1.to_neighbours(), time_limit_case_1)
        case_1d = TwoOptImprovementHeuristic(ordered_path_case_1.to_neighbours(), time_limit_case_1)
        self.assertEqual(case_1a.get_ordered_path(), ordered_path_case_1)
        self.assertEqual(case_1b.get_ordered_path(), ordered_path_case_1)
        self.assertEqual(case_1c.get_ordered_path(), ordered_path_case_1)
        self.assertEqual(case_1d.get_ordered_path(), ordered_path_case_1)
        self.assertEqual(case_1a.get_total_weight(), 14)
        self.assertEqual(case_1b.get_total_weight(), 14)
        self.assertEqual(case_1c.get_total_weight(), 14)
        self.assertEqual(case_1d.get_total_weight(), 14)
        with self.assertRaises(Exception):
            TwoOptImprovementHeuristic(ordered_path_case_1.get_candidate(), time_limit_case_1)
            TwoOptImprovementHeuristic(ordered_path_2, time_limit_case_1)

    def test_two_opt_improvement_heuristic_case_2(self):
        # 1 _ _ 4 _ 3 _ 0 _ _ _ 2 ( "_" = 1 unit of distance) (symmetric case) (best result: d=14)
        distance_matrix_case_2 = np.array([[0, 4, 3, 1, 2],
                                           [4, 0, 7, 3, 2],
                                           [3, 7, 0, 4, 5],
                                           [1, 3, 4, 0, 1],
                                           [2, 2, 5, 1, 0]], dtype=int)
        # d = 4+7+4+1+2 = 17
        ordered_path_case_2 = OrderedPath(np.array([0, 1, 2, 3, 4], dtype=int), distance_matrix_case_2)
        case_2 = TwoOptImprovementHeuristic(ordered_path_case_2)
        self.assertTrue(case_2.get_total_weight() < ordered_path_case_2.distance())
        ordered_path_case_1 = OrderedPath(np.array([0, 3, 4, 1, 2], dtype=int), distance_matrix_case_2)  # d = 14
        self.assertTrue(case_2.get_total_weight() >= 14)


if __name__ == '__main__':
    unittest.main()
