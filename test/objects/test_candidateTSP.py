import unittest
import numpy as np
from src.Python.objects.neighbours import Neighbours
from src.Python.objects.neighboursBinaryMatrix import NeighboursBinaryMatrix
from src.Python.objects.orderedPath import OrderedPath
from src.Python.objects.orderedPathBinaryMatrix import OrderedPathBinaryMatrix

distance_matrix_5 = np.array([[0, 5, 1, 10, 1],
                              [1, 0, 1, 8, 1],
                              [1, 15, 0, 4, 6],
                              [8, 1, 2, 0, 2],
                              [9, 7, 0, 1, 0]], dtype=int)
# solution d = 5+1+4+2+9 = 21
neighbours_1 = Neighbours(np.array([1, 2, 3, 4, 0]), distance_matrix_5)
neighbours_binary_matrix_1 = NeighboursBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                              [1, 0, 0, 0, 0],
                                                              [0, 1, 0, 0, 0],
                                                              [0, 0, 1, 0, 0],
                                                              [0, 0, 0, 1, 0]], dtype=int), distance_matrix_5)
ordered_path_1 = OrderedPath(np.array([0, 1, 2, 3, 4]), distance_matrix_5)
ordered_path_binary_matrix_1 = \
    OrderedPathBinaryMatrix(np.array([[1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1]], dtype=int), distance_matrix_5)
# solution d = 8+1+0+15+8 = 32
ordered_path_2 = OrderedPath(np.array([3, 0, 4, 2, 1], dtype=int), distance_matrix_5)


class TestCandidateTSPMethods(unittest.TestCase):
    def test_comparison_functions_le(self):
        self.assertTrue(neighbours_1 <= ordered_path_1)
        self.assertTrue(ordered_path_1 <= neighbours_binary_matrix_1)
        self.assertTrue(neighbours_binary_matrix_1 <= ordered_path_binary_matrix_1)
        self.assertTrue(ordered_path_binary_matrix_1 <= neighbours_1)
        self.assertTrue(ordered_path_1 <= ordered_path_2)
        self.assertTrue(neighbours_1 <= ordered_path_2.to_ordered_path_binary_matrix())
        with self.assertRaises(Exception):
            var1 = ordered_path_1 <= OrderedPath(np.array([2, 1, 3, 4, 0]), np.ones((5, 5), dtype=int))
            var2 = neighbours_binary_matrix_1 <= OrderedPath(np.array([0, 1, 0, 4, 0]), distance_matrix_5)

    def test_comparison_functions_ge(self):
        self.assertTrue(neighbours_1 >= ordered_path_1)
        self.assertTrue(ordered_path_1 >= neighbours_binary_matrix_1)
        self.assertTrue(neighbours_binary_matrix_1 >= ordered_path_binary_matrix_1)
        self.assertTrue(ordered_path_binary_matrix_1 >= neighbours_1)
        self.assertTrue(ordered_path_2.to_neighbours_binary_matrix() >= ordered_path_1)
        self.assertTrue(ordered_path_2.to_ordered_path_binary_matrix() >= neighbours_1)
        with self.assertRaises(Exception):
            var1 = ordered_path_1 >= OrderedPath(np.array([2, 1, 3, 4, 0]), np.ones((5, 5), dtype=int))
            var2 = ordered_path_binary_matrix_1 >= OrderedPath(np.array([0, 1, 0, 4, 0]), distance_matrix_5)

    def test_comparison_functions_gt(self):
        self.assertFalse(neighbours_1 > ordered_path_1)
        self.assertFalse(ordered_path_1 > neighbours_binary_matrix_1)
        self.assertFalse(neighbours_binary_matrix_1 > ordered_path_binary_matrix_1)
        self.assertFalse(ordered_path_binary_matrix_1 > neighbours_1)
        self.assertTrue(ordered_path_2.to_neighbours_binary_matrix() > ordered_path_1)
        self.assertTrue(ordered_path_2.to_ordered_path_binary_matrix() > neighbours_1)
        with self.assertRaises(Exception):
            var1 = ordered_path_1 > OrderedPath(np.array([2, 1, 3, 4, 0]), np.ones((5, 5), dtype=int))
            var2 = ordered_path_binary_matrix_1 > OrderedPath(np.array([0, 1, 0, 4, 0]), distance_matrix_5)

    def test_comparison_functions_lt(self):
        self.assertFalse(neighbours_1 < ordered_path_1)
        self.assertFalse(ordered_path_1 < neighbours_binary_matrix_1)
        self.assertFalse(neighbours_binary_matrix_1 < ordered_path_binary_matrix_1)
        self.assertFalse(ordered_path_binary_matrix_1 < neighbours_1)
        self.assertTrue(ordered_path_1 < ordered_path_2)
        self.assertTrue(neighbours_1 < ordered_path_2.to_ordered_path_binary_matrix())
        with self.assertRaises(Exception):
            var1 = ordered_path_1 < OrderedPath(np.array([2, 1, 3, 4, 0]), np.ones((5, 5), dtype=int))
            var2 = neighbours_binary_matrix_1 < OrderedPath(np.array([0, 1, 0, 4, 0]), distance_matrix_5)


if __name__ == '__main__':
    unittest.main()
