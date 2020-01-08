import unittest
import numpy as np
from src.objects.neighbours import Neighbours
from src.objects.neighboursBinaryMatrix import NeighboursBinaryMatrix
from src.objects.orderedPath import OrderedPath
from src.objects.orderedPathBinaryMatrix import OrderedPathBinaryMatrix

distance_matrix_5 = np.array([[0, 5, 1, 10, 1],
                              [1, 0, 1, 8, 1],
                              [1, 15, 0, 4, 6],
                              [8, 1, 2, 0, 2],
                              [9, 7, 0, 1, 0]], dtype=int)
ordered_path_binary_matrix_1 = \
    OrderedPathBinaryMatrix(np.array([[1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1]], dtype=int), distance_matrix_5)  # solution d = 5+1+4+2+9 = 21
ordered_path_binary_matrix_1b = \
    OrderedPathBinaryMatrix(np.array([[1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1]]), distance_matrix_5)  # solution d = 5+1+4+2+9 = 21
ordered_path_binary_matrix_1c = \
    OrderedPathBinaryMatrix(np.array([[0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0]]), distance_matrix_5)  # solution d = 5+1+4+2+9 = 21
ordered_path_binary_matrix_2 = OrderedPathBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                                [1, 0, 1, 0, 0],
                                                                [0, 1, 0, 0, 0],
                                                                [0, 0, 0, 1, 0],
                                                                [0, 0, 0, 0, 0]], dtype=int), distance_matrix_5)
ordered_path_binary_matrix_3 = OrderedPathBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                                [1, 0, 0, 0, 1],
                                                                [0, 1, 0, 0, 0],
                                                                [0, 0, 1, 0, 0],
                                                                [0, 1, 0, 1, 0]], dtype=int), distance_matrix_5)
ordered_path_binary_matrix_4 = OrderedPathBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                                [1, 0, 0, 0, 0],
                                                                [0, 0, 1, 0, 0]], dtype=int), distance_matrix_5)
ordered_path_binary_matrix_5 = OrderedPathBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                                ["c", 0, 0, 0, 0],
                                                                [0, 1, 0, 0, 0],
                                                                [0, 0, -1, 0, 0],
                                                                [0, 0, 0, 1, 0]]), distance_matrix_5[:2, :2])
ordered_path_binary_matrix_8 = \
    OrderedPathBinaryMatrix(np.array([[0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 1, 0],
                                      [1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0]], dtype=int), distance_matrix_5)  # solution d = 8+1+0+15+8 = 32


class TestOrderedPathMethods(unittest.TestCase):
    def test_is_valid_structure(self):
        self.assertTrue(ordered_path_binary_matrix_1.is_valid_structure())
        self.assertTrue(ordered_path_binary_matrix_2.is_valid_structure())
        self.assertFalse(ordered_path_binary_matrix_3.is_valid_structure())
        self.assertFalse(ordered_path_binary_matrix_4.is_valid_structure())
        self.assertFalse(ordered_path_binary_matrix_5.is_valid_structure())
        self.assertTrue(ordered_path_binary_matrix_8.is_valid_structure())

    def test_is_solution(self):
        self.assertTrue(ordered_path_binary_matrix_1.is_solution())
        self.assertFalse(ordered_path_binary_matrix_2.is_solution())
        self.assertFalse(ordered_path_binary_matrix_3.is_solution())
        self.assertFalse(ordered_path_binary_matrix_4.is_solution())
        self.assertFalse(ordered_path_binary_matrix_5.is_solution())
        self.assertTrue(ordered_path_binary_matrix_8.is_solution())

    def test_distance(self):
        self.assertEqual(ordered_path_binary_matrix_1.distance(), 21)
        self.assertEqual(ordered_path_binary_matrix_8.distance(), 32)
        with self.assertRaises(Exception):
            ordered_path_binary_matrix_4.distance()

    def test_to_neighbours(self):
        self.assertEqual(ordered_path_binary_matrix_1.to_neighbours(),
                         Neighbours(np.array([1, 2, 3, 4, 0]), distance_matrix_5))
        self.assertEqual(ordered_path_binary_matrix_8.to_neighbours(),
                         Neighbours(np.array([4, 3, 1, 0, 2]), distance_matrix_5))
        with self.assertRaises(Exception):
            ordered_path_binary_matrix_4.to_neighbours()

    def test_to_neighbours_binary_matrix(self):
        neighbours_binary_matrix_1 = np.array([[0, 0, 0, 0, 1],
                                               [1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0]], dtype=int)
        self.assertEqual(ordered_path_binary_matrix_1.to_neighbours_binary_matrix(),
                         NeighboursBinaryMatrix(neighbours_binary_matrix_1, distance_matrix_5))
        neighbours_binary_matrix_8 = np.array([[0, 0, 0, 1, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1],
                                               [0, 1, 0, 0, 0],
                                               [1, 0, 0, 0, 0]], dtype=int)
        self.assertEqual(ordered_path_binary_matrix_8.to_neighbours_binary_matrix(),
                         NeighboursBinaryMatrix(neighbours_binary_matrix_8, distance_matrix_5))
        with self.assertRaises(Exception):
            ordered_path_binary_matrix_4.to_neighbours_binary_matrix()

    def test_to_ordered_path(self):
        ordered_path_1 = OrderedPath(np.array([0, 1, 2, 3, 4]), distance_matrix_5)
        self.assertEqual(ordered_path_binary_matrix_1.to_ordered_path(), ordered_path_1)
        ordered_path_8 = OrderedPath(np.array([3, 0, 4, 2, 1]), distance_matrix_5)
        self.assertEqual(ordered_path_binary_matrix_8.to_ordered_path(), ordered_path_8)
        with self.assertRaises(Exception):
            ordered_path_binary_matrix_4.to_ordered_path()

    def test_eq(self):
        self.assertFalse(ordered_path_binary_matrix_1 == ordered_path_binary_matrix_8)
        self.assertTrue(ordered_path_binary_matrix_1 == ordered_path_binary_matrix_1b)
        self.assertTrue(ordered_path_binary_matrix_1 == ordered_path_binary_matrix_1c)


if __name__ == '__main__':
    unittest.main()
