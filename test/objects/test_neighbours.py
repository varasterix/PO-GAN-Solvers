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
neighbours_1 = Neighbours(np.array([1, 2, 3, 4, 0]), distance_matrix_5)  # solution d = 5+1+4+2+9 = 21
neighbours_2 = Neighbours(np.array([2, 1, 0, 4, 3]), distance_matrix_5)  # 2 cycles
neighbours_3 = Neighbours(np.array([4, 5, 2, 3, 0]), distance_matrix_5)
neighbours_4 = Neighbours(np.array([1, 0, 2, 3, -1]), distance_matrix_5)
neighbours_5 = Neighbours(np.array([0, "c"]), distance_matrix_5[:2, :2])
neighbours_6 = Neighbours(np.array([4, 0, 3, 1, 2]), distance_matrix_5)  # solution d = 1+1+4+1+0 = 7
neighbours_6b = Neighbours(np.array([4, 0, 3, 1, 2]), distance_matrix_5)  # solution d = 1+1+4+1+0 = 7
neighbours_6c = Neighbours(np.array([4, 0, 3, 1, 2]), np.ones((5, 5), dtype=int))  # solution d = 1+1+1+1+1 = 5


class TestNeighboursMethods(unittest.TestCase):
    def test_is_valid_structure(self):
        self.assertTrue(neighbours_1.is_valid_structure())
        self.assertTrue(neighbours_2.is_valid_structure())
        self.assertFalse(neighbours_3.is_valid_structure())
        self.assertFalse(neighbours_4.is_valid_structure())
        self.assertFalse(neighbours_5.is_valid_structure())
        self.assertTrue(neighbours_6.is_valid_structure())

    def test_is_solution(self):
        self.assertTrue(neighbours_1.is_solution())
        self.assertFalse(neighbours_2.is_solution())
        self.assertFalse(neighbours_3.is_solution())
        self.assertFalse(neighbours_4.is_solution())
        self.assertFalse(neighbours_5.is_solution())
        self.assertTrue(neighbours_6.is_solution())

    def test_distance(self):
        self.assertEqual(neighbours_1.distance(), 21)
        self.assertEqual(neighbours_6.distance(), 7)
        self.assertEqual(neighbours_6c.distance(), 5)
        with self.assertRaises(Exception):
            neighbours_2.distance()

    def test_to_ordered_path(self):
        self.assertEqual(neighbours_1.to_ordered_path(),
                         OrderedPath(np.array([2, 3, 4, 0, 1]), distance_matrix_5))
        self.assertEqual(neighbours_6.to_ordered_path(),
                         OrderedPath(np.array([3, 1, 0, 4, 2]), distance_matrix_5))
        with self.assertRaises(Exception):
            neighbours_2.to_ordered_path()

    def test_to_neighbours_binary_matrix(self):
        neighbours_binary_matrix_1 = np.array([[0, 0, 0, 0, 1],
                                               [1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0]], dtype=int)
        self.assertEqual(neighbours_1.to_neighbours_binary_matrix(),
                         NeighboursBinaryMatrix(neighbours_binary_matrix_1, distance_matrix_5))
        neighbours_binary_matrix_6 = np.array([[0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1],
                                               [0, 0, 1, 0, 0],
                                               [1, 0, 0, 0, 0]], dtype=int)
        self.assertEqual(neighbours_6.to_neighbours_binary_matrix(),
                         NeighboursBinaryMatrix(neighbours_binary_matrix_6, distance_matrix_5))
        with self.assertRaises(Exception):
            neighbours_2.to_neighbours_binary_matrix()

    def test_to_ordered_path_binary_matrix(self):
        ordered_path_binary_matrix_1 = np.array([[0, 0, 1, 0, 0],
                                                 [0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 1],
                                                 [1, 0, 0, 0, 0],
                                                 [0, 1, 0, 0, 0]], dtype=int)
        self.assertEqual(neighbours_1.to_ordered_path_binary_matrix(),
                         OrderedPathBinaryMatrix(ordered_path_binary_matrix_1, distance_matrix_5))
        ordered_path_binary_matrix_6 = np.array([[0, 1, 0, 0, 0],
                                                 [1, 0, 0, 0, 0],
                                                 [0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 1],
                                                 [0, 0, 1, 0, 0]], dtype=int)
        self.assertEqual(neighbours_6.to_ordered_path_binary_matrix(),
                         OrderedPathBinaryMatrix(ordered_path_binary_matrix_6, distance_matrix_5))
        with self.assertRaises(Exception):
            neighbours_2.to_ordered_path_binary_matrix()

    def test_eq(self):
        self.assertEqual(neighbours_6, neighbours_6b)
        self.assertNotEqual(neighbours_6, neighbours_6c)
        with self.assertRaises(Exception):
            neighbours_2.__eq__(neighbours_1)


if __name__ == '__main__':
    unittest.main()
