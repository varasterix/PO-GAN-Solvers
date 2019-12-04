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
# solution d = 5+1+4+2+9 = 21
neighbours_binary_matrix_1 = NeighboursBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                              [1, 0, 0, 0, 0],
                                                              [0, 1, 0, 0, 0],
                                                              [0, 0, 1, 0, 0],
                                                              [0, 0, 0, 1, 0]], dtype=int), distance_matrix_5)
neighbours_binary_matrix_2 = NeighboursBinaryMatrix(np.array([[0, 0, 1, 0, 0],
                                                              [0, 1, 0, 0, 0],
                                                              [1, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 1],
                                                              [0, 0, 0, 1, 0]], dtype=int),
                                                    distance_matrix_5)  # 2 cycles
neighbours_binary_matrix_3 = NeighboursBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                              [1, 0, 0, 0, 1],
                                                              [0, 1, 0, 0, 0],
                                                              [0, 0, 1, 0, 0],
                                                              [0, 1, 0, 1, 0]], dtype=int), distance_matrix_5)
neighbours_binary_matrix_4 = NeighboursBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                              [1, 0, 0, 0, 0],
                                                              [0, 0, 1, 0, 0]], dtype=int), distance_matrix_5)
neighbours_binary_matrix_5 = NeighboursBinaryMatrix(np.array([[0, 0, 0, 0, 1],
                                                              ["c", 0, 0, 0, 0],
                                                              [0, 1, 0, 0, 0],
                                                              [0, 0, -1, 0, 0],
                                                              [0, 0, 0, 1, 0]]), distance_matrix_5[:2, :2])
# solution d = 1+1+4+1+0 = 7
neighbours_binary_matrix_6 = NeighboursBinaryMatrix(np.array([[0, 1, 0, 0, 0],
                                                              [0, 0, 0, 1, 0],
                                                              [0, 0, 0, 0, 1],
                                                              [0, 0, 1, 0, 0],
                                                              [1, 0, 0, 0, 0]]), distance_matrix_5)
# solution d = 1+1+4+1+0 = 7
neighbours_binary_matrix_6b = NeighboursBinaryMatrix(np.array([[0, 1, 0, 0, 0],
                                                               [0, 0, 0, 1, 0],
                                                               [0, 0, 0, 0, 1],
                                                               [0, 0, 1, 0, 0],
                                                               [1, 0, 0, 0, 0]]), distance_matrix_5)
# solution d = 1+1+1+1+1 = 5
neighbours_binary_matrix_6c = NeighboursBinaryMatrix(np.array([[0, 1, 0, 0, 0],
                                                               [0, 0, 0, 1, 0],
                                                               [0, 0, 0, 0, 1],
                                                               [0, 0, 1, 0, 0],
                                                               [1, 0, 0, 0, 0]]), np.ones((5, 5), dtype=int))


class TestNeighboursBinaryMatrixMethods(unittest.TestCase):
    def test_is_valid_structure(self):
        self.assertTrue(neighbours_binary_matrix_1.is_valid_structure())
        self.assertTrue(neighbours_binary_matrix_2.is_valid_structure())
        self.assertFalse(neighbours_binary_matrix_3.is_valid_structure())
        self.assertFalse(neighbours_binary_matrix_4.is_valid_structure())
        self.assertFalse(neighbours_binary_matrix_5.is_valid_structure())
        self.assertTrue(neighbours_binary_matrix_6.is_valid_structure())

    def test_is_solution(self):
        self.assertTrue(neighbours_binary_matrix_1.is_solution())
        self.assertFalse(neighbours_binary_matrix_2.is_solution())
        self.assertFalse(neighbours_binary_matrix_3.is_solution())
        self.assertFalse(neighbours_binary_matrix_4.is_solution())
        self.assertFalse(neighbours_binary_matrix_5.is_solution())
        self.assertTrue(neighbours_binary_matrix_6.is_solution())

    def test_distance(self):
        self.assertEqual(neighbours_binary_matrix_1.distance(), 21)
        self.assertEqual(neighbours_binary_matrix_6.distance(), 7)
        self.assertEqual(neighbours_binary_matrix_6c.distance(), 5)
        with self.assertRaises(Exception):
            neighbours_binary_matrix_3.distance()

    def test_to_ordered_path(self):
        ordered_path_1 = OrderedPath(np.array([0, 1, 2, 3, 4]), distance_matrix_5)
        ordered_path_1b = OrderedPath(np.array([2, 3, 4, 0, 1]), distance_matrix_5)
        ordered_path_6 = OrderedPath(np.array([2, 3, 1, 0, 4]), distance_matrix_5)
        self.assertEqual(neighbours_binary_matrix_1.to_ordered_path(), ordered_path_1)
        self.assertEqual(neighbours_binary_matrix_1.to_ordered_path(), ordered_path_1b)
        self.assertEqual(neighbours_binary_matrix_6.to_ordered_path(), ordered_path_6)
        with self.assertRaises(Exception):
            neighbours_binary_matrix_3.to_ordered_path()

    def test_to_neighbours(self):
        neighbours_1 = Neighbours(np.array([1, 2, 3, 4, 0]), distance_matrix_5)
        neighbours_6 = Neighbours(np.array([4, 0, 3, 1, 2]), distance_matrix_5)
        self.assertEqual(neighbours_binary_matrix_1.to_neighbours(), neighbours_1)
        self.assertEqual(neighbours_binary_matrix_6.to_neighbours(), neighbours_6)
        with self.assertRaises(Exception):
            neighbours_binary_matrix_3.to_neighbours()

    def test_to_ordered_path_binary_matrix(self):
        ordered_path_binary_matrix_1 = OrderedPathBinaryMatrix(np.array([[1, 0, 0, 0, 0],
                                                                         [0, 1, 0, 0, 0],
                                                                         [0, 0, 1, 0, 0],
                                                                         [0, 0, 0, 1, 0],
                                                                         [0, 0, 0, 0, 1]]), distance_matrix_5)
        ordered_path_binary_matrix_1b = OrderedPathBinaryMatrix(np.array([[0, 0, 1, 0, 0],
                                                                          [0, 0, 0, 1, 0],
                                                                          [0, 0, 0, 0, 1],
                                                                          [1, 0, 0, 0, 0],
                                                                          [0, 1, 0, 0, 0]]), distance_matrix_5)
        ordered_path_binary_matrix_6 = OrderedPathBinaryMatrix(np.array([[0, 0, 0, 1, 0],
                                                                         [0, 0, 1, 0, 0],
                                                                         [1, 0, 0, 0, 0],
                                                                         [0, 1, 0, 0, 0],
                                                                         [0, 0, 0, 0, 1]]), distance_matrix_5)
        self.assertEqual(neighbours_binary_matrix_1.to_ordered_path_binary_matrix(), ordered_path_binary_matrix_1)
        self.assertEqual(neighbours_binary_matrix_1.to_ordered_path_binary_matrix(), ordered_path_binary_matrix_1b)
        self.assertEqual(neighbours_binary_matrix_6.to_ordered_path_binary_matrix(), ordered_path_binary_matrix_6)
        with self.assertRaises(Exception):
            neighbours_binary_matrix_3.to_ordered_path_binary_matrix()

    def test_eq(self):
        self.assertEqual(neighbours_binary_matrix_6, neighbours_binary_matrix_6b)
        self.assertNotEqual(neighbours_binary_matrix_6, neighbours_binary_matrix_6c)
        with self.assertRaises(Exception):
            neighbours_binary_matrix_1.__eq__(neighbours_binary_matrix_2)


if __name__ == '__main__':
    unittest.main()
