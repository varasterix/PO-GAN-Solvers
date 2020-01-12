import unittest
import numpy as np
from src.objects.objectsTools import is_weight_matrix_symmetric, get_highest_weight

weight_matrix_1 = np.array([[0, 5, 1, 10, 1],
                            [1, 0, 1, 8, 1],
                            [1, 15, 0, 4, 6],
                            [8, 1, 2, 0, 2],
                            [9, 7, 0, 1, 0]], dtype=int)
weight_matrix_2 = np.array([[0, 4, 3, 1, 2],
                            [4, 0, 7, 3, 2],
                            [3, 7, 0, 4, 5],
                            [1, 3, 4, 0, 1],
                            [2, 2, 5, 1, 0]], dtype=int)


class TestObjectToolsMethods(unittest.TestCase):
    def test_is_weight_matrix_symmetric(self):
        self.assertTrue(is_weight_matrix_symmetric(weight_matrix_2))
        self.assertFalse(is_weight_matrix_symmetric(weight_matrix_1))

    def test_get_highest_weight(self):
        self.assertEqual(get_highest_weight(weight_matrix_1), 15)
        self.assertEqual(get_highest_weight(weight_matrix_2), 7)


if __name__ == '__main__':
    unittest.main()
