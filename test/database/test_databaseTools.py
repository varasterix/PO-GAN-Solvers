import unittest
import numpy as np
import src.database.databaseTools as dT
from src.objects.orderedPath import OrderedPath

expected_weight_matrix = np.array([[0, 4, 1, 1, 1],
                                   [0, 0, 5, 1, 0],
                                   [2, 1, 0, 3, 4],
                                   [2, 0, 3, 0, 4],
                                   [4, 5, 3, 4, 0]], dtype=int)
expected_weight_matrix_sym = np.array([[0, 4, 1, 1, 1],
                                       [4, 0, 5, 1, 0],
                                       [1, 5, 0, 3, 4],
                                       [1, 1, 3, 0, 4],
                                       [1, 0, 4, 4, 0]], dtype=int)


class TestDatabaseToolsMethods(unittest.TestCase):
    def test_read_tsp_file(self):
        self.assertTrue((dT.read_tsp_file(5, 0, "test/database/") == expected_weight_matrix).all())

    def test_read_tsp_heuristic_solution_file(self):
        heuristic_data = dT.read_tsp_heuristic_solution_file(5, 0, "test/database/")
        self.assertTrue(heuristic_data[0] ==
                        OrderedPath(np.array([0, 4, 2, 1, 3], dtype=int), expected_weight_matrix_sym))
        self.assertTrue(heuristic_data[1] == 12)


if __name__ == '__main__':
    unittest.main()
