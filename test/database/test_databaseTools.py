import unittest
import numpy as np
import src.database.databaseTools as dT


class TestDatabaseToolsMethods(unittest.TestCase):
    def test_read_tsp_file(self):
        expected_weight_matrix = np.array([[0, 4, 1, 1, 1],
                                           [0, 0, 5, 1, 0],
                                           [2, 1, 0, 3, 4],
                                           [2, 0, 3, 0, 4],
                                           [4, 5, 3, 4, 0]], dtype=int)
        self.assertTrue((dT.read_tsp_file(5, 0, "test/database/") == expected_weight_matrix).all())


if __name__ == '__main__':
    unittest.main()
