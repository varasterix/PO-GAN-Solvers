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

    def test_read_tsp_choco_solution_file(self):
        choco_data = dT.read_tsp_choco_solution_file(10, 1, "test/database/")
        choco_weight_matrix = np.array([[0, 413, 578, 209, 203, 476, 429, 410, 232, 462],
                                        [413, 0, 471, 318, 209, 394, 29, 449, 552, 361],
                                        [578, 471, 0, 369, 488, 103, 498, 198, 507, 126],
                                        [209, 318, 369, 0, 174, 267, 345, 218, 239, 252],
                                        [203, 209, 488, 174, 0, 388, 226, 379, 368, 362],
                                        [476, 394, 103, 267, 388, 0, 422, 127, 422, 33],
                                        [429, 29, 498, 345, 226, 422, 0, 479, 576, 389],
                                        [410, 449, 198, 218, 379, 127, 479, 0, 309, 144],
                                        [232, 552, 507, 239, 368, 422, 576, 309, 0, 425],
                                        [462, 361, 126, 252, 362, 33, 389, 144, 425, 0]], dtype=int)
        choco_candidate = np.array([8, 6, 5, 7, 0, 9, 4, 2, 3, 1], dtype=int)
        choco_total_weight = 1842
        choco_cartesian = np.array([[164, 189],
                                    [57, 588],
                                    [526, 640],
                                    [282, 362],
                                    [109, 385],
                                    [451, 569],
                                    [29, 597],
                                    [484, 446],
                                    [393, 150],
                                    [418, 575]], dtype=int)
        self.assertTrue(choco_data[0] == OrderedPath(choco_candidate, choco_weight_matrix, choco_cartesian))
        self.assertTrue(choco_data[1] == choco_total_weight)
        dT.compute_tsp_nnh_solution_from_choco_database(10, 1, "test/database/", "test/database/")
        dT.compute_tsp_nnh_two_opt_solution_from_choco_database(10, 1, "test/database/", "test/database/", 5)


if __name__ == '__main__':
    unittest.main()
