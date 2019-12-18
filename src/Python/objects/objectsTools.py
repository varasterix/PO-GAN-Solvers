import numpy as np


def is_weight_matrix_valid_structure(weight_matrix):
    """
    The structure of the weight matrix is valid if each element of the matrix (size n x n) is an integer
    :return: True if the structure of the weight matrix is valid, False otherwise
    """
    return (type(weight_matrix) == np.ndarray and weight_matrix.dtype == int and
            weight_matrix.shape[1] == weight_matrix.shape[0])
