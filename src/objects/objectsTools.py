import numpy as np


def is_weight_matrix_valid_structure(weight_matrix):
    """
    The structure of the weight matrix is valid if each element of the matrix (size n x n) is an integer
    :param weight_matrix: object pretending to be a weight/distance matrix
    :return: True if the structure of the weight matrix is valid, False otherwise
    """
    return (type(weight_matrix) == np.ndarray and weight_matrix.dtype == int and
            weight_matrix.shape[1] == weight_matrix.shape[0])


def is_weight_matrix_symmetric(weight_matrix):
    """
    The structure of the weight matrix is valid if each element of the matrix (size n x n) is an integer
    The weight matrix has also to be symmetric
    :param weight_matrix: object pretending to be a weight/distance matrix symmetric
    :return: True if the structure of the weight matrix is valid and symmetric, False otherwise
    """
    if not is_weight_matrix_valid_structure(weight_matrix):
        return False
    else:
        is_symmetric = True
        n = len(weight_matrix)
        for i in range(n):
            for j in range(i):
                if i != j:
                    is_symmetric = is_symmetric and (weight_matrix[i, j] == weight_matrix[j, i])
                if not is_symmetric:
                    break
    return is_symmetric


def get_highest_weight(weight_matrix):
    """
    Returns the highest weight of the weight/distance matrix
    :param weight_matrix: object pretending to be a weight/distance matrix symmetric
    :return: the highest weight of the weight/distance matrix if it has a valid structure, an Exception otherwise
    """
    if not is_weight_matrix_valid_structure(weight_matrix):
        raise Exception("The structure of the weight/distance matrix is not valid")
    else:
        return np.max(weight_matrix)


def normalize_weight_matrix(weight_matrix):
    """
    Returns the corresponding normalized weight/distance matrix
    :param weight_matrix: object pretending to be a weight/distance matrix
    :return: the corresponding normalized weight/distance matrix if the weight/distance has a valid structure,
    an Exception otherwise
    """
    highest_weight = get_highest_weight(weight_matrix)
    return weight_matrix / highest_weight
