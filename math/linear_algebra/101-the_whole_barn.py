#!/usr/bin/env python3
"""
Module Add Matrix
"""


def shape(matrix):
    """
    Module Shape of a Matrix
    """
    shape = []
    while(type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def add_matrices(mat1, mat2):
    """
    Addition of two matrices: of numpy.ndarray
    Parameters:
        two matrices (numpy.ndarray): The list of lists

    Returns:
        matrix (numpy.ndarray): The sum of a new matrix as list
    """
    try:
        if shape(mat2) != shape(mat1):
            return None

        result = []
        for x, y in zip(mat1, mat2):
            temp = add_matrices(x, y)
            if temp is None:
                return None
            result.append(temp)
        return result
    except TypeError:
            return mat1 + mat2
