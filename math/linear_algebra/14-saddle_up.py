#!/usr/bin/env python3
"""
Module Multiply two matrices of numpy.ndarray
"""


import numpy as np


def np_matmul(mat1, mat2):
    """
    Calculates multiplication of two matrices: of numpy.ndarray
    Parameters:
        matrix (numpy.ndarray): The numpy ndarray

    Returns:
        matrix (numpy.ndarray): A new matrix
    """
    prod_result = np.dot(mat1, mat2)
    return prod_result
