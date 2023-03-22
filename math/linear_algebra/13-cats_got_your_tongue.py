#!/usr/bin/env python3
"""
Module Concatenates two matrices of numpy.ndarray
"""


import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis: of numpy.ndarray
    Parameters:
        matrix (numpy.ndarray): The numpy ndarray

    Returns:
        matrix (numpy.ndarray): A new matrix
    """
    newMatrix = np.concatenate((mat1, mat2), axis=axis)
    return newMatrix
