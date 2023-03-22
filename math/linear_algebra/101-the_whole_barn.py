#!/usr/bin/env python3
"""
Module Add Matrix
"""


import numpy as np

def add_matrices(mat1, mat2):
    """
    Addition of two matrices: of numpy.ndarray
    Parameters:
        two matrices (numpy.ndarray): The list of lists

    Returns:
        matrix (numpy.ndarray): The sum of a new matrix as list
    """
    if np.shape(mat2) != np.shape(mat1):
        return None

    sum = np.add(mat1, mat2)
    return sum

    
