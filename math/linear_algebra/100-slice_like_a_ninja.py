#!/usr/bin/env python3
"""
Module Slices a matrix of numpy.ndarray
"""


import numpy as np


def np_slice(matrix, axes={}):
    """
    Slices a matrix along a specific axes: of numpy.ndarray
    Parameters:
        matrix (numpy.ndarray): The numpy ndarray

    Returns:
        matrix (numpy.ndarray): A new matrix
    """
    slices = [slice(None, None, None)] * matrix.ndim
    for axis, axis_slice in axes.items():
        slices[axis] = slice(*axis_slice)
    return matrix[tuple(slices)]
