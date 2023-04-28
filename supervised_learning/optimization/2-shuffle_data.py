#!/usr/bin/env python3
"""
Script to shuff;e data in a matrix
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    Args:
        X (ndarray): first matrix to shuffle
        Y (ndarray): second matrix to shuffle
    Returns: shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
