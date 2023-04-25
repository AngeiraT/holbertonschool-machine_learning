#!/usr/bin/env python3
"""Script one hot encode model
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Args:
        Y (ndarray): contains numeric class labels
        classes (int): maximum number of classes found in Y
    Returns:
        ndarray: one-hot encoding of Y or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) == 1:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    m = Y.shape[0]
    Y_one_hot = np.zeros((classes, m))
    Y_one_hot[Y, np.arange(m)] = 1
    return Y_one_hot
