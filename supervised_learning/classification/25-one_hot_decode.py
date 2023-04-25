#!/usr/bin/env python3
"""Script one hot decode model
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a numeric label vector into a one-hot matrix
    Args:
        Y (ndarray): contains numeric class labels
        classes (int): maximum number of classes found in Y
    Returns:
        ndarray: one-hot encoding of Y or None on failure
    """
    if not isinstance(one_hot, np.ndarray) and len(one_hot) == 2:
        return np.argmax(one_hot, axis=0)
    return None
