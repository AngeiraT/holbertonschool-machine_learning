#!/usr/bin/env python3
"""Script one hot encode model
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
    # Get the number of examples
    m = one_hot.shape[1]

    # Convert the one-hot matrix to a vector of labels
    decoded = np.argmax(one_hot, axis=0)

    # Check if the number of decoded labels is equal to the number of examples
    if len(decoded) != m:
        return None

    return decoded