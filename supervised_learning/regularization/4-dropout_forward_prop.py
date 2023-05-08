#!/usr/bin/env python3
"""Script to implement dropout in a forward propagation"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    Args:
        X (ndarray): contains the input data for the network
        weights (dict): dictionary of the weights and biases of the neural
                        network
        L (int): number of layers in the network
        keep_prob (float): probability that a node will be kept
    Returns:
        dictionary containing the outputs of each layer and the dropout mask
        used on each layer
    """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        if i == 0:
            cache["A0"] = X
        else:
            Z = np.dot(weights["W" + str(i+1)], cache["A" + str(i-1)]) + (weights["b" + str(i)])
            cache["A" + str(i)] = np.tanh(Z) / keep_prob
    return cache
