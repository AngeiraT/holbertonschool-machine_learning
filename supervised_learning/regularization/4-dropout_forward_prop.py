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
    cache['A0'] = X
    for i in range(L):
        w = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        a = cache['A' + str(i)]
        z = np.matmul(w, a) + b
        if i == L - 1:
            t = np.exp(z)
            cache['A' + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            a = np.tanh(z)
            d = np.random.rand(a.shape[0], a.shape[1])
            d = np.where(d < keep_prob, 1, 0)
            cache['D' + str(i + 1)] = d
            cache['A' + str(i + 1)] = a * d / keep_prob
    return cache
