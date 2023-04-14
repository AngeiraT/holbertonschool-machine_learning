#!/usr/bin/env python3
"""Script to create an FNN """

import numpy as np


class NeuralNetwork():
    """ Class neural network"""

    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases
        # Hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        # Output Neuron
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
