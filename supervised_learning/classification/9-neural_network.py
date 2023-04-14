#!/usr/bin/env python3
"""Script to create an FNN with private attributes """

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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # Output Neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0
    
    @property
    def W1(self):
        """
        Getter attr
        Args:
            self: Private attribute
            Returns: Weight vector 1 hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter attr
        Args:
            self: Private attribute
            Returns: Bias1
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter attr
        Args:
            self: Private attribute
            Returns: Activated1
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter attr
        Args:
            self: Private attribute
            Returns: Weight vector 2
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter attr
        Args:
            self: Private attribute
            Returns: Bias2
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter attr
        Args:
            self: Private attribute
            Returns: Activated output 2 prediction
        """
        return self.__A2
