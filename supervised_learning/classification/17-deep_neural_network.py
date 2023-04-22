#!/usr/bin/env python3
"""Script to create an DNN with private attributes"""

import numpy as np


class DeepNeuralNetwork():
    """
    Class Deep Neural Network
    """

    def __init__(self, nx, layers):
        """
        Args:
            nx: input value
            nodes: nodes placed in the hidden layer
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.L):
            if layers[lay] < 1 or type(layers[lay]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))
            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(lay + 1)] = He_val
            if lay > 0:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.__weights["W" + str(lay + 1)] = He_val3

    @property
    def L(self):
        """
        Getter attr
        Returns: Private instance
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter attr
        Returns: Private instance
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter attr
        Returns: Private instance
        """
        return self.__weights
