#!/usr/bin/env python3
"""
Script to create A Neuron with private instance
"""

import numpy as np


class Neuron():
    """Class Neuron"""

    def __init__(self, nx):
        """
        Args:
            nx: Type int the number of n inputs features into the ANN
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)  # Weight
        self.__b = 0  # Bias
        self.__A = 0  # Output

    @property
    def W(self):
        """
        Returns: private instance weight
        """
        return self.__W

    @property
    def b(self):
        """
        Returns: private instance bias
        """
        return self.__b

    @property
    def A(self):
        """
        Returns: private instance output
        """
        return self.__A
    
    def forward_prop(self, X):
        """ Method that calculates the forward propagation of the neuron

            X: numpy.ndarray
               input data of shape (nx, m)
               nx: number of input features to the neuron
               m: number of examples

            Returns: private attribute __A
        """
        # Compute the weighted sum of inputs
        z = np.dot(self.__W, X) + self.__b

        # Compute the activated output using sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))  # (σ): g(z) = 1 / (1 + e^{-z})

        return self.__A
