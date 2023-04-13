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
        if not isinstance(nx, int):
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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y: Y hat, slope
            A: Activated neuron output

        Returns: Cost value, efficiency when cost = 0
        """

        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels for the input data

        Returns the neuron’s prediction and the cost of the network
        """
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        
        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels vector
            A: Activated neuron output
            alpha: learning rate
        Returns: gradient descent bias + adjusted weights
        """

        m = Y.shape[1]
        dz = A - Y  # derivative z
        dw = np.matmul(X, dz.T) / m  # grad of the loss with respect to w
        db = np.sum(dz) / m  # grad of the loss with respect to b

        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - (alpha * db)
