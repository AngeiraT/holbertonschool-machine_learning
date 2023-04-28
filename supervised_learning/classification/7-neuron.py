#!/usr/bin/env python3
"""
Script to train a neuron including graphic
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron
         Args:
            X (ndarray): contains input data
            Y (ndarray): contains the correct labels for the input data
            iterations (int, optional): number of iterations to train over.
                                        Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.
            verbose (bool, optional): defines whether or not to print
                                      information about the training.
                                      Defaults to True.
            graph (bool, optional): defines whether or not to graph
                                    information about the training once it has
                                    completed. Defaults to True.
            step (int, optional): steps of the graph. Defaults to 100.
        Raises:
            TypeError: iterations must be an integer
            ValueError: iterations must be a positive integer
            TypeError: alpha must be a float
            ValueError: alpha must be positive
            TypeError: step must be an integer
            ValueError: step must be positive and <= iterations
        Returns:
            ndarray: evaluation of the training data after iterations of
                     training have occurred
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        steps = 0
        c_ax = np.zeros(iterations + 1)

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)
            if i % step == 0 or i == iterations:
                temp_cost.append(cost)
                temp_iterations.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)


        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)
