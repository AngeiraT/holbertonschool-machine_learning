#!/usr/bin/env python3
"""Script to train model and plotting
    DNN
"""

import numpy as np
import matplotlib.pyplot as plt


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

        self.__L = len(layers)
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

    def forward_prop(self, X):
        """
        Forward propagation function
        Args:
            X: x numpy array with shape (nx, m)
        Returns: forward propagation
        """
        self.__cache["A0"] = X
        for lay in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            Za = np.matmul(weights["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = Za + weights["b" + str(lay + 1)]
            cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))

        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        """
        Cost function using binary cross-entropy
        Args:
            Y: Y hat, slope
            A: Activated neuron output
        Returns: Cost value, efficiency when C = 0
        """

        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels for the input data
        Returns: The neuron prediction and the cost
                of the network
        """

        cache = self.__cache
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)  # broadcasting
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels vector
            cache: Activated neurons in n layer
            alpha: learning rate
        Returns: gradient descent bias + adjusted weights
        """

        m = Y.shape[1]
        weights = self.__weights.copy()
        A2 = self.__cache["A" + str(self.__L - 1)]
        A3 = self.__cache["A" + str(self.__L)]
        W3 = weights["W" + str(self.__L)]
        b3 = weights["b" + str(self.__L)]
        dZ_input = {}
        dZ3 = A3 - Y  # derivative Z3
        dZ_input["dz" + str(self.__L)] = dZ3

        # grad of the loss with respect to w
        dW3 = (1 / m) * np.matmul(A2, dZ3.T)

        # grad of the loss with respect to b
        db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

        self.__weights["W" + str(self.__L)] = W3 - (alpha * dW3).T
        self.__weights["b" + str(self.__L)] = b3 - (alpha * db3)

        for lay in range(self.__L - 1, 0, -1):
            cache = self.__cache
            Aa = cache["A" + str(lay)]
            Ap = cache["A" + str(lay - 1)]
            Wa = weights["W" + str(lay)]
            Wn = weights["W" + str(lay + 1)]
            ba = weights["b" + str(lay)]
            dZ1 = np.matmul(Wn.T, dZ_input["dz" + str(lay + 1)])
            dZ2 = Aa * (1 - Aa)
            dZ = dZ1 * dZ2
            dW = (1 / m) * np.matmul(Ap, dZ.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ_input["dz" + str(lay)] = dZ
            self.__weights["W" + str(lay)] = Wa - (alpha * dW).T
            self.__weights["b" + str(lay)] = ba - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Args:
            X (ndarray): contains the input data
            Y (ndarray): contains the correct labels for the input data
            iterations (int, optional): number of iterations to train over.
                                        Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.
        Raises:
            TypeError: iterations must be an integer
            ValueError: iterations must be a positive
            TypeError: alpha must be a float
            ValueError: alpha must be positive
        Returns:
            float: evaluation of the training data after iterations of
                   training have occured
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
            cost = self.cost(Y, self.__cache["A" + str(self.__L)])
            if i % step == 0 or i == iterations:
                temp_cost.append(cost)
                temp_iterations.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        saves the instance object to a file in pickle format
        Args:
            filename (str): file to which the object should be saved
        """
        import pickle
        if type(filename) != str:
            return
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
            file.close()

    @staticmethod
    def load(filename):
        """
        loads a pickled object
        Args:
            filename (str): file from which the object should be loaded
        Returns:
            obj: loaded object or None if it doesn't exist
        """
        import pickle
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
