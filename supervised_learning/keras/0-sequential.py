#!/usr/bin/env python3
"""Script for build a DNN in keras"""

import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function to create a DNN using keras
    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
                of the network
        activations: list containing the activation functions used
                     for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns: Keras model

    """
    
    model = keras.Sequential()

    # Adding the input layer
    model.add(keras.layers.Dense(layers[0], activation=activations[0], input_shape=(nx,),
                                 kernel_regularizer=keras.regularizers.l2(lambtha)))

    # Adding the hidden layers
    for i in range(1, len(layers)):
        model.add(keras.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=keras.regularizers.l2(lambtha)))
        model.add(keras.layers.Dropout(1 - keep_prob))

    return model
