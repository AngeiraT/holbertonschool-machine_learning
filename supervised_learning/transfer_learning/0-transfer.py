#!/usr/bin/env python3
"""Transfer learning CIFAR-10 in densenet 121"""

import tensorflow as tf
import tensorflow.keras as K
import datetime
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



def preprocess_data(X, Y):
    """
    Function that pre-processes the data for your model
    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
        where m is the number of data points
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    """
    X = X.astype('float32')
    X /= 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y

if __name__ == '__main__':
    # Load CIFAR 10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Load MobileNetV2 model
    base_model = K.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = Lambda(lambda image: K.backend.resize_images(image, (224, 224)))(base_model.output)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

    # Save the model
    model.save('cifar10.h5')
