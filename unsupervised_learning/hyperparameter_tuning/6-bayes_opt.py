#!/usr/bin/env python3
"""Bayesian Optimization with GPyOpt"""
import numpy as np
import GPy
import GPyOpt
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Define the function to optimize (negative accuracy to turn maximization into minimization)
def model_accuracy_hyperopt(hyperparameters):
    """
        save a report of the optimization to the file 'bayes_opt.txt'
    """
    learning_rate = float(hyperparameters[:, 0])
    num_units = int(hyperparameters[:, 1])
    dropout_rate = float(hyperparameters[:, 2])
    l2_reg_weight = float(hyperparameters[:, 3])
    batch_size = int(hyperparameters[:, 4])

    model = Sequential([
        Dense(num_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg_weight),
              input_shape=(784,)),
        Dropout(dropout_rate),
        Dense(num_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg_weight)),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_split=0.1, callbacks=[early_stopping])

    # Evaluate the model on the test set
    y_pred = model.predict_classes(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return -accuracy

# Define the bounds of the hyperparameters to be optimized
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (16, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_reg_weight', 'type': 'continuous', 'domain': (1e-6, 1e-3)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

# Initialize the Bayesian optimizer
optimizer = GPyOpt.methods.BayesianOptimization(f=model_accuracy_hyperopt, domain=bounds, acquisition_type='EI')

# Run optimization for a maximum of 30 iterations
max_iter = 30
optimizer.run_optimization(max_iter)

# Save a report of the optimization
report_file = 'bayes_opt.txt'
with open(report_file, 'w') as f:
    f.write(f'Optimal hyperparameters: {optimizer.x_opt}\n')
    f.write(f'Optimal accuracy: {-optimizer.fx_opt}\n')

# Plot the convergence
plt.figure(figsize=(10, 6))
plt.plot(np.arange(max_iter), -optimizer.Y, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Negative Accuracy')
plt.title('Bayesian Optimization Convergence')
plt.savefig('convergence_plot.png')
plt.show()
