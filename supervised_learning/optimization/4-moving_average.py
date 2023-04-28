#!/usr/bin/env python3
"""Script to calculate the weighted
    moving average of a data set
"""


def moving_average(data, beta):
    """
    Function to calulate Exponential Moving Average
    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    Returns: list containing the moving averages of data
    """
    avg = []
    n = 0
    for i in range(len(data)):
        n = beta * n + (1 - beta) * data[i]
        avg.append(n / (1 - beta ** (i + 1)))
    return avg
