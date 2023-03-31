#!/usr/bin/env python3
"""
module poly_derivative
"""


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial
    Args:
        poly (list): list of coefficients representing a polynomial
    Returns:
        list: new list of coefficients representing the derivative of the
        polynomial
    """
    if type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    if len(poly) == 0:
        return None
    derivative = []
    for power in range(1, len(poly)):
        if type(poly[power]) is not int:
            return None
        derivative.append(poly[power]*power)
    return derivative
