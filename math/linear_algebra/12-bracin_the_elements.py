#!/usr/bin/env python3
"""
Module Calculator of numpy.ndarray
"""


def np_elementwise(mat1, mat2):
    """
    Calculates addition,substraction, multiplication 
    and division of numpy.ndarray
    Parameters:
        matrix (numpy.ndarray): The numpy ndarray

    Returns:
        tuple: containing the element-wise sum, 
        difference, product, and quotient, respectively
    """
    sum_result = (mat1 + mat2)
    diff_result = (mat1 - mat2)
    prod_result = (mat1 * mat2)
    quotient_result = (mat1 / mat2)
    return (sum_result, diff_result, prod_result, quotient_result)
