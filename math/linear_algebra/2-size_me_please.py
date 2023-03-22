#!/usr/bin/env python3
"""
Matrix shape function
"""

def matrix_shape(matrix):
    """
    Matrix size gives the rows ,columns 
    """
    shape = []
    while(type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
