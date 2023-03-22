#!/usr/bin/env python3
"""
Module Multiplies Matrices
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices
    """
    if len(mat1[0]) != len(mat2):
        return None

    newMatrix = [[0 for j in range(len(mat2[0]))] for i in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                newMatrix[i][j] += mat1[i][k] * mat2[k][j]

    return newMatrix
