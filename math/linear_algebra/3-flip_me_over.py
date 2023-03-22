#!/usr/bin/env python3
'''
Module for matrix_transpose
'''


def matrix_transpose(matrix):
    '''
    returns the transpose of a 2D matrix
    '''
    zipped_rows = zip(*matrix)
    transpose_matrix = [list(row) for row in zipped_rows]
    return transpose_matrix
