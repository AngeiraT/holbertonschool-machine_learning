#!/usr/bin/env python3
"""
module summation_i_squared
"""


def summation_i_squared(n):
    """
    calculates summation
    Args:
        n (int): stopping condition
    Returns:
        int: sum
    """
    if n > 0:
        return int(n*(n+1)*((2*n)+1)/6)
    return None
