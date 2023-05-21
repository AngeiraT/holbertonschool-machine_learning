#!/usr/bin/env python3
""" Script to forward propagate over a pooling layer in a NN"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function to forward propagate over a pooling layer in a NN
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
                m: is the number of examples
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
                c_prev: the number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
                      the pooling
                      kh: the kernel height
                      kw: the kernel width
        stride: tuple of (sh, sw) containing the strides for the convolution
                sh: the stride for the height
                sw: the stride for the width
        mode: string containing either max or avg, indicating whether to
              perform maximum or average pooling, respectively
    Returns: output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
 
    # Compute the dimensions of the output
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Initialize the output array
    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            # Extract the current window from the input
            window = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
           
            if mode == 'max':
                # Apply max pooling
                A[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                # Apply average pooling
                A[:, i, j, :] = np.mean(window, axis=(1, 2))

    return A
