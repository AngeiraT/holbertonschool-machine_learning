#!/usr/bin/env python3
""" Script to back propagation over a pooling layer in a NN"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function to back propagation over a pooling layer in a CNN
    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output of
            the convolutional layer
            m: the number of examples
            h_new: the height of the output
            w_new: the width of the output
            c_new: the number of channels
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
                      the pooling
                      kh: the kernel height
                      kw: the kernel width
        stride: tuple of (sh, sw) containing the strides for the convolution
                sh: the stride for the height
                sw: the stride for the width
        mode: string containing either max or avg, indicating whether to
              perform maximum or average pooling, respectively
    Returns: the partial derivatives with respect to the previous layer
             (dA_prev)
    """
    # Retrieve shapes
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    if mode == 'max':
                        a_slice = a_prev[v_start:v_end, h_start:h_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, v_start:v_end, h_start:h_end, c] +=\
                            np.multiply(mask, dA[i, h, w, c])
                    else:
                        da = dA[i, h, w, c]
                        shape = kernel_shape
                        avg = da / (kh * kw)
                        Z = np.ones(shape) * avg
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += Z
    return dA_prev
