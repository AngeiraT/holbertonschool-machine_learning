#!/usr/bin/env python3
""" Script to forward propagate over a convolutional layer in a NN"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function to forward propagate over a convolutional layer in a NN
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
                m: is the number of examples
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
                c_prev: the number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution
            kh: the filter height
            kw: the filter width
            c_prev: the number of channels in the previous layer
            c_new: the number of channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        activation: an activation function applied to the convolution
        padding: string that is either same or valid, indicating the type of
                 padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
                sh: the stride for the height
                sw: the stride for the width
    Returns: the output of the convolutional layer
    """

    # Retrieve the dimensions from A_prev shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape

    # Retrieve the values of the stride
    sh, sw = stride

    if padding == "same":
        padh = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        padw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
        # Create an image pad using np.pad
        A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (padh, padh),
                                          (padw, padw), (0, 0)),
                        mode='constant')
    elif padding == "valid":
        padh = padw = 0  # Padding values
        A_prev_pad = A_prev

    # Compute the dimensions of the CONV output volume
    h_out = int((h_prev - kh + 2 * padh) / sh) + 1
    w_out = int((w_prev - kw + 2 * padw) / sw) + 1

    # Initialize the output volume A (Z) with zeros
    A = np.zeros((m, h_out, w_out, c_new))
    # Loop over the vertical_ax, then horizontal_ax, then over channel
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw
                a_slice_prev = A_prev_pad[:,
                                           vert_start:vert_end,
                                            horiz_start:horiz_end, :]
                Z = np.sum(a_slice_prev * W[:, :, :, k],
                        axis=(1, 2, 3)) + b[:, :, :, k]
                A[:, i, j, k] = activation(Z)

    return A
