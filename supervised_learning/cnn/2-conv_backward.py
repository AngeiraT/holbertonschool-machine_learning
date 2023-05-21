#!/usr/bin/env python3
""" Script to forward propagate over a convolutional layer in a NN"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform back propagation over a convolutional layer of a neural network.

    Arguments:
    dZ -- numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the unactivated output of the convolutional layer
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of the previous layer
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
    b -- numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    padding -- string indicating the type of padding used: "same" or "valid" (default is "same")
    stride -- tuple of (sh, sw) containing the strides for the convolution (default is (1, 1))

    Returns:
    dA_prev -- partial derivatives with respect to the previous layer (numpy.ndarray)
    dW -- partial derivatives with respect to the kernels (numpy.ndarray)
    db -- partial derivatives with respect to the biases (numpy.ndarray)
    """

    # Retrieve the dimensions from A_prev shape
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape

    # Retrieve the values of the stride
    sh, sw = stride

    # Initialize output gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

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


   # Iterate over the training examples
    for i in range(m):
        # Select the current example
        a_prev_pad = A_prev_pad[i]
        da_prev = dA_prev[i]

        # Iterate over the vertical axis
        for h in range(h_new):
            vert_start = h * sh
            vert_end = vert_start + kh

           # Iterate over the horizontal axis
            for w in range(w_new):
                horiz_start = w * sw
                horiz_end = horiz_start + kw
            
            # Iterate over the channels
            for c in range(c_new):
                # Compute gradients for the current slice
                a_slice = a_prev_pad[:, vert_start:vert_end,
                                     horiz_start:horiz_end, :]
                da_prev_slice = da_prev[vert_start:vert_end,
                                        horiz_start:horiz_end, :]
                dW_slice = a_slice * dZ[i, h, w, c]
                db_slice = dZ[i, h, w, c]
                
                # Accumulate gradients
                da_prev[vert_start:vert_end,
                        horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                dW[:, :, :, c] += dW_slice
                db
