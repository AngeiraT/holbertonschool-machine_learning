#!/usr/bin/env python3
"""Script to perform a padding convolution"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function to perform a grayscale same convolution
    Args:
        images: numpy.ndarray with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the
                kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
    Returns: numpy.ndarray containing the convolved images

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding  # padding height and width
    output_h = h + (2 * pad_h) - kh + 1
    output_w = w + (2 * pad_w) - kw + 1

    image_pad = np.pad(images, pad_width=((0, 0),
                                          (pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant')

    # convolution output
    conv_out = np.zeros((m, output_h, output_w))

    image = np.arange(m)
    # Loop every pixel of the output
    for x in range(output_h):
        for y in range(output_w):
            # element wise multiplication of the kernel and the image
            patch = image_pad[image, x:x+kh, y:y+kw]
            conv_out[image, x, y] = np.sum(patch * kernel,
                                           axis=(1, 2))
    return conv_out
