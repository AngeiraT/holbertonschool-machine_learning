#!/usr/bin/env python3
"""Script to perform a strided convolution"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function to perform a convolution of img with channels
    Args:
        images: numpy.ndarray with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw, c) containing
                the kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
                c:  the number of channels in the image
        padding: is either a tuple of (ph, pw), 'same, 'valid'
                 ph: is the padding for the height of the image
                 pw: is the padding for the width of the image
        stride: is a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
    Returns: numpy.ndarray containing the convolved images

    """
    m, h, w, c = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int((((h - 1) * sh + kh - h) / 2) + 1)
        pad_w = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    image_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                          (pad_w, pad_w), (0, 0)), 
                                          mode='constant')

    output_h = int(((h + 2 * pad_h - kh) / sh) + 1)
    output_w = int(((w + 2 * pad_w - kh) / sw) + 1)

    # convolution output
    conv_out = np.zeros((m, output_h, output_w))

    image = np.arange(m)
    # Loop every pixel of the output
    for x in range(output_h):
        for y in range(output_w):
            # element wise multiplication of the kernel and the image
            patch = image_pad[image, x * sh:((x * sh) + kh),
                                            y * sw:((y * sw) + kw)]
            conv_out[image, x, y] = np.sum(patch * kernel,
                                           axis=(1, 2, 3))
    return conv_out
