#!/usr/bin/env python3
"""Module class NST."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor.

        Args:
            style_image (ndarray): image used as a style reference
            content_image (ndarray): image used as a content reference
            alpha (float, optional): weight for content cost. Defaults to 1e4.
            beta (int, optional): weight for style cost. Defaults to 1.
        """
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1 and\
        its largest side is 512 pixels.

        Args:
            image (ndarray): contains the image to be scaled
        Returns:
            scaled image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))
        image = image[tf.newaxis, :]
        image = tf.image.resize_bicubic(image, new_shape)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image
