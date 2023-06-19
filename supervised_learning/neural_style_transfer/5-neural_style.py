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
        self.load_model()
        self.generate_features()

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

    def load_model(self):
        """Create the model used to calculate cost."""
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet',
                                               input_tensor=None,
                                               input_shape=None, pooling=None,
                                               classes=1000)
        custom_object = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_vgg.save('base_vgg')
        vgg = tf.keras.models.load_model('base_vgg',
                                         custom_objects=custom_object)
        for layer in vgg.layers:
            layer.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in
                         self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate gram matrices.

        Args:
            input_layer (tf.Tensor or tf.Variable): contains the layer output
                                                    whose gram matrix should
                                                    be calculated
        Returns:
            tf.Tensor containing the gram matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        result = result / num_locations
        return result

    def generate_features(self):
        """Extract the features used to calculate neural style cost."""
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        outputs_style = self.model(style_image)
        style_outputs = outputs_style[:-1]
        outputs_content = self.model(content_image)
        content_output = outputs_content[-1]
        self.gram_style_features = [self.gram_matrix(style_output) for
                                    style_output in style_outputs]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate the style cost for a single layer.

        Args:
            style_output (Tensor): contains the layer style output of the
                                   generated image
            gram_target (Tensor): gram matrix of the target style output for
                                  that layer
        Returns:
            layer's style cost
        """
        c = style_output.shape[-1]
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError("gram_target must be a tensor of shape\
                [1, {}, {}]".format(c, c))
        if gram_target.shape != (1, c, c):
            raise TypeError("gram_target must be a tensor of shape\
                [1, {}, {}]".format(c, c))
        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))
        return style_cost

    def style_cost(self, style_outputs):
        """
        Calculate the style cost.

        Args:
            style_outputs (Tensor): style outputs for the generated image
        Returns:
            style cost
        """
        if not isinstance(style_outputs, list):
            raise TypeError("style_outputs must be a list with a length of\
                {}".format(len(self.style_layers)))
        if len(self.style_layers) != len(style_outputs):
            raise TypeError("style_outputs must be a list with a length of\
                {}".format(len(self.style_layers)))
        style_costs = []
        weight = 1 / len(self.style_layers)
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            weighted_layer_style_cost = weight * layer_style_cost
            style_costs.append(weighted_layer_style_cost)
        style_cost = tf.add_n(style_costs)
        return style_cost
