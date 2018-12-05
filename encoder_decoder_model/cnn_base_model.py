# -*- coding: utf-8 -*-
# @Time    : 18-12-05
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : cnn_basenet.py
# @IDE     : PyCharm Community Edition

"""
the base convolution neural network mainly implements some useful cnn functions
"""

import tensorflow as tf
import numpy as np

class CNNBaseModel(object):
    """
    base model for other specific cnn models, such as vgg, fcn, denseNet
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(input_data, out_channel, kernel_size,
              padding='SAME',  stride='1', w_init=None, b_init=None,
              split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function

        :param out_channel:     number of output channels
        :param kernel_size:     list or int, if it's int, the kernel shape will be [kernel_size, kernel_size]
        :param padding:         'VALID' or 'SAME'
        :param stride:          list or int, if it's int, the stride shape will be [stride, stride]
        :param w_init:          initializer for convolution weights
        :param b_init:          initializer for bias
        :param split:           split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:        whether to use bias.
        :param data_format:     'NHWC' or 'NCHW', default set to 'NHWC' according tensorflow
        :param name:            operation name

        :return:                tf.Tensor named 'output'
        """
        with tf.variable_scope(name):
            in_shape = input_data.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]

            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0, "param split is not satisfied for in_channel"
            assert out_channel % split == 0, "param split is not satisfied for out_channel"

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel/split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel/split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()           #??? variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)              #??? get_variable()
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)         #??? 为什么是[out_channel]

            if split == 1:
                conv = tf.nn.conv2d(input_data, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(input_data, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, data_format=data_format)

        return ret

    @staticmethod
    def relu(input_data, name=None):
        """

        :param input_data:
        :param name:
        :return:
        """
        return tf.nn.relu(features=input_data, name=name)

    @staticmethod
    def sigmoid(input_data, name=None):
        """

        :param input_data:
        :param name:
        :return:
        """
        return tf.nn.sigmoid(x=input_data, name=name)

    @staticmethod
    def max_pooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :param name:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avg_pooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """


        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def global_avg_pooling(inputdata, data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param data_format:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layer_norm(input_data, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """

        :param input_data:
        :param epsilon:         epsilon to avoid divide-by-zero.
        :param use_bias:        whether to use the extra affine transformation or not.
        :param use_scale:       whether to use the extra affine transformation or not.
        :param data_format:
        :param name:
        :return:
        """

    @staticmethod
    def instance_norm(input_data, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :param name:
        :return:
        """

    @staticmethod
    def dropout(input_data, keep_prob, noise_shape=None, name=None):
        """

        :param input_data:
        :param keep_prob:
        :param noise_shape:
        :param name:
        :return:
        """

    @staticmethod
    def fully_connect(inputdata, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """

        :param inputdata:
        :param out_dim:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """


        :param inputdata:
        :param out_channel:
        :param kernel_size:
        :param padding:
        :param stride:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param activation:
        :param data_format:
        :param trainable:
        :param name:
        :return:
        """

    @staticmethod
    def dilation_conv():
        return None
