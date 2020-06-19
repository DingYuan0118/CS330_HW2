""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.contrib.layers.python import layers as tf_layers

FLAGS = flags.FLAGS

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    # Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter
    # kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]

    normed = tf_layers.batch_norm(conv_output, activation_fn=activation, reuse=reuse, scope=scope)
    return normed

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):  #cross entropy
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.k_shot