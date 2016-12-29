'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope

from utils import encoder, decoder
from generator import Generator


class VAE(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate):
        self.input_tensor = tf.placeholder(
            tf.float32, [None, 28 * 28])

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True}):
            with tf.variable_scope("model") as scope:
                encoded = encoder(self.input_tensor, hidden_size * 2)

                mean = encoded[:, :hidden_size]
                stddev = tf.sqrt(tf.exp(encoded[:, hidden_size:]))

                epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])
                input_sample = mean + epsilon * stddev

                output_tensor = decoder(input_sample)

            with tf.variable_scope("model", reuse=True) as scope:
                self.sampled_tensor = decoder(tf.random_normal(
                    [batch_size, hidden_size]))

        vae_loss = self.__get_vae_cost(mean, stddev)
        rec_loss = self.__get_reconstruction_cost(
            output_tensor, self.input_tensor)

        loss = vae_loss + rec_loss
        self.train = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(
        ), learning_rate=learning_rate, optimizer='Adam', update_ops=[])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __get_vae_cost(self, mean, stddev, epsilon=1e-8):
        '''VAE loss
            See the paper

        Args:
            mean:
            stddev:
            epsilon:
        '''
        return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                    2.0 * tf.log(stddev + epsilon) - 1.0))

    def __get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                             (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images [batch_size, 28*28]

        Returns:
            Current loss value
        '''
        return self.sess.run(self.train, {self.input_tensor: input_tensor})
