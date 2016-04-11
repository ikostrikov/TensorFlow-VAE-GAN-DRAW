'''TensorFlow implementation of http://arxiv.org/pdf/1511.06434.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("g_learning_rate", 1e-2, "learning rate")
flags.DEFINE_float("d_learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")

FLAGS = flags.FLAGS


def discriminator(input_tensor):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(1, activation_fn=None)).tensor


def get_discrinator_loss(D1, D2):
    '''Loss for the discriminator network

    Args:
        D1: logits computed with a discriminator networks from real images
        D2: logits computed with a discriminator networks from generated images

    Returns:
        Cross entropy loss, positive samples have implicit labels 1, negative 0s
    '''
    return tf.reduce_mean(tf.nn.relu(D1) - D1 + tf.log(1.0 + tf.exp(-tf.abs(D1)))) + \
        tf.reduce_mean(tf.nn.relu(D2) + tf.log(1.0 + tf.exp(-tf.abs(D2))))


def generator():
    '''Create a network that generates images
    TODO: Add fixed initialization, so we can draw interpolated images

    Returns:
        A deconvolutional (not true deconv, transposed conv2d) network that
        generated images.
    '''
    input_tensor = tf.random_uniform([FLAGS.batch_size, 1, 1, 100], -1.0, 1.0)
    return (pt.wrap(input_tensor).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor


def get_generator_loss(D2):
    '''Loss for the genetor. Maximize probability of generating images that
    discrimator cannot differentiate.

    Returns:
        see the paper
    '''
    return tf.reduce_mean(tf.nn.relu(D2) - D2 + tf.log(1.0 + tf.exp(-tf.abs(D2))))

  
if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])


    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
            with tf.variable_scope("model"):
                D1 = discriminator(input_tensor)  # positive examples
                D_params_num = len(tf.trainable_variables())
                G = generator()

            with tf.variable_scope("model", reuse=True):
                D2 = discriminator(G)  # generated examples

    D_loss = get_discrinator_loss(D1, D2)
    G_loss = get_generator_loss(D2)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
    params = tf.trainable_variables()
    D_params = params[:D_params_num]
    G_params = params[D_params_num:]
#    train_discrimator = optimizer.minimize(loss=D_loss, var_list=D_params)
#    train_generator = optimizer.minimize(loss=G_loss, var_list=G_params)
    train_discrimator = pt.apply_optimizer(optimizer, losses=[D_loss], regularize=True, include_marked=True, var_list=D_params)
    train_generator = pt.apply_optimizer(optimizer, losses=[G_loss], regularize=True, include_marked=True, var_list=G_params)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(FLAGS.max_epoch):

            discriminator_loss = 0.0
            generator_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(max_value = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, _ = mnist.train.next_batch(FLAGS.batch_size)
                _, loss_value = sess.run([train_discrimator, D_loss], {input_tensor: x, learning_rate: FLAGS.d_learning_rate})
                discriminator_loss += loss_value

                # We still need input for moving averages.
                # Need to find how to fix it.
                _, loss_value = sess.run([train_generator, G_loss], {input_tensor: x, learning_rate: FLAGS.g_learning_rate})
                generator_loss += loss_value

            discriminator_loss = discriminator_loss / FLAGS.updates_per_epoch
            generator_loss = generator_loss / FLAGS.updates_per_epoch

            print("Gen. loss %f, Disc. loss: %f" % (generator_loss,
                                                    discriminator_loss))

            imgs = sess.run(G)
            for k in range(FLAGS.batch_size):
                imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)

                imsave(os.path.join(imgs_folder, '%d.png') % k,
                       imgs[k].reshape(28, 28))
