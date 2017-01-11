'''TensorFlow implementation of http://arxiv.org/pdf/1502.04623v2.pdf

DISCLAIMER
Work in progress. This code requires massive refactoring.
'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("rnn_size", 128, "size of the hidden VAE unit")
flags.DEFINE_integer("rnn_len", 16, "size of the hidden VAE unit")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_integer("N", 5, "crop size")

FLAGS = flags.FLAGS


# inspired by https://github.com/jbornschein/draw
# TODO: better names for N, A, B
def filterbank_matrices(g_x, g_y, delta, sigma, N, A, B):
    ''' Computer filter bank matrices. All inputs are in batches.

    Args:
        g_x, g_y: grid centers, relative to the center of the image
        delta: strides
        sigma: isotropic variance
        N: grid dimension
        A, B: input image dimensions, width and height
    Returns:
        F_x, F_y: filter banks matrices [batch, N, A] and [batch, N, B]
    '''

    rng = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])

    # eq 19
    mu_x = g_x + (rng - N / 2 - 0.5) * delta

    # eq 20
    mu_y = g_y + (rng - N / 2 - 0.5) * delta

    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])

    # reshape for broadcasting
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma = tf.reshape(sigma, [-1, 1, 1])

    F_x = tf.exp(-tf.square((a - mu_x) / sigma))
    F_y = tf.exp(-tf.square((b - mu_y) / sigma))

    # transform in a convenient form for further use
    return F_x, F_y


def apply_filters(image, F_x, F_y, gamma, N, A, B, forward=True, epsilon=1e-9):
    '''Apply a batch of filter banks to a batch of images.

    Args:
        image: image, [batch, w, h, c]
        F_x, F_y: filter banks matrices [batch, N, A] and [batch, N, B]
    Returns:
        filtered image
    '''

    F_x = F_x / tf.maximum(tf.reduce_sum(F_x, 2, keep_dims=True), epsilon)
    F_y = F_y / tf.maximum(tf.reduce_sum(F_y, 2, keep_dims=True), epsilon)
    if forward:
        F_y = tf.reshape(F_y, [-1, N, B, 1, 1])
        image = tf.reshape(image, [-1, 1, B, A, 1])
        image = tf.tile(image, [1, N, 1, 1, 1])
        image = tf.reduce_sum(F_y * image, 2)

        image = tf.reshape(image, [-1, N, A, 1, 1])
        F_x = tf.transpose(F_x, [0, 2, 1])
        F_x = tf.reshape(F_x, [-1, 1, A, N, 1])
        F_x = tf.tile(F_x, [1, N, 1, 1, 1])

        image = tf.reduce_sum(image * F_x, 2)

        return image * tf.reshape(gamma, [-1, 1, 1, 1])
    else:
        F_y = tf.transpose(F_y, [0, 2, 1])
        F_y = tf.reshape(F_y, [-1, B, N, 1, 1])
        image = tf.reshape(image, [-1, 1, N, N, 1])
        image = tf.tile(image, [1, B, 1, 1, 1])
        image = tf.reduce_sum(F_y * image, 2)

        image = tf.reshape(image, [-1, B, N, 1, 1])
        F_x = tf.reshape(F_x, [-1, 1, N, A, 1])
        F_x = tf.tile(F_x, [1, B, 1, 1, 1])
        image = tf.reduce_sum(image * F_x, 2)

        return image * tf.reshape(1.0 / gamma, [-1, 1, 1, 1])


def transform_params(input_tensor, N, A, B):
    '''Transformes a raw NN output into a set of parameters
        See the paper.
    Args:
        input_tensor:
        N:
        A:
        B:
    '''
    g_x, g_y, log_sigma_sqr, log_delta, log_gamma = tf.split(1, 5, input_tensor)
    g_x = (A + 1) / 2 * (g_x + 1)
    g_y = (B + 1) / 2 * (g_y + 1)
    sigma = tf.exp(log_sigma_sqr / 2.0)
    delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)
    gamma = tf.exp(log_gamma)
    return g_x, g_y, delta, sigma, gamma


def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean:
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''
    return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                         (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])
    output_tensor = tf.zeros([FLAGS.batch_size, 28 * 28], tf.float32)
    sampled_tensor = tf.zeros([FLAGS.batch_size, 28 * 28], tf.float32)

    # TODO: Remove magic number
    encoder_state = (pt.wrap(tf.zeros([FLAGS.batch_size, FLAGS.rnn_size], tf.float32)),)
    decoder_state = (pt.wrap(tf.zeros([FLAGS.batch_size, FLAGS.rnn_size], tf.float32)),)

    sampled_state = (pt.wrap(tf.zeros([FLAGS.batch_size, FLAGS.rnn_size], tf.float32)),)

    sampled_tensors = []
    glimpse_tensors = []
    write_tensors = []
    params_tensors = []

    loss = 0.0
    with tf.variable_scope("model"):
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.1,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            # Encoder RNN (Eq. 5)
            encoder_template = (pt.template('input').
                                gru_cell(num_units=FLAGS.rnn_size, state=pt.UnboundVariable('state')))

            # Projection of encoder RNN output (Eq. 1-2)
            encoder_proj_template = (pt.template('input').
                                     fully_connected(FLAGS.hidden_size * 2, activation_fn=None))

            # Params of read from decoder RNN output (Eq. 21)
            decoder_read_params_template = (pt.template('input').
                                            fully_connected(5, activation_fn=None))

            # Decoder RNN (Eq. 7)
            decoder_template = (pt.template('input').
                                gru_cell(num_units=FLAGS.rnn_size, state=pt.UnboundVariable('state')))

            # Projection of decoder RNN output (Eq. 18)
            decoder_proj_template = (pt.template('input').
                                     fully_connected(FLAGS.N * FLAGS.N, activation_fn=None))

            # Projection of decoder RNN output (Eq. 18)
            decoder_write_params_template = (pt.template('input').
                                             fully_connected(5, activation_fn=None))

            for _ in range(FLAGS.rnn_len):
                epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])

                # For unknown reason combination of batch normalization and phases within
                # templates doesn't work. Hopefully, the workaround below works as
                # intended.
                with pt.defaults_scope(phase=pt.Phase.train):
                    attention_params = decoder_read_params_template.construct(
                        input=decoder_state[0].tensor)
                    g_x, g_y, delta, sigma, gamma = transform_params(
                        attention_params, FLAGS.N, 28, 28)
                    F_x, F_y = filterbank_matrices(
                        g_x, g_y, delta, sigma, FLAGS.N, 28, 28)
                    image_tensor = tf.reshape(input_tensor, [FLAGS.batch_size, 28, 28, 1])
                    image_glipse = apply_filters(image_tensor, F_x, F_y, gamma, FLAGS.N, 28, 28)

                    image_hat_tensor = tf.reshape(
                        input_tensor - tf.nn.sigmoid(output_tensor), [FLAGS.batch_size, 28, 28, 1])
                    image_hat_glipse = apply_filters(
                        image_hat_tensor, F_x, F_y, gamma, FLAGS.N, 28, 28)

                    encoder_input_tensor = pt.wrap(
                        tf.concat(1, [tf.reshape(image_glipse, [FLAGS.batch_size, -1]), tf.reshape(image_hat_glipse, [FLAGS.batch_size, -1]), decoder_state[0].tensor]))

                    encoded_tensor, encoder_state = encoder_template.construct(
                        input=encoder_input_tensor, state=encoder_state[0].tensor)

                    hidden_tensor = encoder_proj_template.construct(input=encoded_tensor)
                    mean = hidden_tensor[:, :FLAGS.hidden_size]
                    stddev = tf.sqrt(tf.exp(hidden_tensor[:, FLAGS.hidden_size:]))
                    input_sample = mean + epsilon * stddev

                    decoder_output_tensor, decoder_state = decoder_template.construct(
                        input=input_sample, state=decoder_state[0].tensor)

                    attention_params = decoder_write_params_template.construct(
                        input=decoder_state[0].tensor)
                    g_x, g_y, delta, sigma, gamma = transform_params(
                        attention_params, FLAGS.N, 28, 28)
                    F_x, F_y = filterbank_matrices(
                        g_x, g_y, delta, sigma, FLAGS.N, 28, 28)

                    decoder_output_image_tensor = decoder_proj_template.construct(
                        input=decoder_output_tensor)

                    image_tensor = tf.reshape(decoder_output_image_tensor, [
                                              FLAGS.batch_size, FLAGS.N, FLAGS.N, 1])
                    image_glipse = apply_filters(
                        image_tensor, F_x, F_y, gamma, FLAGS.N, 28, 28, False)

                    output_tensor = output_tensor + \
                        tf.reshape(image_glipse, [FLAGS.batch_size, -1])

                    vae_loss = get_vae_cost(mean, stddev)

                    loss = loss + vae_loss

                with pt.defaults_scope(phase=pt.Phase.test):
                    decoder_output_tensor, sampled_state = decoder_template.construct(
                        input=epsilon, state=sampled_state[0].tensor)

                    attention_params = decoder_write_params_template.construct(
                        input=sampled_state[0].tensor)

                    params_tensors.append(attention_params)

                    g_x, g_y, delta, sigma, gamma = transform_params(
                        attention_params, FLAGS.N, 28, 28)
                    F_x, F_y = filterbank_matrices(
                        g_x, g_y, delta, sigma, FLAGS.N, 28, 28)

                    decoder_output_image_tensor = decoder_proj_template.construct(
                        input=decoder_output_tensor)

                    image_tensor = tf.reshape(decoder_output_image_tensor, [
                                              FLAGS.batch_size, FLAGS.N, FLAGS.N, 1])

                    glimpse_tensors.append(tf.nn.sigmoid(
                        tf.reshape(1.0 / gamma, [-1, 1, 1, 1]) * image_tensor))

                    image_glipse = apply_filters(
                        image_tensor, F_x, F_y, gamma, FLAGS.N, 28, 28, False)

                    write_tensors.append(tf.nn.sigmoid(image_glipse))

                    sampled_tensor = sampled_tensor + \
                        tf.reshape(image_glipse, [FLAGS.batch_size, -1])

                    sampled_tensors.append(tf.nn.sigmoid(sampled_tensor))

    rec_loss = get_reconstruction_cost(tf.nn.sigmoid(output_tensor), input_tensor)
    loss = loss + rec_loss

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.5)
    train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(max_value = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x, _ = mnist.train.next_batch(FLAGS.batch_size)
                _, loss_value = sess.run([train, loss], {input_tensor: x})
                training_loss += loss_value

            training_loss = training_loss / \
                (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

            print("Loss %f" % training_loss)

            results = sess.run(sampled_tensors + write_tensors + glimpse_tensors + params_tensors)

            imgs = []
            write_imgs = []
            glimpse_imgs = []
            img_params = []

            for i in range(len(results) // 4):
                imgs.append(results[i])
                write_imgs.append(results[i + len(results) // 4])
                glimpse_imgs.append(results[i + len(results) // 4 * 2])
                img_params.append(results[i + len(results) // 4 * 3])

            for k in range(FLAGS.batch_size):
                imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)
                for i in range(len(imgs)):
                    imsave(os.path.join(imgs_folder, '%d_%d.png') % (k, i),
                           imgs[i][k].reshape(28, 28))

                    imsave(os.path.join(imgs_folder, '%d_%d_w.png') % (k, i),
                           write_imgs[i][k].reshape(28, 28))

                    imsave(os.path.join(imgs_folder, '%d_%d_g.png') % (k, i),
                           glimpse_imgs[i][k].reshape(FLAGS.N, FLAGS.N))
