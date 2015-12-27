import tensorflow as tf
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import scipy.misc
from deconv import deconv2d

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_num", 100000, "number of updates in training")
flags.DEFINE_float("learning_rate", 0.01, "learning rate")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")

FLAGS = flags.FLAGS

def encoder(input_tensor):
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(FLAGS.hidden_size*2, activation_fn=None)).tensor

def decoder(input_tensor=None):
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:,:FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:,FLAGS.hidden_size:]))        
        input_sample = mean+epsilon*stddev
    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor, mean, stddev

if __name__ == "__main__":
    mnist = input_data.read_data_sets("/work/kostrikov/data/MNIST/", one_hot=True)
    
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28, 28, 1])

    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.1,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            output_tensor, mean, stddev = decoder(encoder(input_tensor))

    with tf.variable_scope("model", reuse=True) as scope:
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.1,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            sampled_tensor, _, _ = decoder()

    vae_loss = tf.reduce_sum(0.5*(tf.square(mean)+tf.square(stddev)-2.0*tf.log(stddev+1e-8)-1.0))
    rec_loss = tf.reduce_sum(-input_tensor*tf.log(output_tensor+1e-8)-(1.0-input_tensor)*tf.log(1.0-output_tensor+1e-8))
    
    loss = vae_loss + rec_loss
    
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = optimizer.minimize(loss=loss)

    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(FLAGS.updates_num):
            x, _ = mnist.train.next_batch(FLAGS.batch_size)
            x = x.reshape([-1, 28, 28, 1])
            _, loss_value = sess.run([train, loss], {input_tensor: x})
            print(epoch, loss_value)
            mean_value, stddev_value = sess.run([mean, stddev], {input_tensor: x})
            print(np.mean(np.abs(stddev_value)))
            if epoch > 0 and epoch % 100 == 0:
                y = sess.run(sampled_tensor)
                for k in range(FLAGS.batch_size):
                    scipy.misc.imsave('/work/kostrikov/tmp_img/%d.png' % k, y[k,:,:,0])
