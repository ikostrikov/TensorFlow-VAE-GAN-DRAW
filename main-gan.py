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
flags.DEFINE_float("learning_rate", 0.1, "learning rate")

FLAGS = flags.FLAGS

def discriminator(input):
    return (pt.wrap(input).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(1, activation_fn=None)).tensor

def generator():
    input_tensor = tf.random_uniform([FLAGS.batch_size, 1, 1, 100], -1.0, 1.0)
    return (pt.wrap(input_tensor).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor

if __name__ == "__main__":
    mnist = input_data.read_data_sets("/work/kostrikov/data/MNIST/", one_hot=True)
    
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28*28])

    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.1,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            D1 = discriminator(input_tensor)
            D_params_num = len(tf.trainable_variables())
            G = generator()
    with tf.variable_scope("model", reuse=True) as scope:
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.1,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            D2 = discriminator(G)

    D_loss = tf.reduce_mean(tf.nn.relu(D1)-D1+tf.log(1.0+tf.exp(-tf.abs(D1))))+tf.reduce_mean(tf.nn.relu(D2)+tf.log(1.0+tf.exp(-tf.abs(D2))))            
    G_loss = tf.reduce_mean(tf.nn.relu(D2)-D2+tf.log(1.0+tf.exp(-tf.abs(D2))))

    momentum = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    params = tf.trainable_variables()
    D_params = params[:D_params_num]
    G_params = params[D_params_num:]
    train_discrimator = optimizer.minimize(loss=D_loss, var_list=D_params)
    train_generator = optimizer.minimize(loss=G_loss, var_list=G_params)

    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(FLAGS.updates_num):
            
            x, _ = mnist.train.next_batch(FLAGS.batch_size)
            _, loss_value = sess.run([train_discrimator, D_loss], {input_tensor: x, momentum: 0.5+min(0.4, 0.4*epoch/FLAGS.updates_num)})
            print(epoch, "D", loss_value)
            
            _, loss_value = sess.run([train_generator, G_loss], {momentum: 0.5+min(0.4, 0.4*epoch/FLAGS.updates_num)})
            print(epoch, "G", loss_value)
            
            if epoch > 0 and epoch % 100 == 0:
                imgs = sess.run(G)
                for k in range(FLAGS.batch_size):
                    scipy.misc.imsave('/work/kostrikov/tmp_img/%d.png' % k, imgs[k].reshape(28, 28))
