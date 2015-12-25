import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import scipy.misc

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("updates_num", 100000, "number of updates in training")
flags.DEFINE_float("learning_rate", 0.01, "learning rate")

FLAGS = flags.FLAGS

def conv(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer(0.0, math.sqrt(2/(kernel_shape[0]*kernel_shape[1]*kernel_shape[2]))))
                              # Create variable named "biases".
                              biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
                              conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
                              return conv + biases

def linear(input,  kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(0.0, math.sqrt(2.0/kernel_shape[0])))
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    linear = tf.nn.xw_plus_b(input, weights, biases)
    return linear + biases


def discriminator(input):
    with tf.variable_scope("D1"):
        output1 = tf.nn.dropout(tf.nn.relu(linear(tf.nn.dropout(input,0.9), [28*28, 1024], 1024)), 0.5)
    with tf.variable_scope("D2"):
        output2 = tf.nn.dropout(tf.nn.relu(linear(output1, [1024, 1024], 1024)), 0.5)
    with tf.variable_scope("D3"):
        output3 = linear(output2, [1024, 1], 1)
        return output3


def generator():
    input = tf.random_uniform([FLAGS.batch_size, 100], -1, 1)
    with tf.variable_scope("G1"):
        output1 = tf.nn.relu(linear(input, [100, 1024], 1024))
    with tf.variable_scope("G2"):
        output2 = tf.nn.relu(linear(output1, [1024, 1024], 1024))
    with tf.variable_scope("G3"):
        output3 = linear(output2, [1024, 28*28], 28*28)
        return tf.nn.sigmoid(output3)

if __name__ == "__main__":
    mnist = input_data.read_data_sets("/work/kostrikov/data/MNIST/", one_hot=True)
    
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28*28])
    
    with tf.variable_scope("model") as scope:
        D1 = discriminator(input_tensor)
        G = generator()
        scope.reuse_variables()
        D2 = discriminator(G)
        
        D_loss = tf.reduce_mean(tf.nn.relu(D1)-D1+tf.log(1.0+tf.exp(-tf.abs(D1))))+tf.reduce_mean(tf.nn.relu(D2)+tf.log(1.0+tf.exp(-tf.abs(D2))))
        
        G_loss = tf.reduce_mean(tf.nn.relu(D2)-D2+tf.log(1.0+tf.exp(-tf.abs(D2))))


    momentum = tf.placeholder(tf.float32)
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum)
    params = tf.trainable_variables()
    D_params = params[:6]
    G_params = params[6:]
    train_discrimator = optimizer.minimize(loss=D_loss, var_list=D_params)
    train_generator = optimizer.minimize(loss=G_loss, var_list=G_params)

init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(FLAGS.updates_num):
            x, _ = mnist.train.next_batch(FLAGS.batch_size)
            _, loss_value = sess.run([train_discrimator, D_loss], {input_tensor: x, momentum: 0.5+min(0.3, 0.3*epoch/FLAGS.updates_num)})
            print(epoch, "D", loss_value)
            
            _, loss_value = sess.run([train_generator, G_loss], {momentum: 0.5+min(0.3, 0.3*epoch/FLAGS.updates_num)})
            print(epoch, "G", loss_value)
            
            if epoch > 0 and epoch % 100 == 0:
                imgs = sess.run(G)
                for k in range(FLAGS.batch_size):
                    scipy.misc.imsave('/work/kostrikov/tmp_img/%d.png' % k, imgs[k].reshape(28, 28))
