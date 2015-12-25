import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import scipy.misc

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("updates_num", 100000, "number of updates in training")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
flags.DEFINE_string("mode", "MLP", "MLP | ConvNet")

FLAGS = flags.FLAGS

def batch_norm(input, depth):
    """Returns a batch-normalized version of x."""
    beta = tf.Variable(tf.constant(0.0, shape=[depth]))
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
    epsilon = 1e-9

    mean, variance = tf.nn.moments(input, [0, 1, 2])
    return tf.nn.batch_norm_with_global_normalization(
        input, mean, variance, beta, gamma,
        epsilon, True)

def conv(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer(0.0, math.sqrt(2.0/(kernel_shape[0]*kernel_shape[1]*kernel_shape[2]))))
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return conv + biases
    
    
def deconv(input, output_size, kernel_shape, bias_shape, bias_value=None, padding='SAME'):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer(0.0, math.sqrt(2.0/(kernel_shape[0]*kernel_shape[1]*kernel_shape[3]))))
    # Create variable named "biases".
    if bias_value is None:
        bias_value = 0.0
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(bias_value))
    conv = tf.nn.deconv2d(input, weights, output_size, strides=[1, 2, 2, 1], padding=padding)
    return conv + biases


def linear(input,  kernel_shape, bias_shape, bias_value=None):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(0.0, math.sqrt(2.0/kernel_shape[0])))
    # Create variable named "biases".
    if bias_value is None:
        bias_value = 0.0
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(bias_value))
    linear = tf.nn.xw_plus_b(input, weights, biases)
    return linear + biases


def discriminator(input):
    if FLAGS.mode == "MLP":
        with tf.variable_scope("D1"):
            output1 = tf.nn.dropout(tf.nn.elu(linear(tf.nn.dropout(input,0.9), [28*28, 1024], 1024)), 0.5)
        with tf.variable_scope("D2"):
            output2 = tf.nn.dropout(tf.nn.elu(linear(output1, [1024, 1024], 1024)), 0.5)
        with tf.variable_scope("D3"):
            output3 = linear(output2, [1024, 1], 1)
            return output3
    elif FLAGS.mode == "ConvNet":
        with tf.variable_scope("D0"):
            input = batch_norm(tf.reshape(input, [FLAGS.batch_size, 28, 28, 1]), 1)
        with tf.variable_scope("D1"):
            output1 = tf.nn.elu(batch_norm(conv(input, [5, 5, 1, 32], 32), 32))
            output1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope("D2"):
            output2 = tf.nn.elu(batch_norm(conv(output1, [5, 5, 32, 64], 64), 64))
            output2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
            output2 = tf.nn.dropout(tf.reshape(output2, [FLAGS.batch_size, 7*7*64]), 0.5)
        with tf.variable_scope("D3"):
            output3 = tf.nn.dropout(tf.nn.elu(linear(output2, [7*7*64, 1024], 1024)), 0.5)
        with tf.variable_scope("D4"):
            output4 = linear(output3, [1024, 1], 1)
            return output4                    

def generator(bias_value=-2.0):
    if FLAGS.mode == "MLP":
        with tf.variable_scope("G0"):
            input = tf.random_uniform([FLAGS.batch_size, 100], 0.0, 1.0)
        with tf.variable_scope("G1"):
            output1 = tf.nn.elu(linear(input, [100, 1024], 1024))
        with tf.variable_scope("G2"):
            output2 = tf.nn.elu(linear(output1, [1024, 1024], 1024))
        with tf.variable_scope("G3"):
            output3 = linear(output2, [1024, 28*28], 28*28, bias_value)
            return tf.nn.sigmoid(output3)
    elif FLAGS.mode == "ConvNet":
        with tf.variable_scope("G0"):
            input = batch_norm(tf.random_uniform([FLAGS.batch_size, 1, 1, 100], 0.0, 1.0), 100)
        with tf.variable_scope("G1"):
            output1 = tf.nn.elu(batch_norm(deconv(input, [FLAGS.batch_size, 4, 4, 128], [4, 4, 128, 100], 128, padding='VALID'), 128))
        with tf.variable_scope("G2"):
            output2 = tf.nn.elu(batch_norm(deconv(output1, [FLAGS.batch_size, 7, 7, 64], [5, 5, 64, 128], 64), 64))
        with tf.variable_scope("G3"):
            output3 = tf.nn.elu(batch_norm(deconv(output2, [FLAGS.batch_size, 14, 14, 32], [5, 5, 32, 64], 32), 32))
        with tf.variable_scope("G4"):
            output4 = batch_norm(deconv(output3, [FLAGS.batch_size, 28, 28, 1], [5, 5, 1, 32], 1, bias_value), 1)
            return tf.nn.sigmoid(output4)


if __name__ == "__main__":
    mnist = input_data.read_data_sets("/work/kostrikov/data/MNIST/", one_hot=True)
    
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28*28])
    
    with tf.variable_scope("model") as scope:
        D1 = discriminator(input_tensor)
        print(len(tf.trainable_variables()))
        G = generator()
        scope.reuse_variables()
        D2 = discriminator(G)
        
        D_loss = tf.reduce_mean(tf.nn.relu(D1)-D1+tf.log(1.0+tf.exp(-tf.abs(D1))))+tf.reduce_mean(tf.nn.relu(D2)+tf.log(1.0+tf.exp(-tf.abs(D2))))
        
        G_loss = tf.reduce_mean(tf.nn.relu(D2)-D2+tf.log(1.0+tf.exp(-tf.abs(D2))))


    momentum = tf.placeholder(tf.float32)
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum)
    params = tf.trainable_variables()
    if FLAGS.mode == "MLP":
        D_params = params[:6] #TODO: FIX THIS
        G_params = params[6:]
    else:
        D_params = params[:14] #TODO: FIX THIS
        G_params = params[14:]
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
