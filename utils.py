import tensorflow as tf
from tensorflow.contrib import layers


def encoder(input_tensor, output_size):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    net = layers.conv2d(net, 32, 5, stride=2)
    net = layers.conv2d(net, 64, 5, stride=2)
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, output_size, activation_fn=None)


def discriminator(input_tensor):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''

    return encoder(input_tensor, 1)


def decoder(input_tensor):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''

    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
    net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
    net = layers.conv2d_transpose(net, 32, 5, stride=2)
    net = layers.conv2d_transpose(
        net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
    net = layers.flatten(net)
    return net
