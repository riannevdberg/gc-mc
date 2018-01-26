import tensorflow as tf
import numpy as np


def weight_variable_truncated_normal(input_dim, output_dim, name=""):
    """Create a weight variable with truncated normal distribution, values
    that are more than 2 stddev away from the mean are redrawn."""

    initial = tf.truncated_normal([input_dim, output_dim], stddev=0.5)
    return tf.Variable(initial, name=name)


def weight_variable_random_uniform(input_dim, output_dim=None, name=""):
    """Create a weight variable with variables drawn from a
    random uniform distribution. Parameters used are taken from paper by
    Xavier Glorot and Yoshua Bengio:
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf"""
    if output_dim is not None:
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = tf.random_uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    else:
        init_range = np.sqrt(6.0 / input_dim)
        initial = tf.random_uniform([input_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def weight_variable_random_uniform_relu(input_dim, output_dim, name=""):
    """Create a weight variable with variables drawn from a
    random uniform distribution. Parameters used are taken from paper by
    Xavier Glorot and Yoshua Bengio:
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    and are optimized for ReLU activation function."""

    init_range = np.sqrt(2.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def bias_variable_truncated_normal(shape, name=""):
    """Create a bias variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial, name=name)


def bias_variable_zero(shape, name=""):
    """Create a bias variable initialized as zero."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def bias_variable_one(shape, name=""):
    """Create a bias variable initialized as ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def orthogonal(shape, scale=1.1, name=None):
    """
    From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[:shape[0], :shape[1]], name=name, dtype=tf.float32)


def bias_variable_const(shape, val, name=""):
    """Create a bias variable initialized as zero."""
    value = tf.to_float(val)
    initial = tf.fill(shape, value, name=name)
    return tf.Variable(initial, name=name)