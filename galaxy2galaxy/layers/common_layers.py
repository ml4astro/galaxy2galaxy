import contextlib
import tensorflow as tf
import tensorflow_gan as tfgan

import galaxy2galaxy.layers.spectral_ops as ops

def usample(x):
  """Upsamples the input volume.
  Args:
    x: The 4D input tensor.
  Returns:
    An upsampled version of the input tensor.
  """
  # Allow the batch dimension to be unknown at graph build time.
  _, image_height, image_width, n_channels = x.shape.as_list()
  # Add extra degenerate dimension after the dimensions corresponding to the
  # rows and columns.
  expanded_x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=4)
  # Duplicate data in the expanded dimensions.
  after_tile = tf.tile(expanded_x, [1, 1, 2, 1, 2, 1])
  return tf.reshape(after_tile,
                    [-1, image_height * 2, image_width * 2, n_channels])

def up_block(x, out_channels, name, training=True):
  """Builds the residual blocks used in the generator.
  Args:
    x: The 4D input tensor.
    out_channels: Integer number of features in the output layer.
    name: The variable scope name for the block.
    training: Whether this block is for training or not.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    bn0 = ops.BatchNorm(name='bn_0')
    bn1 = ops.BatchNorm(name='bn_1')
    x_0 = x
    x = tf.nn.relu(bn0(x))
    x = usample(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv1')
    x = tf.nn.relu(bn1(x))
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv2')

    x_0 = usample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, training, 'snconv3')

    return x_0 + x

def dsample(x):
  """Downsamples the input volume by means of average pooling.
  Args:
    x: The 4D input tensor.
  Returns:
    An downsampled version of the input tensor.
  """
  xd = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  return xd

def down_block(x, out_channels, name, downsample=True, act=tf.nn.relu):
  """Builds the residual blocks used in the discriminator.
  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    downsample: If True, downsample the spatial size the input tensor by
                a factor of 2 on each side. If False, the spatial size of the
                input tensor is unchanged.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    input_channels = x.shape.as_list()[-1]
    x_0 = x
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv2')
    if downsample:
      x = dsample(x)
    if downsample or input_channels != out_channels:
      x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='sn_conv3')
      if downsample:
        x_0 = dsample(x_0)
    return x_0 + x


def down_optimized_block(x, out_channels, name, act=tf.nn.relu):
  """Builds optimized residual blocks for downsampling.
  Compared with block, optimized_block always downsamples the spatial resolution
  by a factor of 2 on each side.
  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    x_0 = x
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv2')
    x = dsample(x)
    x_0 = dsample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='sn_conv3')
    return x + x_0
