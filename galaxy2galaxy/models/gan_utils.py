""" Spectral Norm GAN """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan.python.estimator.gan_estimator import Optimizers, get_gan_model, get_train_estimator_spec, get_eval_estimator_spec, get_predict_estimator_spec
from tensorflow_gan.python import train as tfgan_train
from tensorflow.python.estimator import model_fn as model_fn_lib

from tensor2tensor.utils import t2t_model

from tensorflow_gan.examples.self_attention_estimator import ops
from .commons import pack_images

class AbstractGAN(t2t_model.T2TModel):
  """ Base class for tf-gan based models
  """

  def generator(self, code, mode, out_shape):
    raise NotImplementedError

  def discriminator(self, x, conditioning, mode):
    raise NotImplementedError

  def sample_noise(self):
    raise NotImplementedError

  def discriminator_loss_fn(self):
    raise NotImplementedError

  def generator_loss_fn(self):
    raise NotImplementedError

  @classmethod
  def estimator_model_fn(cls,
                         hparams,
                         features,
                         labels,
                         mode,
                         config=None,
                         params=None,
                         decode_hparams=None,
                         use_tpu=False):

    if mode not in [model_fn_lib.ModeKeys.TRAIN, model_fn_lib.ModeKeys.EVAL,
                  model_fn_lib.ModeKeys.PREDICT]:
      raise ValueError('Mode not recognized: %s' % mode)

    if mode is model_fn_lib.ModeKeys.TRAIN:
      is_training = True
    else:
      is_training = False

    hparams = hparams_lib.copy_hparams(hparams)

    # Instantiate model
    data_parallelism = None
    if not use_tpu and config:
      data_parallelism = config.data_parallelism
    reuse = tf.get_variable_scope().reuse

    # Instantiate model
    self = cls(
        hparams,
        mode,
        data_parallelism=data_parallelism,
        decode_hparams=decode_hparams,
        _reuse=reuse)

    real_data = features['inputs']        # rename inputs for clarity
    generator_inputs = self.sample_noise()

    optimizers = Optimizers(tf.compat.v1.train.AdamOptimizer(
          hparams.generator_lr, hparams.beta1),
          tf.compat.v1.train.AdamOptimizer(
          hparams.discriminator_lr, hparams.beta1)
          )

    # Make GANModel, which encapsulates the GAN model architectures.
    gan_model = get_gan_model(mode,
                              self.generator,
                              self.discriminator,
                              real_data,
                              generator_inputs, add_summaries=True)

    # Make GANLoss, which encapsulates the losses.
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      gan_loss = tfgan_train.gan_loss(
          gan_model,
          self.generator_loss,
          selg.discriminator_loss,
          add_summaries=True)

    # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
    # metrics, and optimizers (if required).
    if mode == tf.estimator.ModeKeys.TRAIN:
      estimator_spec = get_train_estimator_spec(gan_model, gan_loss, optimizers)
    elif mode == tf.estimator.ModeKeys.EVAL:
      estimator_spec = get_eval_estimator_spec(gan_model, gan_loss)
    else:  # tf.estimator.ModeKeys.PREDICT
      estimator_spec = get_predict_estimator_spec(gan_model)

    return estimator_spec

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
