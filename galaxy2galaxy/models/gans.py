
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.estimator.python.gan_estimator_impl import _make_prediction_gan_model, _summary_type_map, _get_estimator_spec
from tensorflow.contrib.gan.python.losses.python.tuple_losses_impl import _args_to_gan_model
from tensorflow.python.summary import summary
from tensorflow.python.ops.losses import util
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.contrib.framework.python.ops import variables as variable_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.util import tf_inspect as inspect

from tensor2tensor.utils import t2t_model
from tensor2tensor.models import vanilla_gan
from tensor2tensor.layers.common_layers import lrelu
from tensor2tensor.utils import hparams_lib
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_hparams

from galaxy2galaxy.utils import registry
from galaxy2galaxy.models.gan_utils import softplus_discriminator_loss, softplus_generator_loss, SpectralNormConstraint

from galaxy2galaxy.models.gan_utils import generator, discriminator

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


@registry.register_model
class WGAN(vanilla_gan.SlicedGan):
  """ WGAN-GP based on tfgan estimator API
  """

  def generator(self, z, is_training, out_shape):
    hparams = self.hparams
    height, width, c_dim = out_shape
    depth = hparams.hidden_size
    num_layers = int(log(height, 2)) - 1
    base_shape = height // 2**(num_layers)

    with tf.variable_scope("generator"):
      current_depth = depth * 2 ** (num_layers - 1)
      net = tf.layers.dense(z, base_shape*base_shape*current_depth, name="dense_1")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          name="dense_bn1")
      net = tf.nn.leaky_relu(net)
      net = tf.reshape(net, [-1, base_shape, base_shape, current_depth])

      for i in range(1, num_layers):
        current_depth = depth * 2 ** (num_layers - i)
        net = tf.layers.conv2d_transpose(net, current_depth, 4, strides=2,
                                       padding='SAME', use_bias=False, name='conv%d'%i) # output_size 16x16
        net = tf.layers.batch_normalization(net, training=is_training,
                                            name="conv_bn%d"%i)
        net = tf.nn.leaky_rely(net)

      net = tf.layers.conv2d_transpose(net, depth, 4, strides=2,
                                       padding='SAME', name='conv')
      out = tf.layers.conv2d(net, c_dim, 1, activation=tf.nn.tanh)
      return out

  def discriminator(self, x, is_training, reuse=False,
                    output_size=1):
    hparams = self.hparams
    depth = hparams.hidden_size

    with tf.variable_scope(
        "discriminator", reuse=reuse):
      batch_size, height, width = common_layers.shape_list(x)[:3]

      for i in xrange(int(log(height, 2))):
        current_depth = depth * 2**i
        net = tf.layers.conv2d(x, current_depth, 4, strides=2,
                             padding="SAME", name="d_conv%d"%i)
        net = tf.nn.leaky_relu(net)
        if hparams.discriminator_batchnorm:
          net = tf.layers.batch_normalization(net, training=is_training,
                                              name="c_bn%d"%i)
      net = tf.layers.flatten(net)
      net = tf.layers.dense(net, output_size, name="d_fcn", activation=None)  # [bs, 1024]
      return net

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

    real_data = common_layers.convert_rgb_to_real(features['inputs'])  # rename inputs for clarity
    generator_inputs = tf.random_uniform([self.hparams.batch_size,
                                          self.hparams.bottleneck_bits],
                                          minval=-1, maxval=1, name="z")

    # rename inputs for clarity
    out_shape = common_layers.shape_list(real_data)[1:4]

    if mode == model_fn_lib.ModeKeys.PREDICT:
      if real_data is not None:
        raise ValueError('`labels` must be `None` when mode is `predict`. '
                           'Instead, found %s' % real_data)
      gan_model = _make_prediction_gan_model(generator_inputs,
                                                   partial(self.generator, is_training=is_training, out_shape=out_shape),
                                                   'Generator')
    # Here should be where we export the model as tf hub

    else:  # model_fn_lib.ModeKeys.TRAIN or model_fn_lib.ModeKeys.EVAL
           # Manual gan_model creation
      with tf.variable_scope('Generator') as gen_scope:
        generated_images = self.generator(generator_inputs, is_training=is_training, out_shape=out_shape)

      with tf.variable_scope('Discriminator') as dis_scope:
        discriminator_gen_outputs = self.discriminator(generated_images, is_training=is_training, output_size=1)

      with tf.variable_scope(dis_scope, reuse=True):
        discriminator_real_outputs = self.discriminator(real_data, is_training=is_training, output_size=1)

      tf.summary.image("generated", pack_images(generated_images, 4, 4), max_outputs=1)
      tf.summary.image("real", pack_images(real_data, 4, 4), max_outputs=1)

      generator_variables = variable_lib.get_trainable_variables(gen_scope)
      discriminator_variables = variable_lib.get_trainable_variables(dis_scope)

      gan_model = tfgan.GANModel(
                generator_inputs,
                generated_images,
                generator_variables,
                gen_scope,
                self.generator,
                real_data,
                discriminator_real_outputs,
                discriminator_gen_outputs,
                discriminator_variables,
                dis_scope,
                self.discriminator)

      opt_gen = tf.train.AdamOptimizer(hparams.learning_rate)
      opt_disc = tf.train.AdamOptimizer(hparams.learning_rate)

      loss = tfgan.gan_loss(gan_model)

    # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
    # metrics, and optimizers (if required).
    return _get_estimator_spec(
      mode, gan_model, loss.generator_loss, loss.discriminator_loss,
      None, opt_gen, opt_disc, None, True)


@registry.register_model
class SpectralNormGan(SlicedGanLarge):
  """ SN-GAN based on tfgan estimator API
  """

  def discriminator(self, x, is_training, reuse=False,
                    output_size=1):
    hparams = self.hparams
    depth = hparams.hidden_size
	do_update = is_training and (not reuse)
    with tf.variable_scope(
        "discriminator", reuse=reuse):
      batch_size, height, width = common_layers.shape_list(x)[:3]

      for i in xrange(int(log(height, 2))):
        current_depth = depth * 2**i
        net = tf.layers.conv2d(x, current_depth, 4, strides=2,
                             padding="SAME", name="d_conv%d"%i,
                             kernel_constraint=SpectralNormConstraint(update=do_update,
                                                                       name='sn%d'%i))
        net = tf.nn.leaky_relu(net)
      net = tf.layers.flatten(net)
      net = tf.layers.dense(net, output_size, name="d_fcn", activation=None,
                             kernel_constraint=SpectralNormConstraint(update=do_update,
                                                                       name='fn_final'))
      return net

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

    real_data = common_layers.convert_rgb_to_real(features['inputs'])  # rename inputs for clarity
    generator_inputs = tf.random_uniform([self.hparams.batch_size,
                                          self.hparams.bottleneck_bits],
                                          minval=-1, maxval=1, name="z")

    # rename inputs for clarity
    out_shape = common_layers.shape_list(real_data)[1:4]

    if mode == model_fn_lib.ModeKeys.PREDICT:
      if real_data is not None:
        raise ValueError('`labels` must be `None` when mode is `predict`. '
                           'Instead, found %s' % real_data)
      gan_model = _make_prediction_gan_model(generator_inputs,
                                                   partial(self.generator, is_training=is_training, out_shape=out_shape),
                                                   'Generator')
    # Here should be where we export the model as tf hub

    else:  # model_fn_lib.ModeKeys.TRAIN or model_fn_lib.ModeKeys.EVAL
            # Manual gan_model creation
      with tf.variable_scope('Generator') as gen_scope:
        generated_images = self.generator(generator_inputs, is_training=is_training, out_shape=out_shape)

      with tf.variable_scope('Discriminator') as dis_scope:
        discriminator_gen_outputs = self.discriminator(generated_images, is_training=is_training)

      with tf.variable_scope(dis_scope, reuse=True):
        discriminator_real_outputs = self.discriminator(real_data, is_training=is_training, reuse=True)

      tf.summary.image("generated", pack_images(generated_images, 4, 4), max_outputs=1)
      tf.summary.image("real", pack_images(real_data, 4, 4), max_outputs=1)

      generator_variables = variable_lib.get_trainable_variables(gen_scope)
      discriminator_variables = variable_lib.get_trainable_variables(dis_scope)

      gan_model = tfgan.GANModel(
                generator_inputs,
                generated_images,
                generator_variables,
                gen_scope,
                self.generator,
                real_data,
                discriminator_real_outputs,
                discriminator_gen_outputs,
                discriminator_variables,
                dis_scope,
                self.discriminator)

      opt_gen = tf.train.AdamOptimizer(hparams.learning_rate)
      opt_disc = tf.train.AdamOptimizer(hparams.learning_rate)

    # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
    # metrics, and optimizers (if required).
    return _get_estimator_spec(
      mode, gan_model, softplus_generator_loss, softplus_discriminator_loss,
      None, opt_gen, opt_disc, None, True)

@registry.register_hparams
def gan_large():
  """Basic parameters for large gan."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate = 0.0001
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-6
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("discriminator_batchnorm", True)
  return hparams
