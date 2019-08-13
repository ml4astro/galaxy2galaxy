
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

from galaxy2galaxy.utils import registry
from galaxy2galaxy.models.gan_utils import softplus_discriminator_loss, softplus_generator_loss, SperctralNormConstraint


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
class SlicedGanLarge(vanilla_gan.SlicedGan):
  """ Customized sliced gan for larger images
  """

  def discriminator(self, x, is_training, reuse=False,
                    output_size=1024):
    """Discriminator architecture with Spectral Normalization.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.
      reuse: boolean, should params be re-used.

    Returns:
      out_logit: the output logits (before sigmoid).
    """
    hparams = self.hparams
    with tf.variable_scope(
        "discriminator", reuse=reuse,
        initializer=tf.random_normal_initializer(stddev=0.02)):
      batch_size, height, width = common_layers.shape_list(x)[:3]

      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 32, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv1")
      net = lrelu(net)
      net = tf.layers.conv2d(net, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1b")
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="c_bn1")
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 64, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv2")
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2b")
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="c_bn2")
      # [bs, h/4, w/4, 128]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv3")
      net = lrelu(net)
      net = tf.layers.conv2d(net, 256, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv3b")
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="c_bn3")
      # [bs, h/8, w/8, 256]
      net = lrelu(net)
      net = tf.layers.flatten(net)
      net = tf.layers.dense(net, output_size, name="d_fc3")  # [bs, 1024]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn3")
      net = lrelu(net)
      return net

  def generator(self, z, is_training, out_shape):
    """Generator outputting image in [0, 1]."""
    hparams = self.hparams
    height, width, c_dim = out_shape
    batch_size = hparams.batch_size
    with tf.variable_scope("generator"):
      net = tf.layers.dense(z, 1024, name="g_fc1")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn1")
      net = lrelu(net)
      net = tf.layers.dense(net, 128 * (height // 16) * (width // 16),
                            name="g_fc2")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [batch_size, height // 16, width // 16, 128])
      # Size [6, 6, 128]

      net = tf.layers.conv2d_transpose(net, 128, 3, strides=2,
                                       padding='SAME', use_bias=False, name='conv1') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn1")
      net = lrelu(net)

      # Size [12, 12, 128]
      net = tf.layers.conv2d_transpose(net, 128, 3, strides=2,
                                       padding='SAME', use_bias=False, name='conv2') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn2")
      net = lrelu(net)

      # Size [24, 24, 128]
      net = tf.layers.conv2d_transpose(net, 128, 3, strides=2,
                                       padding='SAME', use_bias=False, name='conv3') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn3")
      net = lrelu(net)

      # Size [48, 48, 128]
      net = tf.layers.conv2d_transpose(net, 128, 3, strides=2,
                                       padding='SAME', use_bias=False, name='conv4') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn4")
      net = lrelu(net)

      # Final convolutionn to [96, 96, 3]
      net = tf.layers.conv2d(net, 3, (3,3), padding='SAME', name='output_conv')

      out = tf.nn.sigmoid(net)
      return out

@registry.register_model
class GanEstimator(SlicedGanLarge):
    """ GAN based on tfgan estimator API
    """

  def discriminator(self, x, is_training, reuse=False,
                    output_size=1):
    """Discriminator architecture with Spectral Normalization.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.
      reuse: boolean, should params be re-used.

    Returns:
      out_logit: the output logits (before sigmoid).
    """
    hparams = self.hparams
    with tf.variable_scope(
        "discriminator", reuse=reuse,
        initializer=tf.random_normal_initializer(stddev=0.02)):
      batch_size, height, width = common_layers.shape_list(x)[:3]
      do_update = is_training and (not reuse)

      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 32, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv1",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn1'))
      net = lrelu(net)
      net = tf.layers.conv2d(net, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1b",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn1b'))
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 64, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv2",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn2'))
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2b",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn2b'))
      # [bs, h/4, w/4, 128]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv3",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn3'))
      net = lrelu(net)
      net = tf.layers.conv2d(net, 256, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv3b",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn3b'))
      # [bs, h/8, w/8, 256]
      net = lrelu(net)
      net = tf.layers.flatten(net)
      net = tf.layers.dense(net, output_size, name="d_fc3",
                             kernel_constraint=SperctralNormConstraint(update=do_update,
                                                                       name='sn4'))  # [bs, 1024]

      net = lrelu(net)
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

    real_data =  common_layers.convert_rgb_to_real(features['inputs'])  # rename inputs for clarity
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
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 256
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-6
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("discriminator_batchnorm", True)
  hparams.add_hparam("num_sliced_vecs", 4096)
  return hparams
