""" Spectral Norm GAN """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

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

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.layers import latent_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import t2t_model
from tensor2tensor.models import vanilla_gan
from tensor2tensor.models.research.transformer_vae import residual_conv
from tensor2tensor.layers.common_layers import lrelu

from galaxy2galaxy.utils import registry
from galaxy2galaxy.models.gan_utils import softplus_discriminator_loss, softplus_generator_loss, SperctraNormConstraint


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
class SnGAN(vanilla_gan.AbstractGAN):
  """Spectral Norm GAN"""

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
      sn_update = is_training and (not reuse)

      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 32, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv1",
                             kernel_constraint=SperctraNormConstraint(sn_update, name='u1'))
      net = lrelu(net)
      net = tf.layers.conv2d(net, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1b",
                             kernel_constraint=SperctraNormConstraint(sn_update, name='u2'))
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 64, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv2",
                             kernel_constraint=SperctraNormConstraint(sn_update, name='u3'))
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2b",
                             kernel_constraint=SperctraNormConstraint(sn_update, name='u4'))
      # [bs, h/4, w/4, 128]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (3, 3), strides=(1, 1),
                             padding="SAME", name="d_conv3",
                             kernel_constraint=SperctraNormConstraint(sn_update, name='u5'))
      net = lrelu(net)
      net = tf.layers.conv2d(net, 256, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv3b",
                             kernel_constraint=SperctraNormConstraint(sn_update, name='u6'))
      # [bs, h/8, w/8, 256]
      net = lrelu(net)
      net = tf.layers.flatten(net)
      net = tf.layers.dense(net, output_size, name="d_fc3",
                           kernel_constraint=SperctraNormConstraint(sn_update, name='u7'))  # [bs, 1024]
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

      net = tf.layers.conv2d_transpose(net, 128, 4, strides=2,
                                       padding='SAME', use_bias=False, name='conv1') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn1")
      net = lrelu(net)

      # Size [12, 12, 128]
      net = tf.layers.conv2d_transpose(net, 128, 4, strides=2,
                                       padding='SAME', use_bias=False, name='conv2') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn2")
      net = lrelu(net)

      # Size [24, 24, 128]
      net = tf.layers.conv2d_transpose(net, 128, 4, strides=2,
                                       padding='SAME', use_bias=False, name='conv3') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn3")
      net = lrelu(net)

      # Size [48, 48, 128]
      net = tf.layers.conv2d_transpose(net, 128, 4, strides=2,
                                       padding='SAME', use_bias=False, name='conv4') # output_size 16x16
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="conv_bn4")
      net = lrelu(net)

      # Final convolutionn to [96, 96, 3]
      net = tf.layers.conv2d(net, 3, (3,3), padding='SAME', name='output_conv')

      out = tf.nn.sigmoid(net)
      return common_layers.convert_real_to_rgb(out)

  def body(self, features):
    """Body of the model.

    Args:
      features: a dictionary with the tensors.

    Returns:
      A pair (predictions, losses) where predictions is the generated image
      and losses is a dictionary of losses (that get added for the final loss).
    """
    features["targets"] = features["inputs"]
    is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

    # Input images.
    real_data = tf.to_float(features["targets_raw"])
    out_shape = common_layers.shape_list(real_data)[1:4]

    # Noise vector.
    generator_inputs = tf.random_uniform([self.hparams.batch_size,
                                          self.hparams.bottleneck_bits],
                                          minval=-1, maxval=1, name="z")

    # Manual gan_model creation
    generated_images = vanilla_gan.reverse_gradient(self.generator(generator_inputs, is_training, out_shape))

    tf.logging.info("Shape of generated images", generated_images.shape )
    discriminator_gen_outputs = self.discriminator(common_layers.convert_rgb_to_symmetric_real(generated_images), is_training)
    discriminator_real_outputs = self.discriminator(common_layers.convert_rgb_to_symmetric_real(real_data), is_training, reuse=True)

    gan_model = tfgan.GANModel(
        generator_inputs, generated_images,
        None, None, None,
        real_data, discriminator_real_outputs, discriminator_gen_outputs,
        None, None, None)

    losses = tfgan.gan_loss(gan_model,
                    generator_loss_fn=softplus_generator_loss,
                    discriminator_loss_fn=softplus_discriminator_loss)

    losses = {'training': losses.generator_loss + losses.discriminator_loss,
              'gen_loss': losses.generator_loss,
              'disc_loss': losses.discriminator_loss}

    summary_g_image = tf.reshape(
        generated_images[0, :], [1] + common_layers.shape_list(real_data)[1:])

    tf.summary.image("generated", pack_images(generated_images, 4, 4), max_outputs=1)
    tf.summary.image("real", pack_images(real_data, 4, 4), max_outputs=1)

    if is_training:  # Returns an dummy output and the losses dictionary.
      return tf.zeros_like(real_data), losses
    return tf.reshape(generated_images, tf.shape(real_data)), losses

@registry.register_hparams
def sn_gan():
  """Basic parameters for a spectral norm gan."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 128
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-6
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.add_hparam("bottleneck_bits", 128)
  return hparams
