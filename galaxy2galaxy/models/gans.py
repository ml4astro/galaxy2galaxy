
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow.python.summary import summary

from tensor2tensor.utils import hparams_lib
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_hparams

from galaxy2galaxy.utils import registry
from galaxy2galaxy.models.gan_utils import AbstractGAN
from galaxy2galaxy.models.gan_utils import SummaryType

from galaxy2galaxy.layers import spectral_ops as ops
from galaxy2galaxy.layers.common_layers import up_block, down_block, down_optimized_block


@registry.register_model
class SelfAttentionGan(AbstractGAN):
  """ Implementation of Self Attention GAN
  """

  def sample_noise(self):
    p = self.hparams
    shape = [p.batch_size, p.bottleneck_bits]
    z = tf.random.normal(shape, name='z0', dtype=tf.float32)
    return z

  def generator(self, code, mode):
    """Builds the generator segment of the graph, going from z -> G(z).
    Args:
    zs: Tensor representing the latent variables.
    gf_dim: The gf dimension.
    training: Whether in train mode or not. This affects things like batch
      normalization and spectral normalization.
    Returns:
    - The output layer of the generator.
    """
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    p = self.hparams
    gf_dim = p.hidden_size
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as gen_scope:
      act0 = ops.snlinear(code, gf_dim * 16 * 4 * 4, training=training, name='g_snh0')
      act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

      # pylint: disable=line-too-long
      act1 = up_block(act0, gf_dim * 16, 'g_block1', training)  # 8
      act2 = up_block(act1, gf_dim * 8, 'g_block2', training)  # 16
      act3 = up_block(act2, gf_dim * 4, 'g_block3', training)  # 32
      act3 = ops.sn_non_local_block_sim(act3, training, name='g_ops')  # 32
      act4 = up_block(act3, gf_dim * 2, 'g_block4', training)  # 64
      if p.noise_sigma >0:
        act4 = tf.concat([act4, p.noise_sigma*tf.random_normal((p.batch_size, 64, 64,3))],axis=-1)
      act5 = up_block(act4, gf_dim, 'g_block5', training)  # 128
      bn = ops.BatchNorm(name='g_bn')

      act5 = tf.nn.relu(bn(act5))
      act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, training, 'g_snconv_last')
      out = tf.nn.tanh(act6)
      return out

  def discriminator(self, image, conditioning, mode):
    """Builds the discriminator graph.
    Args:
        image: The current batch of images to classify as fake or real.
        df_dim: The df dimension.
        act: The activation function used in the discriminator.
      Returns:
        - A `Tensor` representing the logits of the discriminator.
        - A list containing all trainable varaibles defined by the model.
    """
    p = self.hparams
    df_dim = p.hidden_size
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    act=tf.nn.relu
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as dis_scope:
      h0 = down_optimized_block(image, df_dim, 'd_optimized_block1', act=act)  # 64 * 64
      h1 = down_block(h0, df_dim * 2, 'd_block2', act=act)  # 32 * 32
      h1 = ops.sn_non_local_block_sim(h1, name='d_ops')  # 32 * 32
      h2 = down_block(h1, df_dim * 4, 'd_block3', act=act)  # 16 * 16
      h3 = down_block(h2, df_dim * 8, 'd_block4', act=act)  # 8 * 8
      h4 = down_block(h3, df_dim * 16, 'd_block5', act=act)  # 4 * 4
      h5 = down_block(h4, df_dim * 16, 'd_block6', downsample=False, act=act)
      h5_act = act(h5)
      h6 = tf.reduce_sum(h5_act, [1, 2])
      output = ops.snlinear(h6, 1, name='d_sn_linear')
    return output

  def generator_loss(self,*args, **kwargs):
    return tfgan.losses.wasserstein_hinge_generator_loss(*args, **kwargs)

  def discriminator_loss(selg, *args, **kwargs):
    return tfgan.losses.wasserstein_hinge_discriminator_loss(*args, **kwargs)

@registry.register_hparams
def sagan():
  """Basic parameters for 128x128 SAGAN."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate = 0.0001
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 64
  hparams.hidden_size = 32
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-6
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("generator_lr", 0.0001)
  hparams.add_hparam("discriminator_lr", 0.0004)
  hparams.add_hparam("beta1", 0.)
  hparams.add_hparam("noise_sigma", 0.)
  hparams.add_hparam("gen_steps", 1)
  hparams.add_hparam("disc_steps", 1)
  return hparams

@registry.register_hparams
def sagan_noise():
  """Basic parameters for 128x128 SAGAN."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate = 0.0001
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 64
  hparams.hidden_size = 32
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-6
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("generator_lr", 0.0001)
  hparams.add_hparam("discriminator_lr", 0.0004)
  hparams.add_hparam("beta1", 0.)
  hparams.add_hparam("noise_sigma", 0.1)
  hparams.add_hparam("gen_steps", 1)
  hparams.add_hparam("disc_steps", 3)
  return hparams
