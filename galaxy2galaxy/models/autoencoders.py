# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoencoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.layers import latent_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import t2t_model
from tensor2tensor.models.research import autoencoders

from galaxy2galaxy.utils import registry
from galaxy2galaxy.models.autoencoders_utils import autoencoder_body

import tensorflow as tf

@registry.register_model
class ContinuousAutoencoderBasic(autoencoders.AutoencoderBasic):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderResidual(autoencoders.AutoencoderResidual):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderResidualVAE(autoencoders.AutoencoderResidualVAE):

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      layers = []
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        with tf.variable_scope("layer_%d" % i):
          x = self.make_even_size(x)
          layers.append(x)
          x = self.dropout(x)
          filters = hparams.hidden_size * 2**(i + 1)
          filters = min(filters, hparams.max_hidden_size)

          x = tf.layers.conv2d(
              x,
              filters,
              kernel,
              strides=strides,
              padding="SAME",
              activation=common_layers.belu,
              name="strided")
          y = x
          y = tf.nn.dropout(y, 1.0 - hparams.residual_dropout)
          for r in range(hparams.num_residual_layers):
            residual_filters = filters
            if r < hparams.num_residual_layers - 1:
              residual_filters = int(
                  filters * hparams.residual_filter_multiplier)
            y = residual_conv(
                y,
                residual_filters,
                residual_kernel,
                padding="SAME",
                activation=common_layers.belu,
                name="residual_%d" % r)
          x += y
          x = common_layers.layer_norm(x, name="ln")
      return x, layers

  def decoder(self, x, encoder_layers=None):
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
        j = hparams.num_hidden_layers - i - 1
        if is_training:
          nomix_p = common_layers.inverse_lin_decay(
              int(hparams.bottleneck_warmup_steps * 0.25 * 2**j)) + 0.01
          if common_layers.should_generate_summaries():
            tf.summary.scalar("nomix_p_%d" % j, nomix_p)
        filters = hparams.hidden_size * 2**j
        filters = min(filters, hparams.max_hidden_size)
        with tf.variable_scope("layer_%d" % i):
          j = hparams.num_hidden_layers - i - 1
          x = tf.layers.conv2d_transpose(
              x,
              filters,
              kernel,
              strides=strides,
              padding="SAME",
              activation=common_layers.belu,
              name="strided")
          y = x
          for r in range(hparams.num_residual_layers):
            residual_filters = filters
            if r < hparams.num_residual_layers - 1:
              residual_filters = int(
                  filters * hparams.residual_filter_multiplier)
            y = residual_conv(
                y,
                residual_filters,
                residual_kernel,
                padding="SAME",
                activation=common_layers.belu,
                name="residual_%d" % r)
          x += tf.nn.dropout(y, 1.0 - hparams.residual_dropout)
          x = common_layers.layer_norm(x, name="ln")

          if encoder_layers is not None:
            enc_x = encoder_layers[j]

            enc_shape = common_layers.shape_list(x)
            x_mix_enc = enc_x[:enc_shape[0], :enc_shape[1], :enc_shape[2], :enc_shape[3]]
            x_mix = x
            if is_training:  # Mix at the beginning of training.
              rand = tf.random_uniform(common_layers.shape_list(x_mix))
              x_mix = tf.where(tf.less(rand, nomix_p), x_mix, x_mix_enc)
            if hparams.gan_loss_factor != 0:
              x_gan = x[enc_shape[0]:, :enc_shape[1], :enc_shape[2], :]
              x = tf.concat([x_mix, x_gan], axis=0)
            else:
              x = x_mix
      return x

  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderBasicDiscrete(autoencoders.AutoencoderBasicDiscrete):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderResidualDiscrete(autoencoders.AutoencoderResidualDiscrete):
  """Discrete residual autoencoder."""
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_hparams
def continuous_autoencoder_basic():
  """Basic autoencoder model."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 5
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.dropout = 0.05
  hparams.add_hparam("max_hidden_size", 1024)
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_shared_bits", 0)
  hparams.add_hparam("bottleneck_shared_bits_start_warmup", 0)
  hparams.add_hparam("bottleneck_shared_bits_stop_warmup", 0)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("bottleneck_warmup_steps", 2000)
  hparams.add_hparam("sample_height", 32)
  hparams.add_hparam("sample_width", 32)
  hparams.add_hparam("bottleneck_l2_factor", 0.05)
  hparams.add_hparam("gumbel_temperature", 0.5)
  hparams.add_hparam("gumbel_noise_factor", 0.5)
  hparams.add_hparam("vq_temperature", 0.001)
  hparams.add_hparam("gan_loss_factor", 0.0)

  # hparams related to the PSF
  hparams.add_hparam("encode_psf", True) # Should we use the PSF at the encoder
  hparams.add_hparam("apply_psf", True)  # Should we apply the PSF at the decoder
  hparams.add_hparam("psf_convolution_pad_factor", 0.)  # Zero padding factor for convolution

  # hparams related to output apodization for Fourier purposes
  hparams.add_hparam("output_apodization", 8)  # Number of pixels at the border affected by the apodization window
  hparams.add_hparam("apodization_loss", 1.0)  # Factor to penalize non zero borders

  # hparams related to output activation
  hparams.add_hparam("output_activation", 'softplus') # either none or softplus

  # hparams related to additional regularization of the output
  hparams.add_hparam("total_variation_loss", 0.001) # Factor to apply to a loss penalizing the TV of the unconvolved image

  # hparams related to the likelihood
  hparams.add_hparam("likelihood_type", "Fourier") # Pixel or Fourier
  hparams.add_hparam("noise_rms", 0.03) # Value of noise RMS, used for diagonal likelihood
  return hparams

@registry.register_hparams
def continuous_autoencoder_residual():
  """Residual autoencoder model. This works well for images of size 32x32."""
  hparams = continuous_autoencoder_basic()
  hparams.optimizer = "Adafactor"
  hparams.clip_grad_norm = 1.0
  hparams.learning_rate_constant = 0.5
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup * rsqrt_decay"
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.max_hidden_size = 1024

  hparams.add_hparam("autoregressive_decode_steps", 0)
  hparams.add_hparam("num_residual_layers", 2)
  hparams.add_hparam("residual_kernel_height", 3)
  hparams.add_hparam("residual_kernel_width", 3)
  hparams.add_hparam("residual_filter_multiplier", 2.0)
  hparams.add_hparam("residual_dropout", 0.2)
  hparams.add_hparam("residual_use_separable_conv", int(True))

  # Weight factor for the KL term of the VAE
  hparams.add_hparam("kl_beta", 1.0)
  return hparams

@registry.register_hparams
def continuous_autoencoder_residual_128():
  """Residual autoencoder model. This works well for images of size 128x128."""
  hparams = continuous_autoencoder_basic()
  hparams.optimizer = "Adafactor"
  hparams.clip_grad_norm = 1.0
  hparams.learning_rate_constant = 0.25
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup * rsqrt_decay"
  hparams.num_hidden_layers = 7
  hparams.hidden_size = 32
  hparams.max_hidden_size = 512
  hparams.batch_size = 32
  hparams.bottleneck_bits = 64

  hparams.bottleneck_warmup_steps = 5000

  hparams.add_hparam("autoregressive_decode_steps", 0)
  hparams.add_hparam("num_residual_layers", 2)
  hparams.add_hparam("residual_kernel_height", 3)
  hparams.add_hparam("residual_kernel_width", 3)
  hparams.add_hparam("residual_filter_multiplier", 2.0)
  hparams.add_hparam("residual_dropout", 0.1)
  hparams.add_hparam("residual_use_separable_conv", int(True))

  # Weight factor for the KL term of the VAE
  hparams.add_hparam("kl_beta", 1.0)
  return hparams
