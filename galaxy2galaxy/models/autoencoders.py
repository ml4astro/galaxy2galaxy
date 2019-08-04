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

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.layers import latent_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import t2t_model
from tensor2tensor.models.research import autoencoders

from galaxy2galaxy.utils import registry

import tensorflow as tf

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
class ContinuousAutoencoderBasic(autoencoders.AutoencoderBasic):
  """Continuous version of the basic Autoencoder"""

  def reconstruction_loss(self, values, targets):
    hparams = self.hparams
    pz = tf.reduce_sum(tf.abs(values - targets)**2, axis=[-1, -2, -3]) / hparams.reconstruction_loss_sigma**2
    return tf.reduce_mean(pz)

  def image_summary(self, name, image_logits, max_outputs=1, rows=8, cols=8):
    """Helper for image summaries that are safe on TPU."""
    if len(image_logits.get_shape()) != 4:
      tf.logging.info("Not generating image summary, maybe not an image.")
      return
    return tf.summary.image(
        name, pack_images(image_logits, rows, cols),
        #common_layers.tpu_safe_image_summary(pack_images(tensor, rows, cols)),
        max_outputs=max_outputs)

  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    encoder_layers = None
    self.is1d = hparams.sample_width == 1
    if (hparams.mode != tf.estimator.ModeKeys.PREDICT
        or self._encode_on_predict):
      labels = features["targets_raw"]
      labels_shape = common_layers.shape_list(labels)
      # handle videos
      if len(labels.shape) == 5:
        labels = time_to_channels(labels)
      shape = common_layers.shape_list(labels)
      x = tf.expand_dims(labels, axis=-1)
      x = self.embed(x)
      target_codes = x
      print(x)
      if shape[2] == 1:
        self.is1d = True
      # Run encoder.
      x, encoder_layers = self.encoder(x)
      # Bottleneck.
      b, b_loss = self.bottleneck(x)
      xb_loss = 0.0
      b_shape = common_layers.shape_list(b)
      self._cur_bottleneck_tensor = b
      res_size = common_layers.shape_list(x)[-1]
      b = self.unbottleneck(b, res_size)
      if not is_training:
        x = b
      else:
        l = 2**hparams.num_hidden_layers
        warm_step = int(hparams.bottleneck_warmup_steps * 0.25 * l)
        nomix_p = common_layers.inverse_lin_decay(warm_step) + 0.01
        if common_layers.should_generate_summaries():
          tf.summary.scalar("nomix_p_bottleneck", nomix_p)
        rand = tf.random_uniform(common_layers.shape_list(x))
        # This is the distance between b and x. Having this as loss helps learn
        # the bottleneck function, but if we back-propagated to x it would be
        # minimized by just setting x=0 and b=0 -- so we don't want too much
        # of the influence of this, and we stop-gradient to not zero-out x.
        x_stop = tf.stop_gradient(x)
        xb_loss = tf.reduce_mean(tf.reduce_sum(
            tf.squared_difference(x_stop, b), axis=-1))
        # To prevent this loss from exploding we clip at 1, but anneal clipping.
        clip_max = 1.0 / common_layers.inverse_exp_decay(
            warm_step, min_value=0.001)
        xb_clip = tf.maximum(tf.stop_gradient(xb_loss), clip_max)
        xb_loss *= clip_max / xb_clip
        x = tf.where(tf.less(rand, nomix_p), b, x)
    else:
      if self._cur_bottleneck_tensor is None:
        b = self.sample()
      else:
        b = self._cur_bottleneck_tensor
      self._cur_bottleneck_tensor = b
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x, encoder_layers)

    # Cut to the right size and mix before returning.
    res = x
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      res = x[:, :shape[1], :shape[2], :]

    # Final dense layer.
    res = tf.layers.dense(
        res, self.num_channels * hparams.hidden_size, name="res_dense")

    output_shape = common_layers.shape_list(res)[:-1] + [
        self.num_channels, self.hparams.hidden_size
    ]
    res = tf.reshape(res, output_shape)

    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      reconstr = tf.layers.dense(res, self.num_channels, name="autoencoder_final")
      return reconstr, {"bottleneck_loss": 0.0}

    # Losses.
    losses = {
        "bottleneck_extra": b_loss,
        "bottleneck_l2": hparams.bottleneck_l2_factor * xb_loss
    }

    reconstr = tf.layers.dense(res, self.num_channels, name="autoencoder_final")
    reconstr = tf.reshape(reconstr, labels_shape)

    targets_loss = self.reconstruction_loss(reconstr, labels)
    losses["training"] = targets_loss

    self.image_summary("inputs", labels)
    self.image_summary("ae", reconstr)
    return reconstr, losses


@registry.register_model
class ContinuousAutoencoderResidual(ContinuousAutoencoderBasic):
  """Residual autoencoder."""

  def dropout(self, x):
    is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN
    hparams = self.hparams
    if hparams.dropout <= 0.0 or not is_training:
      return x
    warm_step = hparams.bottleneck_warmup_steps * 2**hparams.num_hidden_layers
    dropout = common_layers.inverse_lin_decay(warm_step // 2) * hparams.dropout
    return common_layers.dropout_with_broadcast_dims(
        x, 1.0 - dropout, broadcast_dims=[-1])

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
          x = common_attention.add_timing_signal_nd(x)
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
          x = common_attention.add_timing_signal_nd(x)
          if encoder_layers is not None:
            enc_x = encoder_layers[j]
            enc_shape = common_layers.shape_list(enc_x)
            x_mix = x[:enc_shape[0], :enc_shape[1], :enc_shape[2], :]
            if is_training:  # Mix at the beginning of training.
              rand = tf.random_uniform(common_layers.shape_list(x_mix))
              x_mix = tf.where(tf.less(rand, nomix_p), x_mix, enc_x)
            x = x_mix
      return x


@registry.register_model
class ContinuousAutoencoderResidualVAE(ContinuousAutoencoderResidual):
  """Residual VAE autoencoder."""

  def bottleneck(self, x):
    hparams = self.hparams
    z_size = hparams.bottleneck_bits
    x_shape = common_layers.shape_list(x)
    with tf.variable_scope("vae"):
      mu = tf.layers.dense(x, z_size, name="mu")
      if hparams.mode != tf.estimator.ModeKeys.TRAIN:
        return mu, 0.0  # No sampling or kl loss on eval.
      log_sigma = tf.layers.dense(x, z_size, name="log_sigma")
      epsilon = tf.random_normal(x_shape[:-1] + [z_size])
      z = mu + tf.exp(log_sigma / 2) * epsilon
      kl = 0.5 * tf.reduce_mean(
          tf.expm1(log_sigma) + tf.square(mu) - log_sigma, axis=-1)
      free_bits = z_size // 4
      kl_loss = tf.reduce_mean(tf.maximum(kl - free_bits, 0.0))
    return z, kl_loss * hparams.kl_beta

  def sample(self, features=None, shape=None):
    del features
    hparams = self.hparams
    div_x = 2**hparams.num_hidden_layers
    div_y = 1 if self.is1d else 2**hparams.num_hidden_layers
    size = [
        hparams.batch_size, hparams.sample_height // div_x,
        hparams.sample_width // div_y, hparams.bottleneck_bits
    ]
    size = size if shape is None else shape
    return tf.random_normal(size)


@registry.register_model
class ContinuousAutoencoderBasicDiscrete(ContinuousAutoencoderBasic):
  """Discrete autoencoder."""

  def bottleneck(self, x):
    hparams = self.hparams
    x = tf.tanh(tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck"))
    d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      noise = tf.random_uniform(common_layers.shape_list(x))
      noise = 2.0 * tf.to_float(tf.less(hparams.bottleneck_noise, noise)) - 1.0
      d *= noise
    x = common_layers.mix(d, x, hparams.discretize_warmup_steps,
                          hparams.mode == tf.estimator.ModeKeys.TRAIN)
    return x, 0.0

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [
        hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
        hp.bottleneck_bits
    ]
    size = size if shape is None else shape
    rand = tf.random_uniform(size)
    return 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0


@registry.register_model
class ContinuousAutoencoderResidualDiscrete(ContinuousAutoencoderResidual):
  """Discrete residual autoencoder."""

  def variance_loss(self, b):
    part = tf.random_uniform(common_layers.shape_list(b))
    selection = tf.to_float(tf.less(part, tf.random_uniform([])))
    selection_size = tf.reduce_sum(selection)
    part_avg = tf.abs(tf.reduce_sum(b * selection)) / (selection_size + 1)
    return part_avg

  def bottleneck(self, x, bottleneck_bits=None):  # pylint: disable=arguments-differ
    if bottleneck_bits is not None:
      old_bottleneck_bits = self.hparams.bottleneck_bits
      self.hparams.bottleneck_bits = bottleneck_bits
    res, loss = discretization.parametrized_bottleneck(x, self.hparams)
    if bottleneck_bits is not None:
      self.hparams.bottleneck_bits = old_bottleneck_bits
    return res, loss

  def unbottleneck(self, x, res_size, reuse=None):
    with tf.variable_scope("unbottleneck", reuse=reuse):
      return discretization.parametrized_unbottleneck(x, res_size, self.hparams)

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [
        hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
        hp.bottleneck_bits
    ]
    size = size if shape is None else shape
    rand = tf.random_uniform(size)
    res = 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0
    # If you want to set some first bits to a fixed value, do this:
    # fixed = tf.zeros_like(rand) - 1.0
    # nbits = 3
    # res = tf.concat([fixed[:, :, :, :nbits], res[:, :, :, nbits:]], axis=-1)
    return res


@registry.register_model
class ContinuousAutoencoderOrderedDiscrete(ContinuousAutoencoderResidualDiscrete):
  """Ordered discrete autoencoder."""

  def bottleneck(self, x):  # pylint: disable=arguments-differ
    hparams = self.hparams
    if hparams.unordered:
      return super(AutoencoderOrderedDiscrete, self).bottleneck(x)
    noise = hparams.bottleneck_noise
    hparams.bottleneck_noise = 0.0  # We'll add noise below.
    x, loss = discretization.parametrized_bottleneck(x, hparams)
    hparams.bottleneck_noise = noise
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      # We want a number p such that p^bottleneck_bits = 1 - noise.
      # So log(p) * bottleneck_bits = log(noise)
      log_p = tf.log1p(-float(noise) / 2) / float(hparams.bottleneck_bits)
      # Probabilities of flipping are p, p^2, p^3, ..., p^bottleneck_bits.
      noise_mask = 1.0 - tf.exp(tf.cumsum(tf.zeros_like(x) + log_p, axis=-1))
      # Having the no-noise mask, we can make noise just uniformly at random.
      ordered_noise = tf.random_uniform(tf.shape(x))
      # We want our noise to be 1s at the start and random {-1, 1} bits later.
      ordered_noise = tf.to_float(tf.less(noise_mask, ordered_noise))
      # Now we flip the bits of x on the noisy positions (ordered and normal).
      x *= 2.0 * ordered_noise - 1
    return x, loss


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
  hparams.add_hparam("num_sliced_vecs", 20000)
  hparams.add_hparam("sliced_do_tanh", int(True))
  hparams.add_hparam("code_loss_factor", 1.0)
  hparams.add_hparam("bottleneck_l2_factor", 0.05)
  hparams.add_hparam("reconstruction_loss_sigma", 0.03)
  return hparams

@registry.register_hparams
def continuous_autoencoder_residual():
  """Residual autoencoder model."""
  hparams = continuous_autoencoder_basic()
  hparams.optimizer = "Adafactor"
  hparams.clip_grad_norm = 1.0
  hparams.learning_rate_constant = 0.5
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup * rsqrt_decay"
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.max_hidden_size = 1024
  hparams.add_hparam("num_residual_layers", 2)
  hparams.add_hparam("residual_kernel_height", 3)
  hparams.add_hparam("residual_kernel_width", 3)
  hparams.add_hparam("residual_filter_multiplier", 2.0)
  hparams.add_hparam("residual_dropout", 0.2)
  hparams.add_hparam("residual_use_separable_conv", int(True))
  hparams.add_hparam("kl_beta", 1.0)
  return hparams

@registry.register_hparams
def continuous_autoencoder_basic_discrete():
  """Basic autoencoder model."""
  hparams = continuous_autoencoder_basic()
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.bottleneck_bits = 1024
  hparams.bottleneck_noise = 0.1
  hparams.add_hparam("discretize_warmup_steps", 16000)
  return hparams

@registry.register_hparams
def continuous_autoencoder_residual_discrete():
  """Residual discrete autoencoder model."""
  hparams = continuous_autoencoder_residual()
  hparams.bottleneck_bits = 1024
  hparams.bottleneck_noise = 0.05
  hparams.add_hparam("discretize_warmup_steps", 16000)
  hparams.add_hparam("bottleneck_kind", "tanh_discrete")
  hparams.add_hparam("isemhash_noise_dev", 0.5)
  hparams.add_hparam("isemhash_mix_prob", 0.5)
  hparams.add_hparam("isemhash_filter_size_multiplier", 2.0)
  hparams.add_hparam("vq_beta", 0.25)
  hparams.add_hparam("vq_decay", 0.999)
  hparams.add_hparam("vq_epsilon", 1e-5)
  return hparams

@registry.register_hparams
def continuous_autoencoder_residual_discrete_big():
  """Residual discrete autoencoder model, big version."""
  hparams = continuous_autoencoder_residual_discrete()
  hparams.hidden_size = 128
  hparams.max_hidden_size = 4096
  hparams.bottleneck_noise = 0.1
  hparams.residual_dropout = 0.4
  return hparams

@registry.register_hparams
def continuous_autoencoder_ordered_discrete():
  """Ordered discrete autoencoder model."""
  hparams = continuous_autoencoder_residual_discrete()
  hparams.bottleneck_noise = 0.05  # Use 0.8 for ordered.
  hparams.add_hparam("unordered", True)
  return hparams

@registry.register_hparams
def continuous_autoencoder_ordered_discrete_vq():
  """Ordered discrete autoencoder model with VQ bottleneck."""
  hparams = continuous_autoencoder_ordered_discrete()
  hparams.bottleneck_kind = "vq"
  hparams.bottleneck_bits = 16
  return hparams

@registry.register_hparams
def continuous_autoencoder_discrete_cifar():
  """Discrete autoencoder model for compressing cifar."""
  hparams = continuous_autoencoder_ordered_discrete()
  hparams.bottleneck_noise = 0.0
  hparams.bottleneck_bits = 90
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.num_residual_layers = 4
  hparams.batch_size = 32
  hparams.learning_rate_constant = 1.0
  return hparams
