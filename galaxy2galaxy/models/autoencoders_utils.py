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

"""Utility functions for autoencoders."""

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

import tensorflow as tf

def autoencoder_body(self, features):
  """ Customized body function for autoencoders acting on continuous images.
  This is based on tensor2tensor.models.research.AutoencoderBasic.body
  and should be compatible with most derived classes.
  """
  hparams = self.hparams
  is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
  vocab_size = self._problem_hparams.vocab_size["targets"]
  if hasattr(self._hparams, "vocab_divisor"):
    vocab_size += (-vocab_size) % self._hparams.vocab_divisor
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
    x = self.embed(x)
    target_codes = x
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
    if hparams.gan_loss_factor != 0.0:
      # Add a purely sampled batch on which we'll compute the GAN loss.
      g = self.unbottleneck(
          self.sample(shape=b_shape),
          common_layers.shape_list(x)[-1],
          reuse=True)
      x = tf.concat([x, g], axis=0)
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
    if hparams.use_vq_loss:
      (reconstr, _, _, _, _) = discretization.vq_loss(res, labels, vocab_size)
    else:
      reconstr = tf.layers.dense(res, vocab_size, name="autoencoder_final")
    return reconstr, {"bottleneck_loss": 0.0}

  if hparams.gan_loss_factor != 0.0:
    res, res_gan = tf.split(res, 2, axis=0)

  # Losses.
  losses = {
      "bottleneck_extra": b_loss,
      "bottleneck_l2": hparams.bottleneck_l2_factor * xb_loss
  }

  if hparams.use_vq_loss:
    vq_temperature = hparams.vq_temperature / common_layers.inverse_exp_decay(
        hparams.gan_codes_warmup_steps * 1.2,
        min_value=hparams.vq_temperature * 2)
    if hparams.mode != tf.estimator.ModeKeys.TRAIN:
      vq_temperature = None
    with tf.variable_scope("vq_loss"):
      (reconstr, _, target_codes, code_loss,
       targets_loss) = discretization.vq_loss(
           res, labels, vocab_size, temperature=vq_temperature)
    losses["code_loss"] = code_loss * hparams.code_loss_factor
    losses["training"] = targets_loss
  else:
    reconstr = tf.layers.dense(res, self.num_channels, name="autoencoder_final")
    reconstr = tf.reshape(reconstr, labels_shape)

    pz = tf.reduce_sum(tf.abs(values - targets)**2, axis=[-1, -2, -3])
    targets_loss = tf.reduce_mean(pz)
    losses["training"] = targets_loss

  # GAN losses.
  if hparams.gan_loss_factor != 0.0:
    update_means_factor = common_layers.inverse_exp_decay(
        hparams.gan_codes_warmup_steps, min_value=0.0001)
    if hparams.use_vq_loss:
      with tf.variable_scope("vq_loss", reuse=True):
        update_means = tf.less(tf.random_uniform([]), update_means_factor)
        reconstr_gan, gan_codes, _, code_loss_gan, _ = discretization.vq_loss(
            res_gan,
            labels,
            vocab_size,
            do_update=update_means,
            temperature=vq_temperature)
        reconstr_gan_nonoise = reconstr_gan
        code_loss_gan *= hparams.code_loss_factor * update_means_factor
        losses["code_loss_gan"] = code_loss_gan
    else:
      reconstr_gan = tf.layers.dense(
          res_gan, vocab_size, name="autoencoder_final", reuse=True)
      reconstr_gan_nonoise = reconstr_gan
      reconstr_gan = self.gumbel_sample(reconstr_gan)
      # Embed to codes.
      gan_codes = self.embed(reconstr_gan)

  # Add GAN loss if requested.
  gan_loss = 0.0
  if hparams.gan_loss_factor != 0.0:
    self.image_summary("gan", reconstr_gan_nonoise)

    def discriminate(x):
      """Run a dioscriminator depending on the hparams."""
      if hparams.discriminator == "default":
        return common_layers.deep_discriminator(
            x, hparams.discriminator_batchnorm, is_training)
      elif hparams.discriminator == "patched":
        return common_layers.patch_discriminator(x)
      elif hparams.discriminator == "single":
        return common_layers.single_discriminator(
            x,
            hparams.discriminator_size,
            hparams.discriminator_kernel_size,
            hparams.discriminator_strides,
            pure_mean=hparams.discriminator_pure_mean)
      elif hparams.discriminator == "double":
        return common_layers.double_discriminator(
            x,
            hparams.discriminator_size,
            hparams.discriminator_kernel_size,
            hparams.discriminator_strides,
            pure_mean=hparams.discriminator_pure_mean)
      else:
        raise Exception("Unknown discriminator %s" % hparams.discriminator)

    tc_shape = common_layers.shape_list(target_codes)
    if len(tc_shape) > 4:
      target_codes = tf.reshape(target_codes,
                                tc_shape[:-2] + [tc_shape[-1] * tc_shape[-2]])
      gan_codes = tf.reshape(gan_codes,
                             tc_shape[:-2] + [tc_shape[-1] * tc_shape[-2]])
    gan_lr = common_layers.inverse_exp_decay(
        hparams.gan_codes_warmup_steps * 1.5)
    rev_grad_gan_codes = reverse_gradient(gan_codes, lr=gan_lr)
    gan_loss = common_layers.sliced_gan_loss(
        target_codes,
        rev_grad_gan_codes,
        discriminate,
        self.hparams.num_sliced_vecs,
        do_tanh=hparams.sliced_do_tanh)
    gan_loss *= hparams.gan_loss_factor * update_means_factor
    losses["gan_loss"] = -gan_loss

  self.image_summary("ae", reconstr)

  logits = tf.reshape(reconstr, labels_shape + [vocab_size])
  return logits, losses
