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
import tensorflow_hub as hub

from galaxy2galaxy.layers.image_utils import pack_images

def loglikelihood_fn(xin, yin, features, hparams):
  if hparams.likelihood_type == 'Fourier':
    x = tf.spectral.rfft2d(xin[...,0]) / tf.complex(tf.sqrt(tf.exp(features['ps'])),0.)
    y = tf.spectral.rfft2d(yin[...,0]) / tf.complex(tf.sqrt(tf.exp(features['ps'])),0.)

    # Compute FFT normalization factor
    size = xin.get_shape().as_list()[1]
    pz = tf.reduce_sum(tf.abs(x - y)**2, axis=[-1, -2]) / size**2
    return -pz
  elif hparams.likelihood_type == 'Pixel':
    # TODO: include per example noise std
    pz = tf.reduce_sum(tf.abs(xin[:,:,:,0] - y)**2, axis=[-1, -2]) / hparams.noise_rms**2
    return -pz
  else:
    raise NotImplementedError

def image_summary(name, image_logits, max_outputs=1, rows=8, cols=8):
  """Helper for image summaries that are safe on TPU."""
  if len(image_logits.get_shape()) != 4:
    tf.logging.info("Not generating image summary, maybe not an image.")
    return
  return tf.summary.image(name, pack_images(image_logits, rows, cols),
      max_outputs=max_outputs)

def autoencoder_body(self, features):
  """ Customized body function for autoencoders acting on continuous images.
  This is based on tensor2tensor.models.research.AutoencoderBasic.body
  and should be compatible with most derived classes.

  The original autoencoder class relies on embedding the channels to a discrete
  vocabulary and defines the loss on that vocab. It's cool and all, but here we
  prefer expressing the reconstruction loss as an actual continuous likelihood
  function.
  """
  hparams = self.hparams
  is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN

  if hparams.mode == tf.estimator.ModeKeys.PREDICT:
    # In predict mode, we also define TensorFlow Hub modules for all pieces of
    # the autoencoder
    input_shape =  [None, ] + common_layers.shape_list(features["inputs"])[1:]
    # First build encoder spec
    def make_model_spec():
      input_layer = tf.placeholder(tf.float32, shape=input_shape)
      x = self.embed(tf.expand_dims(input_layer, -1))
      x, encoder_layers = self.encoder(x)
      b, b_loss = self.bottleneck(x)
      hub.add_signature(inputs=input_layer, outputs=b)

    def make_model_spec_psf():
      input_layer = tf.placeholder(tf.float32, shape=input_shape)
      psf_layer = tf.placeholder(tf.float32, shape=input_shape)
      x = self.embed(tf.expand_dims(input_layer, -1))
      x, encoder_layers = self.encoder(x)
      net_psf = tf.layers.conv2d(psf_layer, hparams.hidden_size, 5,
      padding='same', name="psf_embed_1")
      net_psf = common_layers.layer_norm(net_psf, name="psf_norm")
      kernel, strides = self._get_kernel_and_strides()
      for i in range(hparams.num_hidden_layers):
        net_psf = self.make_even_size(net_psf)
        net_psf = tf.layers.conv2d( net_psf,
                  hparams.hidden_size * 2**(i + 1),
                  kernel,
                  strides=strides,
                  padding="SAME",
                  activation=common_layers.belu,
                  name="conv_psf_%d" % i)
        net_psf = common_layers.layer_norm(net_psf, name="psf_ln_%d" % i)
      b, b_loss = self.bottleneck(tf.concat([x, net_psf], axis=-1))
      hub.add_signature(inputs={'input':input_layer, 'psf':psf_layer}, outputs=b)

    spec = hub.create_module_spec(make_model_spec_psf if hparams.encode_psf else make_model_spec, drop_collections=['checkpoints'])
    encoder = hub.Module(spec, name="encoder_module")
    hub.register_module_for_export(encoder, "encoder")

    if hparams.encode_psf:
      code = encoder({'input':features["inputs"], 'psf': features['psf']})
    else:
      code = encoder(features["inputs"])
    b_shape = [None, ] + common_layers.shape_list(code)[1:]
    res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
    res_size = min(res_size, hparams.max_hidden_size)

    # Second build decoder spec
    def make_model_spec():
      input_layer = tf.placeholder(tf.float32, shape=b_shape)
      x = self.unbottleneck(input_layer, res_size)
      x = self.decoder(x, None)
      reconstr = tf.layers.dense(x, self.num_channels, name="autoencoder_final")
      hub.add_signature(inputs=input_layer, outputs=reconstr)
    spec = hub.create_module_spec(make_model_spec, drop_collections=['checkpoints'])
    decoder = hub.Module(spec, name="decoder_module")
    hub.register_module_for_export(decoder, "decoder")

    reconstr = decoder(code)
    return reconstr , {"bottleneck_loss": 0.0}

  encoder_layers = None
  self.is1d = hparams.sample_width == 1
  if (hparams.mode != tf.estimator.ModeKeys.PREDICT
      or self._encode_on_predict):
    labels = features["targets_raw"]
    labels_shape = common_layers.shape_list(labels)

    shape = common_layers.shape_list(labels)
    with tf.variable_scope('encoder_module'):
      x = self.embed(tf.expand_dims(labels, -1))

    if shape[2] == 1:
      self.is1d = True

    # Run encoder.
    with tf.variable_scope('encoder_module'):
      x, encoder_layers = self.encoder(x)

    # Bottleneck.
    with tf.variable_scope('encoder_module'):
      # If we have access to the PSF, we add this information to the encoder
      if hparams.encode_psf and 'psf' in features:
        net_psf = tf.layers.conv2d(features['psf'], hparams.hidden_size, 5,
                                   padding='same', name="psf_embed_1")
        net_psf = common_layers.layer_norm(net_psf, name="psf_norm")
        kernel, strides = self._get_kernel_and_strides()
        for i in range(hparams.num_hidden_layers):
          net_psf = self.make_even_size(net_psf)
          net_psf = tf.layers.conv2d(
              net_psf,
              hparams.hidden_size * 2**(i + 1),
              kernel,
              strides=strides,
              padding="SAME",
              activation=common_layers.belu,
              name="conv_psf_%d" % i)
          net_psf = common_layers.layer_norm(net_psf, name="psf_ln_%d" % i)
          # net_psf = tf.layers.average_pooling2d(net_psf,
          #                                       pool_size=common_layers.shape_list(net_psf)[1:3],
          #                                       strides=common_layers.shape_list(net_psf)[1:3])
        b, b_loss = self.bottleneck(tf.concat([x, net_psf], axis=-1))
      else:
        b, b_loss = self.bottleneck(x)

    xb_loss = 0.0
    b_shape = common_layers.shape_list(b)
    self._cur_bottleneck_tensor = b
    res_size = common_layers.shape_list(x)[-1]
    with tf.variable_scope('decoder_module'):
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

    with tf.variable_scope('decoder_module'):
      x = self.unbottleneck(b, res_size)

  # Run decoder.
  with tf.variable_scope('decoder_module'):
    x = self.decoder(x, encoder_layers)

  # Cut to the right size and mix before returning.
  res = x
  if hparams.mode != tf.estimator.ModeKeys.PREDICT:
    res = x[:, :shape[1], :shape[2], :]

  with tf.variable_scope('decoder_module'):
    reconstr = tf.layers.dense(res, self.num_channels, name="autoencoder_final")

  # Apply channel-wise convolution with the PSF if requested
  # TODO: Handle multiple bands
  if hparams.apply_psf and 'psf' in features:
    if self.num_channels > 1:
      raise NotImplementedError
    rec_padded = tf.pad(reconstr[:,:,:,0], [[0,0],
                                            [0, int(hparams.psf_convolution_pad_factor*shape[1])],
                                            [0, int(hparams.psf_convolution_pad_factor*shape[2])]])
    psf_padded = tf.pad(features['psf'][...,0], [[0,0],
                                            [0, int(hparams.psf_convolution_pad_factor*shape[1])],
                                            [0, int(hparams.psf_convolution_pad_factor*shape[2])]])
    reconstr = tf.expand_dims(tf.spectral.irfft2d(tf.spectral.rfft2d(rec_padded)*tf.cast(tf.abs(tf.spectral.rfft2d(psf_padded)), tf.complex64)),axis=-1)
    reconstr = reconstr[:, :shape[1], :shape[2], :]

  # Losses.
  losses = {
      "bottleneck_extra": b_loss,
      "bottleneck_l2": hparams.bottleneck_l2_factor * xb_loss
  }

  loglik = loglikelihood_fn(labels, reconstr, features, hparams)
  targets_loss = tf.reduce_mean(- loglik)

  tf.summary.scalar("negloglik", targets_loss)
  tf.summary.scalar("bottleneck_loss", b_loss)

  losses["training"] = targets_loss
  logits = tf.reshape(reconstr, labels_shape)

  image_summary("ae", reconstr)
  image_summary("input", labels)

  return logits, losses
