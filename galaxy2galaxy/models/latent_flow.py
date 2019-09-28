"""Normalizing flow models learning the latent space of an existing Auto-Encoder
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from galaxy2galaxy.layers.flows import masked_autoregressive_conditional_template

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub


class LatentFlow(t2t_model.T2TModel):
  """ Base class for latent flows

  This assumes that an already exported tensorflow hub autoencoder is provided
  in hparams.
  """

  def normalizing_flow(self, condition, mode):
    """ Function building a normalizing flow, returned as a Tensorflow probability
    distribution
    """
    raise NotImplementedError

  def body(self, features):
    hparams = self.hparams
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    x = features['inputs']
    y = features['attributes']


    # Load the encoder and decoder modules
    encoder = hub.Module(hparams.encoder_module, trainable=False)
    decoder = hub.Module(hparams.decoder_module, trainable=False)

    # Encode the input image
    if hparams.encode_psf and 'psf' in features:
      code = encoder(tf.concat([x, net_psf], axis=-1))
    else:
      code = encoder(x)

    #

    with tf.variable_scope("flow_module"):
      cond_layer = tf.concat([tf.expand_dims(y[k], axis=1) for k in hparams.problem.attributes],axis=1)
      # Apply some amount of normalization to the features
      y = common_layers.layer_norm(y, name="y_norm")
      flow = self.normalizing_flow(cond_layer, is_training)
      loglikelihood = flow.log_prob(code['sample'])

    # This is the loglikelihood of a batch of images
    tf.summary.scalar('loglikelihood', tf.reduce_mean(loglikelihood))
    loss = - tf.reduce_mean(loglikelihood)




    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      y = features
      def flow_module_spec():
        inputs = {k: tf.placeholder(tf.float32, shape=[None]) for k in y.keys()}
        cond_layer = tf.concat([tf.expand_dims(inputs[k], axis=1) for k in inputs.keys()],axis=1)
        flow = self.normalizing_flow(cond_layer, is_training)
        hub.add_signature(inputs=inputs,
                          outputs=flow.sample(tf.shape(cond_layer)[0]))

      flow_spec = hub.create_module_spec(flow_module_spec)
      flow = hub.Module(flow_spec, name='flow_module')
      hub.register_module_for_export(flow, "code_sampler")
      code_sample = flow(y)
      return code_sample, {}
