"""Autoregressive models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np

from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from pixel_cnn_pp.model import model_spec

import tensorflow as tf
import tensorflow_probability as tfp

@registry.register_model
class Img2imgPixelCNNpp(t2t_model.T2TModel):

  def body(self, features):
    hparams = self.hparams
    model_opt = { 'nr_resnet': hparams.nr_resnet,
                  'nr_filters': hparams.hidden_size,
                  'nr_logistic_mix': 1,
                  'resnet_nonlinearity': 'concat_elu',
                  'energy_distance': False}

    model = tf.make_template('model', model_spec)
    out = model(features["inputs"], None, ema=None,
                    dropout_p=hparams.dropout, **model_opt)
    out = tf.layers.dense(out, 2, activation=None)
    loc, scale = tf.split(out, num_or_size_splits=2, axis=-1)
    scale = tf.nn.softplus(scale) + 1e-4
    distribution = tfp.distributions.Independent( tfp.distributions.Normal(loc=loc, scale=scale))

    return out, {"training": - distribution.log_prob(features["targets"])}

@registry.register_hparams
def image_transformer2d_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 64
  hparams.batch_size = 128
  hparams.dropout = 0.
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 0.2
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.bottom["targets"] = modalities.make_targets_bottom(
      modalities.image_channel_embeddings_bottom)
  hparams.top["targets"] = modalities.identity_top
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  # PixelCNN model opt
  hparams.add_hparam("nr_resnet", 2)
