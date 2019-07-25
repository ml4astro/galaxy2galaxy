"""Continuous Autoencoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.layers import latent_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from tensor2tensor.models.research import autoencoders

import tensorflow as tf

@registry.register_model
class AutoencoderResidualVAEContinuous(autoencoders.AutoencoderResidualVAE):
  """A residual VAE with custom reconstruction loss function.
  """
  def body(self, features):
    hparams = self.hparams
    # Run the parent autoencoder part first.
    renconstr, losses = super().body(features)
    # Replaces the training loss by the right reconstruction loss
    if "training" in losses:
      labels = features["targets"]
      losses['training'] = tf.losses.mean_squared_error(y, reconstr)
    return renconstr, losses

@registry.register_hparams
def autoencoder_residual_vae_continuous():
  """Basic autoencoder model."""
  hparams = autoencoders.autoencoder_residual()
  return hparams
