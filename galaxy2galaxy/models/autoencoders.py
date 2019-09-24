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
from galaxy2galaxy.models.autoencoders_utils import autoencoder_body

import tensorflow as tf

@registry.register_model
class ContinuousAutoencoderBasic(autoencoders.AutoencoderBasic):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderResidual(ContinuousAutoencoderBasic):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderResidualVAE(ContinuousAutoencoderResidual):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderBasicDiscrete(ContinuousAutoencoderBasic):
  def body(self, features):
    return autoencoder_body(self, features)

@registry.register_model
class ContinuousAutoencoderResidualDiscrete(ContinuousAutoencoderResidual):
  """Discrete residual autoencoder."""
  def body(self, features):
    return autoencoder_body(self, features)
