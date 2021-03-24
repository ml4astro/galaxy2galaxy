

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
import numpy as np


import collections
import functools
from tensorflow_probability.python.bijectors import affine_scalar
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util

from galaxy2galaxy.layers.tfp_utils import RationalQuadraticSpline

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = ['masked_autoregressive_conditional_template',
           'ConditionalNeuralSpline',
           'conditional_neural_spline_template',
           'autoregressive_conditional_neural_spline_template',
           '_clip_by_value_preserve_grad']

def masked_autoregressive_conditional_template(hidden_layers,
                                            conditional_tensor,
                                           shift_only=False,
                                           activation=tf.nn.relu,
                                           log_scale_min_clip=-5.,
                                           log_scale_max_clip=3.,
                                           log_scale_clip_gradient=False,
                                           name=None,
                                           *args,  # pylint: disable=keyword-arg-before-vararg
                                           **kwargs):
  name = name or "masked_autoregressive_default_template"
  with tf.name_scope(name, values=[log_scale_min_clip, log_scale_max_clip]):
    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.

      input_shape = (
          np.int32(x.shape.as_list())
          if x.shape.is_fully_defined() else tf.shape(x))
      if len(x.shape) == 1:
        x = x[tf.newaxis, ...]


      x = tf.concat([conditional_tensor, x],  axis=1)
      cond_depth = conditional_tensor.shape.with_rank_at_least(1)[-1].value
      input_depth = x.shape.with_rank_at_least(1)[-1].value

      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")

      for i, units in enumerate(hidden_layers):
        x = tfb.masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)

      x = tfb.masked_dense(
          inputs=x,
          units=(1 if shift_only else 2) * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)

      if shift_only:
        x = x[:, cond_depth:]
        x = tf.reshape(x, shape=input_shape)
        return x, None
      else:
        x = x[:, 2*cond_depth:]
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
      which_clip = (
          tf.clip_by_value
          if log_scale_clip_gradient else _clip_by_value_preserve_grad)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale

    return tf.make_template(name, _fn)

def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
  """Clips input while leaving gradient unaltered."""
  with tf.name_scope(name, "clip_by_value_preserve_grad",
                     [x, clip_value_min, clip_value_max]):
    clip_x = tf.clip_by_value(x, clip_value_min, clip_value_max)
    return x + tf.stop_gradient(clip_x - x)

class ConditionalNeuralSpline(tf.Module):
  def __init__(self, conditional_tensor=None, nbins=32, hidden_layers=[256],
               activation='relu',name=None):
    self._nbins = nbins
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None
    self._layers= []
    self._activation = activation
    self._hidden_layers = hidden_layers
    self._conditional_tensor = conditional_tensor
    super(ConditionalNeuralSpline, self).__init__(name)

  def __call__(self, x, nunits):
    if not self._built:
      def _bin_positions(x):
        x = tf.reshape(x, [-1, nunits, self._nbins])
        return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

      def _slopes(x):
        x = tf.reshape(x, [-1, nunits, self._nbins - 1])
        return tf.math.softplus(x) + 1e-2

      for i, units in enumerate(self._hidden_layers):
        self._layers.append(tf.keras.layers.Dense(units, activation=self._activation,
                                                  name='layer_%d'%i))
      self._bin_widths = tf.keras.layers.Dense(
          nunits * self._nbins, activation=_bin_positions, name='w')

      self._bin_heights = tf.keras.layers.Dense(
          nunits * self._nbins, activation=_bin_positions, name='h')

      self._knot_slopes = tf.keras.layers.Dense(
          nunits * (self._nbins - 1), activation=_slopes, name='s')
      self._built = True

    # If provided, we append the condition as an input to the network
    if self._conditional_tensor is not None:
      net = tf.concat([x, self._conditional_tensor], axis=-1)
    else:
      net = x

    # Apply hidden layers
    for layer in self._layers:
      net = layer(net)

    return RationalQuadraticSpline(
        bin_widths=self._bin_widths(net),
        bin_heights=self._bin_heights(net),
        knot_slopes=self._knot_slopes(net))


def conditional_neural_spline_template(conditional_tensor=None,
                                       nbins=32,
                                       hidden_layers=[256],
                                       activation=tf.nn.relu,
                                       name=None):
  with tf.name_scope(name):
    def _fn(x, nunits):
      # If provided, we append the condition as an input to the network
      if conditional_tensor is not None:
        net = tf.concat([x, conditional_tensor], axis=-1)
      else:
        net = x

      for i, units in enumerate(hidden_layers):
        net = tf.layers.dense(net, units, activation=activation, name='layer_%d'%i)

      def _bin_positions(x):
        x = tf.reshape(x, [-1, nunits, nbins])
        return tf.math.softmax(x, axis=-1) * (2 - nbins * 1e-2) + 1e-2

      def _slopes(x):
        x = tf.reshape(x, [-1, nunits, nbins - 1])
        return tf.math.softplus(x) + 1e-2

      bin_widths = tf.layers.dense(net, nunits * nbins, activation=_bin_positions, name='w')
      bin_heights = tf.layers.dense(net, nunits * nbins, activation=_bin_positions, name='h')
      knot_slopes = tf.layers.dense(net, nunits * (nbins - 1), activation=_slopes, name='s')

      return RationalQuadraticSpline(
          bin_widths=bin_widths,
          bin_heights=bin_heights,
          knot_slopes=knot_slopes)

    return tf.make_template(name, _fn)


def autoregressive_conditional_neural_spline_template(conditional_tensor,
                                                 hidden_layers=[256],
                                                 nbins=32,
                                                 activation=tf.nn.relu,
                                                 name=None,
                                                 *args,  # pylint: disable=keyword-arg-before-vararg
                                                 **kwargs):
  name = name or "autoregressive_conditional_neural_spline_template"
  with tf.name_scope(name):
    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.

      input_shape = (
          np.int32(x.shape.as_list())
          if x.shape.is_fully_defined() else tf.shape(x))
      if len(x.shape) == 1:
        x = x[tf.newaxis, ...]

      x = tf.concat([conditional_tensor, x],  axis=1)
      cond_depth = conditional_tensor.shape.with_rank_at_least(1)[-1].value
      input_depth = x.shape.with_rank_at_least(1)[-1].value

      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")

      def _bin_positions(x):
        x = tf.reshape(x, [-1, input_depth , nbins])
        return tf.math.softmax(x, axis=-1) * (2 - nbins * 1e-2) + 1e-2

      def _slopes(x):
        x = tf.reshape(x, [-1, input_depth, nbins - 1])
        return tf.math.softplus(x) + 1e-2

      for i, units in enumerate(hidden_layers):
        x = tfb.masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)

      bin_widths = tfb.masked_dense(
          inputs=x,
          units=input_depth*nbins,
          num_blocks=input_depth,
          activation=_bin_positions,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)

      bin_heights = tfb.masked_dense(
          inputs=x,
          units=input_depth*nbins,
          num_blocks=input_depth,
          activation=_bin_positions,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)

      knot_slopes = tfb.masked_dense(
          inputs=x,
          units=input_depth*(nbins -1),
          num_blocks=input_depth,
          activation=_slopes,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)

      return RationalQuadraticSpline(
          bin_widths=bin_widths[:, cond_depth:],
          bin_heights=bin_heights[:, cond_depth:],
          knot_slopes=knot_slopes[:, cond_depth:])

    return tf.make_template(name, _fn)
