from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
import numpy as np

import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.internal import tensorshape_util

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = ['masked_autoregressive_conditional_template',
           'real_nvp_conditional_template',
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

def real_nvp_conditional_template(hidden_layers,
                                  conditional_tensor,
                                  shift_only=False,
                                  activation=tf.nn.relu,
                                  log_scale_min_clip=-5.,
                                  log_scale_max_clip=3.,
                                  log_scale_clip_gradient=False,
                                  name=None,
                                  *args,  # pylint: disable=keyword-arg-before-vararg
                                  **kwargs):
  """Build a scale-and-shift function using a multi-layer neural network.
  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the `d`-dimensional input x[0:d] and returns the `D-d`
  dimensional outputs `loc` ('mu') and `log_scale` ('alpha').
  The default template does not support conditioning and will raise an
  exception if `condition_kwargs` are passed to it. To use conditioning in
  Real NVP bijector, implement a conditioned shift/scale template that
  handles the `condition_kwargs`.
  Arguments:
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    conditional_tensor: Tensor used to condition each RealNVP layer.
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed (i.e. NICE bijector). Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    name: A name for ops managed by this function. Default:
      'real_nvp_default_template'.
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.
  Returns:
    shift: `Float`-like `Tensor` of shift terms ('mu' in
      [Papamakarios et al.  (2016)][1]).
    log_scale: `Float`-like `Tensor` of log(scale) terms ('alpha' in
      [Papamakarios et al. (2016)][1]).
  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution, or if `condition_kwargs` is not empty.
  #### References
  [1]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  """
  with tf.name_scope(name or 'real_nvp_default_template'):
    def _fn(x, output_units, **condition_kwargs):
      """Fully connected MLP parameterized via `real_nvp_template`."""
      if condition_kwargs:
        raise NotImplementedError(
            'Conditioning not implemented in the default template.')

      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
        reshape_output = lambda x: x[0]
      else:
        reshape_output = lambda x: x
      for units in hidden_layers:
        x = tf1.layers.dense(
            inputs=tf.concat([x, conditional_tensor],axis=-1) ,
            units=units,
            activation=activation,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)
      x = tf1.layers.dense(
          inputs=tf.concat([x, conditional_tensor],axis=-1),
          units=(1 if shift_only else 2) * output_units,
          activation=None,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)
      if shift_only:
        return reshape_output(x), None
      shift, log_scale = tf.split(x, 2, axis=-1)
      which_clip = (
          tf.clip_by_value
          if log_scale_clip_gradient else _clip_by_value_preserve_grad)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return reshape_output(shift), reshape_output(log_scale)

    return tf1.make_template('real_nvp_default_template', _fn)

def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
  """Clips input while leaving gradient unaltered."""
  with tf.name_scope(name, "clip_by_value_preserve_grad",
                     [x, clip_value_min, clip_value_max]):
    clip_x = tf.clip_by_value(x, clip_value_min, clip_value_max)
    return x + tf.stop_gradient(clip_x - x)
