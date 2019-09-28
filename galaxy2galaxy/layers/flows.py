from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = ['masked_autoregressive_conditional_template', '_clip_by_value_preserve_grad']

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
