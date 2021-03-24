# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
""" Backport to TF 1.15 and TFP 0.7 of newer TFP tools.
  - RealNVP: not modified from v0.10
  - RationalQuadraticSpline: modification of all tf.where to avoid shape issues
  and assertions errors
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import functools

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import affine_scalar
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient

__all__ = ["RealNVP", "MaskedAutoregressiveFlow", "RationalQuadraticSpline"]


def _ensure_at_least_1d(t):
  t = tf.convert_to_tensor(t)
  return t + tf.zeros([1], dtype=t.dtype)


def _padded(t, lhs, rhs=None):
  """Left pads and optionally right pads the innermost axis of `t`."""
  lhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
  zeros = tf.zeros([tf.rank(t) - 1, 2], dtype=tf.int32)
  lhs_paddings = tf.concat([zeros, [[1, 0]]], axis=0)
  result = tf.pad(t, paddings=lhs_paddings, constant_values=lhs)
  if rhs is not None:
    rhs = tf.convert_to_tensor(rhs, dtype=t.dtype)
    rhs_paddings = tf.concat([zeros, [[0, 1]]], axis=0)
    result = tf.pad(result, paddings=rhs_paddings, constant_values=rhs)
  return result


def _knot_positions(bin_sizes, range_min):
  return _padded(tf.cumsum(bin_sizes, axis=-1) + range_min, lhs=range_min)


_SplineShared = collections.namedtuple(
    'SplineShared', 'out_of_bounds,x_k,y_k,d_k,d_kp1,h_k,w_k,s_k')

class RealNVP(bijector_lib.Bijector):
  """RealNVP 'affine coupling layer' for vector-valued events.
  Real NVP models a normalizing flow on a `D`-dimensional distribution via a
  single `D-d`-dimensional conditional distribution [(Dinh et al., 2017)][1]:
  `y[d:D] = x[d:D] * tf.exp(log_scale_fn(x[0:d])) + shift_fn(x[0:d])`
  `y[0:d] = x[0:d]`
  The last `D-d` units are scaled and shifted based on the first `d` units only,
  while the first `d` units are 'masked' and left unchanged. Real NVP's
  `shift_and_log_scale_fn` computes vector-valued quantities. For
  scale-and-shift transforms that do not depend on any masked units, i.e.
  `d=0`, use the `tfb.Affine` bijector with learned parameters instead.
  Masking is currently only supported for base distributions with
  `event_ndims=1`. For more sophisticated masking schemes like checkerboard or
  channel-wise masking [(Papamakarios et al., 2016)[4], use the `tfb.Permute`
  bijector to re-order desired masked units into the first `d` units. For base
  distributions with `event_ndims > 1`, use the `tfb.Reshape` bijector to
  flatten the event shape.
  Recall that the MAF bijector [(Papamakarios et al., 2016)][4] implements a
  normalizing flow via an autoregressive transformation. MAF and IAF have
  opposite computational tradeoffs - MAF can train all units in parallel but
  must sample units sequentially, while IAF must train units sequentially but
  can sample in parallel. In contrast, Real NVP can compute both forward and
  inverse computations in parallel. However, the lack of an autoregressive
  transformations makes it less expressive on a per-bijector basis.
  A 'valid' `shift_and_log_scale_fn` must compute each `shift` (aka `loc` or
  'mu' in [Papamakarios et al. (2016)][4]) and `log(scale)` (aka 'alpha' in
  [Papamakarios et al. (2016)][4]) such that each are broadcastable with the
  arguments to `forward` and `inverse`, i.e., such that the calculations in
  `forward`, `inverse` [below] are possible. For convenience,
  `real_nvp_default_template` is offered as a possible `shift_and_log_scale_fn`
  function.
  NICE [(Dinh et al., 2014)][2] is a special case of the Real NVP bijector
  which discards the scale transformation, resulting in a constant-time
  inverse-log-determinant-Jacobian. To use a NICE bijector instead of Real
  NVP, `shift_and_log_scale_fn` should return `(shift, None)`, and
  `is_constant_jacobian` should be set to `True` in the `RealNVP` constructor.
  Calling `real_nvp_default_template` with `shift_only=True` returns one such
  NICE-compatible `shift_and_log_scale_fn`.
  The `bijector_fn` argument allows specifying a more general coupling relation,
  such as the LSTM-inspired activation from [5], or Neural Spline Flow [6].
  Caching: the scalar input depth `D` of the base distribution is not known at
  construction time. The first call to any of `forward(x)`, `inverse(x)`,
  `inverse_log_det_jacobian(x)`, or `forward_log_det_jacobian(x)` memoizes
  `D`, which is re-used in subsequent calls. This shape must be known prior to
  graph execution (which is the case if using tf.layers).
  #### Examples
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  nvp = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=[0., 0., 0.]),
      bijector=tfb.RealNVP(
          num_masked=2,
          shift_and_log_scale_fn=tfb.real_nvp_default_template(
              hidden_layers=[512, 512])))
  x = nvp.sample()
  nvp.log_prob(x)
  nvp.log_prob(0.)
  ```
  For more examples, see [Jang (2018)][3].
  #### References
  [1]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803
  [2]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear
       Independent Components Estimation. _arXiv preprint arXiv:1410.8516_,
       2014. https://arxiv.org/abs/1410.8516
  [3]: Eric Jang. Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows.
       _Technical Report_, 2018. http://blog.evjang.com/2018/01/nf2.html
  [4]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  [5]: Diederik P Kingma, Tim Salimans, Max Welling. Improving Variational
       Inference with Inverse Autoregressive Flow. In _Neural Information
       Processing Systems_, 2016. https://arxiv.org/abs/1606.04934
  [6]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
       Spline Flows, 2019. http://arxiv.org/abs/1906.04032
  """

  def __init__(self,
               num_masked=None,
               fraction_masked=None,
               shift_and_log_scale_fn=None,
               bijector_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               name=None):
    """Creates the Real NVP or NICE bijector.
    Args:
      num_masked: Python `int`, indicating the number of units of the
        event that should should be masked. Must be in the closed interval
        `[0, D-1]`, where `D` is the event size of the base distribution.
        If the value is negative, then the last `d` units of the event are
        masked instead. Must be `None` if `fraction_masked` is defined.
      fraction_masked: Python `float`, indicating the number of units of the
        event that should should be masked. Must be in the closed interval
        `[-1, 1]`, and the value represents the fraction of the values to be
        masked. The final number of values to be masked will be the input size
        times the fraction, rounded to the the nearest integer towards zero.
        If negative, then the last fraction of units are masked instead. Must
        be `None` if `num_masked` is defined.
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from both the forward domain (`x`) and the inverse domain
        (`y`). Calculation must respect the 'autoregressive property' (see class
        docstring). Suggested default
        `masked_autoregressive_default_template(hidden_layers=...)`.
        Typically the function contains `tf.Variables` and is wrapped using
        `tf.make_template`. Returning `None` for either (both) `shift`,
        `log_scale` is equivalent to (but more efficient than) returning zero.
      bijector_fn: Python `callable` which returns a `tfb.Bijector` which
        transforms the last `D-d` unit with the signature `(masked_units_tensor,
        output_units, **condition_kwargs) -> bijector`. The bijector must
        operate on scalar or vector events and must not alter the rank of its
        input.
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.
    Raises:
      ValueError: If both or none of `shift_and_log_scale_fn` and `bijector_fn`
          are specified.
    """
    name = name or 'real_nvp'
    with tf.name_scope(name) as name:
      # At construction time, we don't know input_depth.
      self._input_depth = None
      if num_masked is not None and fraction_masked is not None:
        raise ValueError('Exactly one of `num_masked` and '
                         '`fraction_masked` should be specified.')

      if num_masked is not None:
        if int(num_masked) != num_masked:
          raise TypeError('`num_masked` must be an integer. Got: {} of type {}'
                          ''.format(num_masked, type(num_masked)))
        self._num_masked = int(num_masked)
        self._fraction_masked = None
        self._reverse_mask = self._num_masked < 0
      else:
        if not np.issubdtype(type(fraction_masked), np.floating):
          raise TypeError('`fraction_masked` must be a float. Got: {} of type '
                          '{}'.format(fraction_masked, type(fraction_masked)))
        if np.abs(fraction_masked) >= 1.:
          raise ValueError(
              '`fraction_masked` must be in (-1, 1), but is {}.'.format(
                  fraction_masked))
        self._num_masked = None
        self._fraction_masked = float(fraction_masked)
        self._reverse_mask = self._fraction_masked < 0

      if shift_and_log_scale_fn is not None and bijector_fn is not None:
        raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                         '`bijector_fn` should be specified.')

      if shift_and_log_scale_fn:
        def _bijector_fn(x0, input_depth, **condition_kwargs):
          shift, log_scale = shift_and_log_scale_fn(x0, input_depth,
                                                    **condition_kwargs)
          return affine_scalar.AffineScalar(shift=shift, log_scale=log_scale)

        bijector_fn = _bijector_fn

      if validate_args:
        bijector_fn = _validate_bijector_fn(bijector_fn)

      # Still do this assignment for variable tracking.
      self._shift_and_log_scale_fn = shift_and_log_scale_fn
      self._bijector_fn = bijector_fn

      super(RealNVP, self).__init__(
          forward_min_event_ndims=1,
          is_constant_jacobian=is_constant_jacobian,
          validate_args=validate_args,
          name=name)

  @property
  def _masked_size(self):
    masked_size = (
        self._num_masked if self._num_masked is not None else int(
            np.round(self._input_depth * self._fraction_masked)))
    return masked_size

  def _cache_input_depth(self, x):
    if self._input_depth is None:
      self._input_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
      if self._input_depth is None:
        raise NotImplementedError(
            'Rightmost dimension must be known prior to graph execution.')

      if abs(self._masked_size) >= self._input_depth:
        raise ValueError(
            'Number of masked units {} must be smaller than the event size {}.'
            .format(self._masked_size, self._input_depth))

  def _bijector_input_units(self):
    return self._input_depth - abs(self._masked_size)

  def _forward(self, x, **condition_kwargs):
    self._cache_input_depth(x)

    x0, x1 = x[..., :self._masked_size], x[..., self._masked_size:]

    if self._reverse_mask:
      x0, x1 = x1, x0

    y1 = self._bijector_fn(x0, self._bijector_input_units(),
                           **condition_kwargs).forward(x1)

    if self._reverse_mask:
      y1, x0 = x0, y1

    y = tf.concat([x0, y1], axis=-1)
    return y

  def _inverse(self, y, **condition_kwargs):
    self._cache_input_depth(y)

    y0, y1 = y[..., :self._masked_size], y[..., self._masked_size:]

    if self._reverse_mask:
      y0, y1 = y1, y0

    x1 = self._bijector_fn(y0, self._bijector_input_units(),
                           **condition_kwargs).inverse(y1)

    if self._reverse_mask:
      x1, y0 = y0, x1

    x = tf.concat([y0, x1], axis=-1)
    return x

  def _forward_log_det_jacobian(self, x, **condition_kwargs):
    self._cache_input_depth(x)

    x0, x1 = x[..., :self._masked_size], x[..., self._masked_size:]

    if self._reverse_mask:
      x0, x1 = x1, x0

    return self._bijector_fn(x0, self._bijector_input_units(),
                             **condition_kwargs).forward_log_det_jacobian(
                                 x1, event_ndims=1)

  def _inverse_log_det_jacobian(self, y, **condition_kwargs):
    self._cache_input_depth(y)

    y0, y1 = y[..., :self._masked_size], y[..., self._masked_size:]

    if self._reverse_mask:
      y0, y1 = y1, y0

    return self._bijector_fn(y0, self._bijector_input_units(),
                             **condition_kwargs).inverse_log_det_jacobian(
                                 y1, event_ndims=1)

class MaskedAutoregressiveFlow(bijector_lib.Bijector):
  """Affine MaskedAutoregressiveFlow bijector.
  The affine autoregressive flow [(Papamakarios et al., 2016)][3] provides a
  relatively simple framework for user-specified (deep) architectures to learn a
  distribution over continuous events.  Regarding terminology,
    'Autoregressive models decompose the joint density as a product of
    conditionals, and model each conditional in turn.  Normalizing flows
    transform a base density (e.g. a standard Gaussian) into the target density
    by an invertible transformation with tractable Jacobian.'
    [(Papamakarios et al., 2016)][3]
  In other words, the 'autoregressive property' is equivalent to the
  decomposition, `p(x) = prod{ p(x[perm[i]] | x[perm[0:i]]) : i=0, ..., d }`
  where `perm` is some permutation of `{0, ..., d}`.  In the simple case where
  the permutation is identity this reduces to:
  `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`.
  In TensorFlow Probability, 'normalizing flows' are implemented as
  `tfp.bijectors.Bijector`s.  The `forward` 'autoregression' is implemented
  using a `tf.while_loop` and a deep neural network (DNN) with masked weights
  such that the autoregressive property is automatically met in the `inverse`.
  A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
  (expensive) forward-mode calculation to draw samples and the (cheap)
  reverse-mode calculation to compute log-probabilities.  Conversely, a
  `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
  the (expensive) forward-mode calculation to compute log-probabilities and the
  (cheap) reverse-mode calculation to compute samples.  See 'Example Use'
  [below] for more details.
  Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
  (a sequence of) affine transformations.  A 'valid' `shift_and_log_scale_fn`
  must compute each `shift` (aka `loc` or 'mu' in [Germain et al. (2015)][1])
  and `log(scale)` (aka 'alpha' in [Germain et al. (2015)][1]) such that each
  are broadcastable with the arguments to `forward` and `inverse`, i.e., such
  that the calculations in `forward`, `inverse` [below] are possible.
  For convenience, `tfp.bijectors.AutoregressiveNetwork` is offered as a
  possible `shift_and_log_scale_fn` function.  It implements the MADE
  architecture [(Germain et al., 2015)][1].  MADE is a feed-forward network that
  computes a `shift` and `log(scale)` using masked dense layers in a deep
  neural network. Weights are masked to ensure the autoregressive property. It
  is possible that this architecture is suboptimal for your task. To build
  alternative networks, either change the arguments to
  `tfp.bijectors.AutoregressiveNetwork` or use some other architecture, e.g.,
  using `tf.keras.layers`.
  Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
  enforces the 'autoregressive property'.
  Assuming `shift_and_log_scale_fn` has valid shape and autoregressive
  semantics, the forward transformation is
  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-event_dims:].num_elements()
    for _ in range(event_size):
      shift, log_scale = shift_and_log_scale_fn(y)
      y = x * tf.exp(log_scale) + shift
    return y
  ```
  and the inverse transformation is
  ```python
  def inverse(y):
    shift, log_scale = shift_and_log_scale_fn(y)
    return (y - shift) / tf.exp(log_scale)
  ```
  Notice that the `inverse` does not need a for-loop.  This is because in the
  forward pass each calculation of `shift` and `log_scale` is based on the `y`
  calculated so far (not `x`).  In the `inverse`, the `y` is fully known, thus
  is equivalent to the scaling used in `forward` after `event_size` passes,
  i.e., the 'last' `y` used to compute `shift`, `log_scale`.  (Roughly speaking,
  this also proves the transform is bijective.)
  The `bijector_fn` argument allows specifying a more general coupling relation,
  such as the LSTM-inspired activation from [4], or Neural Spline Flow [5].  It
  must logically operate on each element of the input individually, and still
  obey the 'autoregressive property' described above.  The forward
  transformation is
  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-event_dims:].num_elements()
    for _ in range(event_size):
      bijector = bijector_fn(y)
      y = bijector.forward(x)
    return y
  ```
  and inverse transformation is
  ```python
  def inverse(y):
      bijector = bijector_fn(y)
      return bijector.inverse(y)
  ```
  #### Examples
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  dims = 2
  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution.  (However, any continuous distribution would work.) E.g.,
  maf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
              params=2, hidden_units=[512, 512])),
      event_shape=[dims])
  x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  maf.log_prob(x)   # Almost free; uses Bijector caching.
  # Cheap; no `tf.while_loop` despite no Bijector caching.
  maf.log_prob(tf.zeros(dims))
  # [Papamakarios et al. (2016)][3] also describe an Inverse Autoregressive
  # Flow [(Kingma et al., 2016)][2]:
  iaf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
              params=2, hidden_units=[512, 512]))),
      event_shape=[dims])
  x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
  iaf.log_prob(x)   # Almost free; uses Bijector caching.
  # Expensive; uses `tf.while_loop`, no Bijector caching.
  iaf.log_prob(tf.zeros(dims))
  # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
  # poor choice.  Here's an example of using a 'shift only' version and with a
  # different number/depth of hidden layers.
  made = tfb.AutoregressiveNetwork(params=1, hidden_units=[32])
  maf_no_scale_hidden2 = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          lambda y: (made(y)[..., 0], None),
          is_constant_jacobian=True),
      event_shape=[dims])
  maf_no_scale_hidden2._made = made  # Ensure maf_no_scale_hidden2.trainable
  # NOTE: The last line ensures that maf_no_scale_hidden2.trainable_variables
  # will include all variables from `made`.
  ```
  #### Variable Tracking
  NOTE: Like all subclasses of `tfb.Bijector`, `tfb.MaskedAutoregressiveFlow`
  subclasses `tf.Module` for variable tracking.
  A `tfb.MaskedAutoregressiveFlow` instance saves a reference to the values
  passed as `shift_and_log_scale_fn` and `bijector_fn` to its constructor.
  Thus, for most values passed as `shift_and_log_scale_fn` or `bijector_fn`,
  variables referenced by those values will be found and tracked by the
  `tfb.MaskedAutoregressiveFlow` instance.  Please see the `tf.Module`
  documentation for further details.
  However, if the value passed to `shift_and_log_scale_fn` or `bijector_fn` is a
  Python function, then `tfb.MaskedAutoregressiveFlow` cannot automatically
  track variables used inside `shift_and_log_scale_fn` or `bijector_fn`.  To get
  `tfb.MaskedAutoregressiveFlow` to track such variables, either:
   1. Replace the Python function with a `tf.Module`, `tf.keras.Layer`,
      or other callable object through which `tf.Module` can find variables.
   2. Or, add a reference to the variables to the `tfb.MaskedAutoregressiveFlow`
      instance by setting an attribute -- for example:
      ````
      made1 = tfb.AutoregressiveNetwork(params=1, hidden_units=[10, 10])
      made2 = tfb.AutoregressiveNetwork(params=1, hidden_units=[10, 10])
      maf = tfb.MaskedAutoregressiveFlow(lambda y: (made1(y), made2(y) + 1.))
      maf._made_variables = made1.variables + made2.variables
      ````
  #### References
  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  [2]: Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya
       Sutskever, and Max Welling. Improving Variational Inference with Inverse
       Autoregressive Flow. In _Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.04934
  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  [4]: Diederik P Kingma, Tim Salimans, Max Welling. Improving Variational
       Inference with Inverse Autoregressive Flow. In _Neural Information
       Processing Systems_, 2016. https://arxiv.org/abs/1606.04934
  [5]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
       Spline Flows, 2019. http://arxiv.org/abs/1906.04032
  """

  def __init__(self,
               shift_and_log_scale_fn=None,
               bijector_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               unroll_loop=False,
               event_ndims=1,
               name=None):
    """Creates the MaskedAutoregressiveFlow bijector.
    Args:
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from the inverse domain (`y`). Calculation must respect the
        'autoregressive property' (see class docstring). Suggested default
        `tfb.AutoregressiveNetwork(params=2, hidden_layers=...)`.
        Typically the function contains `tf.Variables`. Returning `None` for
        either (both) `shift`, `log_scale` is equivalent to (but more efficient
        than) returning zero. If `shift_and_log_scale_fn` returns a single
        `Tensor`, the returned value will be unstacked to get the `shift` and
        `log_scale`: `tf.unstack(shift_and_log_scale_fn(y), num=2, axis=-1)`.
      bijector_fn: Python `callable` which returns a `tfb.Bijector` which
        transforms event tensor with the signature
        `(input, **condition_kwargs) -> bijector`. The bijector must operate on
        scalar events and must not alter the rank of its input. The
        `bijector_fn` will be called with `Tensors` from the inverse domain
        (`y`). Calculation must respect the 'autoregressive property' (see
        class docstring).
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      unroll_loop: Python `bool` indicating whether the `tf.while_loop` in
        `_forward` should be replaced with a static for loop. Requires that
        the final dimension of `x` be known at graph construction time. Defaults
        to `False`.
      event_ndims: Python `integer`, the intrinsic dimensionality of this
        bijector. 1 corresponds to a simple vector autoregressive bijector as
        implemented by the `tfp.bijectors.AutoregressiveNetwork`, 2 might be
        useful for a 2D convolutional `shift_and_log_scale_fn` and so on.
      name: Python `str`, name given to ops managed by this object.
    Raises:
      ValueError: If both or none of `shift_and_log_scale_fn` and `bijector_fn`
          are specified.
    """
    name = name or 'masked_autoregressive_flow'
    with tf.name_scope(name) as name:
      self._unroll_loop = unroll_loop
      self._event_ndims = event_ndims
      if bool(shift_and_log_scale_fn) == bool(bijector_fn):
        raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                         '`bijector_fn` should be specified.')
      if shift_and_log_scale_fn:
        def _bijector_fn(x, **condition_kwargs):
          params = shift_and_log_scale_fn(x, **condition_kwargs)
          if tf.is_tensor(params):
            shift, log_scale = tf.unstack(params, num=2, axis=-1)
          else:
            shift, log_scale = params
          return affine_scalar.AffineScalar(shift=shift, log_scale=log_scale)

        bijector_fn = _bijector_fn

      if validate_args:
        bijector_fn = _validate_bijector_fn(bijector_fn)
      # Still do this assignment for variable tracking.
      self._shift_and_log_scale_fn = shift_and_log_scale_fn
      self._bijector_fn = bijector_fn
      super(MaskedAutoregressiveFlow, self).__init__(
          forward_min_event_ndims=self._event_ndims,
          is_constant_jacobian=is_constant_jacobian,
          validate_args=validate_args,
          name=name)

  def _forward(self, x, **kwargs):
    static_event_size = tensorshape_util.num_elements(
        tensorshape_util.with_rank_at_least(
            x.shape, self._event_ndims)[-self._event_ndims:])

    if self._unroll_loop:
      if not static_event_size:
        raise ValueError(
            'The final {} dimensions of `x` must be known at graph '
            'construction time if `unroll_loop=True`. `x.shape: {!r}`'.format(
                self._event_ndims, x.shape))
      y = tf.zeros_like(x, name='y0')

      for _ in range(static_event_size):
        y = self._bijector_fn(y, **kwargs).forward(x)
      return y

    event_size = tf.reduce_prod(tf.shape(x)[-self._event_ndims:])
    y0 = tf.zeros_like(x, name='y0')
    # call the template once to ensure creation
    if not tf.executing_eagerly():
      _ = self._bijector_fn(y0, **kwargs).forward(y0)
    def _loop_body(index, y0):
      """While-loop body for autoregression calculation."""
      # Set caching device to avoid re-getting the tf.Variable for every while
      # loop iteration.
      with tf1.variable_scope(tf1.get_variable_scope()) as vs:
        if vs.caching_device is None and not tf.executing_eagerly():
          vs.set_caching_device(lambda op: op.device)
        bijector = self._bijector_fn(y0, **kwargs)
      y = bijector.forward(x)
      return index + 1, y
    # If the event size is available at graph construction time, we can inform
    # the graph compiler of the maximum number of steps. If not,
    # static_event_size will be None, and the maximum_iterations argument will
    # have no effect.
    _, y = tf.while_loop(
        cond=lambda index, _: index < event_size,
        body=_loop_body,
        loop_vars=(0, y0),
        maximum_iterations=static_event_size)
    return y

  def _inverse(self, y, **kwargs):
    bijector = self._bijector_fn(y, **kwargs)
    return bijector.inverse(y)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    return self._bijector_fn(y, **kwargs).inverse_log_det_jacobian(
        y, event_ndims=self._event_ndims)

class RationalQuadraticSpline(bijector.Bijector):
  """A piecewise rational quadratic spline, as developed in [1].
  This transformation represents a monotonically increasing piecewise rational
  quadratic function. Outside of the bounds of `knot_x`/`knot_y`, the transform
  behaves as an identity function.
  Typically this bijector will be used as part of a chain, with splines for
  trailing `x` dimensions conditioned on some of the earlier `x` dimensions, and
  with the inverse then solved first for unconditioned dimensions, then using
  conditioning derived from those inverses, and so forth. For example, if we
  split a 15-D `xs` vector into 3 components, we may implement a forward and
  inverse as follows:
  ```python
  nsplits = 3
  class SplineParams(tf.Module):
    def __init__(self, nbins=32):
      self._nbins = nbins
      self._built = False
      self._bin_widths = None
      self._bin_heights = None
      self._knot_slopes = None
    def _bin_positions(self, x):
      x = tf.reshape(x, [-1, self._nbins])
      return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2
    def _slopes(self, x):
      x = tf.reshape(x, [-1, self._nbins - 1])
      return tf.math.softplus(x) + 1e-2
    def __call__(self, x, nunits):
      if not self._built:
        self._bin_widths = tf.keras.layers.Dense(
            nunits * self._nbins, activation=self._bin_positions, name='w')
        self._bin_heights = tf.keras.layers.Dense(
            nunits * self._nbins, activation=self._bin_positions, name='h')
        self._knot_slopes = tf.keras.layers.Dense(
            nunits * (self._nbins - 1), activation=self._slopes, name='s')
        self._built = True
      return tfb.RationalQuadraticSpline(
          bin_widths=self._bin_widths(x),
          bin_heights=self._bin_heights(x),
          knot_slopes=self._knot_slopes(x))
  xs = np.random.randn(1, 15).astype(np.float32)  # Keras won't Dense(.)(vec).
  splines = [SplineParams() for _ in range(nsplits)]
  def spline_flow():
    stack = tfb.Identity()
    for i in range(nsplits):
      stack = tfb.RealNVP(5 * i, bijector_fn=splines[i])(stack)
    return stack
  ys = spline_flow().forward(xs)
  ys_inv = spline_flow().inverse(ys)  # ys_inv ~= xs
  ```
  For a one-at-a-time autoregressive flow as in [1], it would be profitable to
  implement a mask over `xs` to parallelize either the inverse or the forward
  pass and implement the other using a `tf.while_loop`. See
  `tfp.bijectors.MaskedAutoregressiveFlow` for support doing so (paired with
  `tfp.bijectors.Invert` depending which direction should be parallel).
  #### References
  [1]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
       Spline Flows. _arXiv preprint arXiv:1906.04032_, 2019.
       https://arxiv.org/abs/1906.04032
  """

  def __init__(self,
               bin_widths,
               bin_heights,
               knot_slopes,
               range_min=-1.,
               validate_args=False,
               name=None):
    """Construct a new RationalQuadraticSpline bijector.
    For each argument, the innermost axis indexes bins/knots and batch axes
    index axes of `x`/`y` spaces. A `RationalQuadraticSpline` with a separate
    transform for each of three dimensions might have `bin_widths` shaped
    `[3, 32]`. To use the same spline for each of `x`'s three dimensions we may
    broadcast against `x` and use a `bin_widths` parameter shaped `[32]`.
    Parameters will be broadcast against each other and against the input
    `x`/`y`s, so if we want fixed slopes, we can use kwarg `knot_slopes=1`.
    A typical recipe for acquiring compatible bin widths and heights would be:
    ```python
    nbins = unconstrained_vector.shape[-1]
    range_min, range_max, min_bin_size = -1, 1, 1e-2
    scale = range_max - range_min - nbins * min_bin_size
    bin_widths = tf.math.softmax(unconstrained_vector) * scale + min_bin_size
    ```
    Args:
      bin_widths: The widths of the spans between subsequent knot `x` positions,
        a floating point `Tensor`. Must be positive, and at least 1-D. Innermost
        axis must sum to the same value as `bin_heights`. The knot `x` positions
        will be a first at `range_min`, followed by knots at `range_min +
        cumsum(bin_widths, axis=-1)`.
      bin_heights: The heights of the spans between subsequent knot `y`
        positions, a floating point `Tensor`. Must be positive, and at least
        1-D. Innermost axis must sum to the same value as `bin_widths`. The knot
        `y` positions will be a first at `range_min`, followed by knots at
        `range_min + cumsum(bin_heights, axis=-1)`.
      knot_slopes: The slope of the spline at each knot, a floating point
        `Tensor`. Must be positive. `1`s are implicitly padded for the first and
        last implicit knots corresponding to `range_min` and `range_min +
        sum(bin_widths, axis=-1)`. Innermost axis size should be 1 less than
        that of `bin_widths`/`bin_heights`, or 1 for broadcasting.
      range_min: The `x`/`y` position of the first knot, which has implicit
        slope `1`. `range_max` is implicit, and can be computed as `range_min +
        sum(bin_widths, axis=-1)`. Scalar floating point `Tensor`.
      validate_args: Toggles argument validation (can hurt performance).
      name: Optional name scope for associated ops. (Defaults to
        `'RationalQuadraticSpline'`).
    """
    with tf.name_scope(name or 'RationalQuadraticSpline') as name:
      dtype = dtype_util.common_dtype(
          [bin_widths, bin_heights, knot_slopes, range_min])
      self._bin_widths = bin_widths
      self._bin_heights = bin_heights
      self._knot_slopes = knot_slopes
      self._range_min = range_min
      super(RationalQuadraticSpline, self).__init__(
          dtype=dtype,
          forward_min_event_ndims=0,
          validate_args=validate_args,
          name=name)

  @property
  def bin_widths(self):
    return self._bin_widths

  @property
  def bin_heights(self):
    return self._bin_heights

  @property
  def knot_slopes(self):
    return self._knot_slopes

  @property
  def range_min(self):
    return self._range_min

  @classmethod
  def _is_increasing(cls):
    return True

  def _compute_shared(self, x=None, y=None):
    """Captures shared computations across forward/inverse/logdet.
    Only one of `x` or `y` should be specified.
    Args:
      x: The `x` values we will search for.
      y: The `y` values we will search for.
    Returns:
      data: A namedtuple with named fields containing shared computations.
    """
    assert (x is None) != (y is None)
    is_x = x is not None

    range_min = tf.convert_to_tensor(self.range_min, name='range_min')
    kx = _knot_positions(self.bin_widths, range_min)
    ky = _knot_positions(self.bin_heights, range_min)
    kd = _padded(_ensure_at_least_1d(self.knot_slopes), lhs=1, rhs=1)
    kx_or_ky = kx if is_x else ky
    kx_or_ky_min = kx_or_ky[..., 0]
    kx_or_ky_max = kx_or_ky[..., -1]
    x_or_y = x if is_x else y
    out_of_bounds = (x_or_y <= kx_or_ky_min) | (x_or_y >= kx_or_ky_max)
    x_or_y = tf.where(out_of_bounds, kx_or_ky_min, x_or_y)

    shape = functools.reduce(
        tf.broadcast_dynamic_shape,
        (
            tf.shape(x_or_y[..., tf.newaxis]),  # Add a n_knots dim.
            tf.shape(kx),
            tf.shape(ky),
            tf.shape(kd)))

    bc_x_or_y = tf.broadcast_to(x_or_y, shape[:-1])
    bc_kx = tf.broadcast_to(kx, shape)
    bc_ky = tf.broadcast_to(ky, shape)
    bc_kd = tf.broadcast_to(kd, shape)
    bc_kx_or_ky = bc_kx if is_x else bc_ky
    indices = tf.maximum(
        tf.zeros([], dtype=tf.int64),
        tf.searchsorted(
            bc_kx_or_ky[..., :-1],
            bc_x_or_y[..., tf.newaxis],
            side='right',
            out_type=tf.int64) - 1)

    def gather_squeeze(params, indices):
      rank = tensorshape_util.rank(indices.shape)
      if rank is None:
        raise ValueError('`indices` must have statically known rank.')
      return tf.gather(params, indices, axis=-1, batch_dims=rank - 1)[..., 0]

    x_k = gather_squeeze(bc_kx, indices)
    x_kp1 = gather_squeeze(bc_kx, indices + 1)
    y_k = gather_squeeze(bc_ky, indices)
    y_kp1 = gather_squeeze(bc_ky, indices + 1)
    d_k = gather_squeeze(bc_kd, indices)
    d_kp1 = gather_squeeze(bc_kd, indices + 1)
    h_k = y_kp1 - y_k
    w_k = x_kp1 - x_k
    s_k = h_k / w_k

    return _SplineShared(
        out_of_bounds=out_of_bounds,
        x_k=x_k,
        y_k=y_k,
        d_k=d_k,
        d_kp1=d_kp1,
        h_k=h_k,
        w_k=w_k,
        s_k=s_k)

  def _forward(self, x):
    """Compute the forward transformation (Appendix A.1)."""
    d = self._compute_shared(x=x)
    relx = (x - d.x_k) / d.w_k
    spline_val = (
        d.y_k + ((d.h_k * (d.s_k * relx**2 + d.d_k * relx * (1 - relx))) /
                 (d.s_k + (d.d_kp1 + d.d_k - 2 * d.s_k) * relx * (1 - relx))))
    y_val = tf.where(d.out_of_bounds, x, spline_val)
    return y_val

  def _inverse(self, y):
    """Compute the inverse transformation (Appendix A.3)."""
    d = self._compute_shared(y=y)
    rely = tf.where(d.out_of_bounds, tf.zeros_like(y), y - d.y_k)
    term2 = rely * (d.d_kp1 + d.d_k - 2 * d.s_k)
    # These terms are the a, b, c terms of the quadratic formula.
    a = d.h_k * (d.s_k - d.d_k) + term2
    b = d.h_k * d.d_k - term2
    c = -d.s_k * rely
    # The expression used here has better numerical behavior for small 4*a*c.
    relx = tf.where(
        tf.equal(rely, 0), tf.zeros_like(a),
        (2 * c) / (-b - tf.sqrt(b**2 - 4 * a * c)))
    return tf.where(d.out_of_bounds, y, relx * d.w_k + d.x_k)

  def _forward_log_det_jacobian(self, x):
    """Compute the forward derivative (Appendix A.2)."""
    d = self._compute_shared(x=x)
    relx = (x - d.x_k) / d.w_k
    relx = tf.where(d.out_of_bounds, 0.5*tf.ones_like(x), relx)
    grad = (
        2 * tf.math.log(d.s_k) +
        tf.math.log(d.d_kp1 * relx**2 + 2 * d.s_k * relx * (1 - relx) +  # newln
                    d.d_k * (1 - relx)**2) -
        2 * tf.math.log((d.d_kp1 + d.d_k - 2 * d.s_k) * relx *
                        (1 - relx) + d.s_k))
    return tf.where(d.out_of_bounds, tf.zeros_like(grad), grad)

  def _parameter_control_dependencies(self, is_init):
    """Validate parameters."""
    bw, bh, kd = None, None, None
    try:
      shape = tf.broadcast_static_shape(self.bin_widths.shape,
                                        self.bin_heights.shape)
    except ValueError as e:
      raise ValueError('`bin_widths`, `bin_heights` must broadcast: {}'.format(
          str(e)))
    bin_sizes_shape = shape
    try:
      shape = tf.broadcast_static_shape(shape[:-1], self.knot_slopes.shape[:-1])
    except ValueError as e:
      raise ValueError(
          '`bin_widths`, `bin_heights`, and `knot_slopes` must broadcast on '
          'batch axes: {}'.format(str(e)))

    assertions = []
    if (tensorshape_util.is_fully_defined(bin_sizes_shape[-1:]) and
        tensorshape_util.is_fully_defined(self.knot_slopes.shape[-1:])):
      if tensorshape_util.rank(self.knot_slopes.shape) > 0:
        num_interior_knots = tensorshape_util.dims(bin_sizes_shape)[-1] - 1
        if tensorshape_util.dims(
            self.knot_slopes.shape)[-1] not in (1, num_interior_knots):
          raise ValueError(
              'Innermost axis of non-scalar `knot_slopes` must broadcast with '
              '{}; got {}.'.format(num_interior_knots, self.knot_slopes.shape))
    # elif self.validate_args:
    #   if is_init != any(tensor_util.is_ref(t)
    #       for t in (self.bin_widths, self.bin_heights, self.knot_slopes)):
    #     bw = tf.convert_to_tensor(self.bin_widths) if bw is None else bw
    #     bh = tf.convert_to_tensor(self.bin_heights) if bh is None else bh
    #     kd = _ensure_at_least_1d(self.knot_slopes) if kd is None else kd
    #     shape = tf.broadcast_dynamic_shape(
    #         tf.shape((bw + bh)[..., :-1]), tf.shape(kd))
    #     assertions.append(
    #         assert_util.assert_greater(
    #             tf.shape(shape)[0],
    #             tf.zeros([], dtype=shape.dtype),
    #             message='`(bin_widths + bin_heights)[..., :-1]` must broadcast '
    #             'with `knot_slopes` to at least 1-D.'))

    if not self.validate_args:
      assert not assertions
      return assertions

    # if (is_init != tensor_util.is_ref(self.bin_widths) or
    #     is_init != tensor_util.is_ref(self.bin_heights)):
    #   bw = tf.convert_to_tensor(self.bin_widths) if bw is None else bw
    #   bh = tf.convert_to_tensor(self.bin_heights) if bh is None else bh
    #   assertions += [
    #       assert_util.assert_near(
    #           tf.reduce_sum(bw, axis=-1),
    #           tf.reduce_sum(bh, axis=-1),
    #           message='`sum(bin_widths, axis=-1)` must equal '
    #           '`sum(bin_heights, axis=-1)`.'),
    #   ]
    # if is_init != tensor_util.is_ref(self.bin_widths):
    #   bw = tf.convert_to_tensor(self.bin_widths) if bw is None else bw
    #   assertions += [
    #       assert_util.assert_positive(
    #           bw, message='`bin_widths` must be positive.'),
    #   ]
    # if is_init != tensor_util.is_ref(self.bin_heights):
    #   bh = tf.convert_to_tensor(self.bin_heights) if bh is None else bh
    #   assertions += [
    #       assert_util.assert_positive(
    #           bh, message='`bin_heights` must be positive.'),
    #   ]
    # if is_init != tensor_util.is_ref(self.knot_slopes):
    #   kd = _ensure_at_least_1d(self.knot_slopes) if kd is None else kd
    #   assertions += [
    #       assert_util.assert_positive(
    #           kd, message='`knot_slopes` must be positive.'),
    #   ]
    return assertions
