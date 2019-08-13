""" Spectral Norm GAN """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.estimator.python.gan_estimator_impl import _make_prediction_gan_model, _summary_type_map, _get_estimator_spec
from tensorflow.contrib.gan.python.losses.python.tuple_losses_impl import _args_to_gan_model
from tensorflow.python.summary import summary
from tensorflow.python.ops.losses import util
from tensorflow.contrib.framework.python.ops import variables as variable_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.util import tf_inspect as inspect

from tensor2tensor.layers.common_layers import apply_spectral_norm


class SpectralNormConstraint(tf.keras.constraints.Constraint):
    """Constrains the weights to be normalized.
    """
    def __init__(self, update, name):
        self.update = update
        self.name = name

    def __call__(self, w):
        with tf.variable_scope(self.name):
            w, assign_op = apply_spectral_norm(w)
        if self.update:
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_op)
        return w

def _softplus_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    """
    Applies softplus
    """
    with ops.name_scope(scope, 'generator_softplus_loss',
                        (discriminator_gen_outputs, weights)) as scope:
        discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)

        loss = tf.nn.softplus(- discriminator_gen_outputs)
        loss = losses.compute_weighted_loss(
            loss, weights, scope, loss_collection, reduction)

        if add_summaries:
            summary.scalar('generator_softplus_loss', loss)
        return loss

def _softplus_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    """
    Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
    Returns:
    A loss Tensor. The shape depends on `reduction`.
    """
    with ops.name_scope(scope, 'discriminator_softplus_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights)) as scope:
        discriminator_real_outputs = tf.nn.softplus( - math_ops.to_float(discriminator_real_outputs))
        discriminator_gen_outputs = tf.nn.softplus( math_ops.to_float(discriminator_gen_outputs))
        discriminator_real_outputs.shape.assert_is_compatible_with(
            discriminator_gen_outputs.shape)

        loss_on_generated = losses.compute_weighted_loss(
            discriminator_gen_outputs, generated_weights, scope,
            loss_collection=None, reduction=reduction)
        loss_on_real = losses.compute_weighted_loss(
            discriminator_real_outputs, real_weights, scope, loss_collection=None,
            reduction=reduction)
        loss = loss_on_generated + loss_on_real
        util.add_loss(loss, loss_collection)

        if add_summaries:
            summary.scalar('discriminator_gen_softplus_loss', loss_on_generated)
            summary.scalar('discriminator_real_softplus_loss', loss_on_real)
            summary.scalar('discriminator_softplus_loss', loss)
        return loss

softplus_generator_loss = _args_to_gan_model(
    _softplus_generator_loss)
softplus_discriminator_loss = _args_to_gan_model(
    _softplus_discriminator_loss)
