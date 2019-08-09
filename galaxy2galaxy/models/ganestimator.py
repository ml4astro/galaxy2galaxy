
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.estimator.python.gan_estimator_impl import _make_prediction_gan_model, _summary_type_map, _get_estimator_spec
from tensorflow.contrib.gan.python.losses.python.tuple_losses_impl import _args_to_gan_model
from tensorflow.python.summary import summary
from tensorflow.python.ops.losses import util
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.contrib.framework.python.ops import variables as variable_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.util import tf_inspect as inspect

from tensor2tensor.utils import t2t_model
from tensor2tensor.models import vanilla_gan
from tensor2tensor.layers.common_layers import lrelu
from tensor2tensor.utils import hparams_lib
from tensor2tensor.layers import common_layers

from galaxy2galaxy.utils import registry
from galaxy2galaxy.models.gan_utils import softplus_discriminator_loss, softplus_generator_loss, SperctraNormConstraint
from galaxy2galaxy.models.sngan import SnGAN


@registry.register_model
class GanEstimator(SnGAN):
    """ GAN based on tfgan estimator API
    """

    @classmethod
    def estimator_model_fn(cls,
                         hparams,
                         features,
                         labels,
                         mode,
                         config=None,
                         params=None,
                         decode_hparams=None,
                         use_tpu=False):

        if mode not in [model_fn_lib.ModeKeys.TRAIN, model_fn_lib.ModeKeys.EVAL,
                      model_fn_lib.ModeKeys.PREDICT]:
            raise ValueError('Mode not recognized: %s' % mode)

        if mode is model_fn_lib.ModeKeys.TRAIN:
            is_training = True
        hparams = hparams_lib.copy_hparams(hparams)

        # Instantiate model
        data_parallelism = None
        if not use_tpu and config:
            data_parallelism = config.data_parallelism
        reuse = tf.get_variable_scope().reuse

        # Instantiate model
        self = cls(
            hparams,
            mode,
            data_parallelism=data_parallelism,
            decode_hparams=decode_hparams,
            _reuse=reuse)


        real_data =  common_layers.convert_rgb_to_real(features['inputs'])  # rename inputs for clarity
        generator_inputs = tf.random_uniform([self.hparams.batch_size,
                                          self.hparams.bottleneck_bits],
                                          minval=-1, maxval=1, name="z")

         # rename inputs for clarity
        out_shape = common_layers.shape_list(real_data)[1:4]

        if mode == model_fn_lib.ModeKeys.PREDICT:
            if real_data is not None:
                raise ValueError('`labels` must be `None` when mode is `predict`. '
                           'Instead, found %s' % real_data)
            gan_model = _make_prediction_gan_model(generator_inputs,
                                                   partial(self.generator, is_training=is_training, out_shape=out_shape),
                                                   'Generator')
            # Here should be where we export the model as tf hub

        else:  # model_fn_lib.ModeKeys.TRAIN or model_fn_lib.ModeKeys.EVAL
            # Manual gan_model creation
            with tf.variable_scope('Generator') as gen_scope:
                generated_images = self.generator(generator_inputs, is_training=is_training, out_shape=out_shape)

            with tf.variable_scope('Discriminator') as dis_scope:
                discriminator_gen_outputs = self.discriminator(generated_images, is_training=is_training)

            with tf.variable_scope(dis_scope, reuse=True):
                discriminator_real_outputs = self.discriminator(real_data, is_training=is_training)

            generator_variables = variable_lib.get_trainable_variables(gen_scope)
            discriminator_variables = variable_lib.get_trainable_variables(dis_scope)

            gan_model = tfgan.GANModel(
                generator_inputs,
                generated_images,
                generator_variables,
                gen_scope,
                self.generator,
                real_data,
                discriminator_real_outputs,
                discriminator_gen_outputs,
                discriminator_variables,
                dis_scope,
                self.discriminator)

        opt_gen = tf.train.AdamOptimizer(hparams.learning_rate)
        opt_disc = tf.train.AdamOptimizer(hparams.learning_rate)

        # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
        # metrics, and optimizers (if required).
        return _get_estimator_spec(
          mode, gan_model, softplus_generator_loss, softplus_discriminator_loss,
          None, opt_gen, opt_disc, None, True)
