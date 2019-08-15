""" Spectral Norm GAN """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

from tensorflow_gan.python.estimator.gan_estimator import Optimizers, get_gan_model, get_train_estimator_spec, get_eval_estimator_spec, get_predict_estimator_spec
from tensorflow_gan.python import train as tfgan_train
from tensorflow_gan.python import namedtuples
from tensorflow_gan.python.estimator.gan_estimator import SummaryType
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers

from galaxy2galaxy.layers import spectral_ops as ops


class AbstractGAN(t2t_model.T2TModel):
  """ Base class for tf-gan based models
  """

  def generator(self, code, mode):
    raise NotImplementedError

  def discriminator(self, x, conditioning, mode):
    raise NotImplementedError

  def discriminator_loss_fn(self):
    raise NotImplementedError

  def generator_loss_fn(self):
    raise NotImplementedError

  @property
  def summaries(self):
    return [SummaryType.IMAGES]

  def sample_noise(self):
    p = self.hparams
    shape = [p.batch_size, p.bottleneck_bits]
    z = tf.random.normal(shape, name='z0', dtype=tf.float32)
    return z

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
    else:
      is_training = False

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

    generator_inputs = self.sample_noise()
    # rename inputs for clarity
    real_data = features['inputs']
    img_shape = common_layers.shape_list(real_data)[1:4]
    real_data.set_shape([hparams.batch_size]+img_shape)

    # To satify the TFGAN API setting real data to none on predict
    if mode == tf.estimator.ModeKeys.PREDICT:
      real_data =None

    optimizers = Optimizers(tf.compat.v1.train.AdamOptimizer(
          hparams.generator_lr, hparams.beta1),
          tf.compat.v1.train.AdamOptimizer(
          hparams.discriminator_lr, hparams.beta1)
          )

    # Creates tfhub modules for both generator and discriminator
    def make_discriminator_spec():
      input_layer = tf.placeholder(tf.float32, shape=[None] + img_shape)
      disc_output = self.discriminator(input_layer, None, mode)
      hub.add_signature(inputs=input_layer, outputs=disc_output)
    disc_spec = hub.create_module_spec(make_discriminator_spec)

    def make_generator_spec():
      input_layer = tf.placeholder(tf.float32, shape=[None] + common_layers.shape_list(generator_inputs)[1:])
      gen_output = self.generator(input_layer, mode)
      hub.add_signature(inputs=input_layer, outputs=gen_output)
    gen_spec = hub.create_module_spec(make_generator_spec)

    # Create the modules
    discriminator_module = hub.Module(disc_spec, name="Discriminator_Module", trainable=True)
    generator_module = hub.Module(gen_spec, name="Generator_Module", trainable=True)

    # Wraps the modules into functions expected by TF-GAN
    generator = lambda code, mode: generator_module(code)
    discriminator =  lambda image, conditioning, mode: discriminator_module(image)

    # Make GANModel, which encapsulates the GAN model architectures.
    gan_model = get_gan_model(mode,
                              generator,
                              discriminator,
                              real_data,
                              generator_inputs,
                              add_summaries=self.summaries)

    # Make GANLoss, which encapsulates the losses.
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      gan_loss = tfgan_train.gan_loss(
          gan_model,
          self.generator_loss,
          self.discriminator_loss,
          add_summaries=True)

    # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
    # metrics, and optimizers (if required).
    if mode == tf.estimator.ModeKeys.TRAIN:
      get_hooks_fn = tfgan_train.get_sequential_train_hooks(namedtuples.GANTrainSteps(hparams.gen_steps, hparams.disc_steps))
      estimator_spec = get_train_estimator_spec(gan_model, gan_loss, optimizers, get_hooks_fn,  is_chief=True)
    elif mode == tf.estimator.ModeKeys.EVAL:
      estimator_spec = get_eval_estimator_spec(gan_model, gan_loss)
    else:  # tf.estimator.ModeKeys.PREDICT
      # Register hub modules for export
      hub.register_module_for_export(generator_module, "generator")
      hub.register_module_for_export(discriminator_module, "discriminator")
      estimator_spec = get_predict_estimator_spec(gan_model)
    return estimator_spec
