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

import tensorflow as tf

@registry.register_model
class ContinuousAutoencoderBasic(t2t_model.T2TModel):
  """A basic autoencoder with physical reconstruction loss.

  This class closely follows tensor2tensor.models.autoencoders.AutoencoderBasic
  but removes the embedding, the GAN loss, and the cross-entropy loss.
  """

  def __init__(self, *args, **kwargs):
    super(ContinuousAutoencoderBasic, self).__init__(*args, **kwargs)
    self._cur_bottleneck_tensor = None
    self._encode_on_predict = False

  @property
  def num_channels(self):
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    return num_channels

  def image_summary(self, name, image, max_outputs=1):
    """Helper for image summaries that are safe on TPU."""
    if len(image.get_shape()) != 4:
      tf.logging.info("Not generating image summary, maybe not an image.")
      return
    return tf.summary.image(
        name,
        common_layers.tpu_safe_image_summary(image),
        max_outputs=max_outputs)

  def bottleneck(self, x):
    with tf.variable_scope("bottleneck"):
      hparams = self.hparams
      x = tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck")
      if hparams.mode == tf.estimator.ModeKeys.TRAIN:
        noise = 2.0 * tf.random_uniform(common_layers.shape_list(x)) - 1.0
        return tf.tanh(x) + noise * hparams.bottleneck_noise, 0.0
      return tf.tanh(x), 0.0

  def unbottleneck(self, x, res_size, reuse=None):
    with tf.variable_scope("unbottleneck", reuse=reuse):
      x = tf.layers.dense(x, res_size, name="dense")
      return x

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      layers = []
      kernel, strides = self._get_kernel_and_strides()
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        x = common_layers.make_even_size(x)
        layers.append(x)
        x = tf.layers.conv2d(
            x,
            hparams.hidden_size * 2**(i + 1),
            kernel,
            strides=strides,
            padding="SAME",
            activation=common_layers.belu,
            name="conv_%d" % i)
        x = common_layers.layer_norm(x, name="ln_%d" % i)
      return x, layers

  def decoder(self, x, encoder_layers):
    del encoder_layers
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
        j = hparams.num_hidden_layers - i - 1
        x = tf.layers.conv2d_transpose(
            x,
            hparams.hidden_size * 2**j,
            kernel,
            strides=strides,
            padding="SAME",
            activation=common_layers.belu,
            name="deconv_%d" % j)
        x = common_layers.layer_norm(x, name="ln_%d" % i)
      return x


  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    encoder_layers = None

    if (hparams.mode != tf.estimator.ModeKeys.PREDICT
        or self._encode_on_predict):

      x = features["inputs"]
      y = features["targets"]
      shape = common_layers.shape_list(x)
      print(x,y)
      # Run encoder.
      x, encoder_layers = self.encoder(x)
      # Bottleneck.
      b, b_loss = self.bottleneck(x)
      xb_loss = 0.0
      b_shape = common_layers.shape_list(b)
      self._cur_bottleneck_tensor = b
      res_size = common_layers.shape_list(x)[-1]
      b = self.unbottleneck(b, res_size)
      if not is_training:
        x = b
      else:
        l = 2**hparams.num_hidden_layers
        warm_step = int(hparams.bottleneck_warmup_steps * 0.25 * l)
        nomix_p = common_layers.inverse_lin_decay(warm_step) + 0.01
        if common_layers.should_generate_summaries():
          tf.summary.scalar("nomix_p_bottleneck", nomix_p)
        rand = tf.random_uniform(common_layers.shape_list(x))
        # This is the distance between b and x. Having this as loss helps learn
        # the bottleneck function, but if we back-propagated to x it would be
        # minimized by just setting x=0 and b=0 -- so we don't want too much
        # of the influence of this, and we stop-gradient to not zero-out x.
        x_stop = tf.stop_gradient(x)
        xb_loss = tf.reduce_mean(tf.reduce_sum(
            tf.squared_difference(x_stop, b), axis=-1))
        # To prevent this loss from exploding we clip at 1, but anneal clipping.
        clip_max = 1.0 / common_layers.inverse_exp_decay(
            warm_step, min_value=0.001)
        xb_clip = tf.maximum(tf.stop_gradient(xb_loss), clip_max)
        xb_loss *= clip_max / xb_clip
        x = tf.where(tf.less(rand, nomix_p), b, x)
    else:
      if self._cur_bottleneck_tensor is None:
        b = self.sample()
      else:
        b = self._cur_bottleneck_tensor
      self._cur_bottleneck_tensor = b
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x, encoder_layers)

    # Cut to the right size and mix before returning.
    res = x
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      res = x[:, :shape[1], :shape[2], :]

    # Final dense layer.
    reconstr = tf.layers.dense(res, self.num_channels, name="res_dense")

    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      return reconstr, {"bottleneck_loss": 0.0}

    # Losses.
    losses = {
        "bottleneck_extra": b_loss,
        "bottleneck_l2": hparams.bottleneck_l2_factor * xb_loss
    }

    targets_loss = tf.losses.mean_squared_error(y, reconstr)
    losses["training"] = targets_loss
    print(reconstr)
    self.image_summary("ae", reconstr)

    return reconstr, losses

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 2**hp.num_hidden_layers
    size = [
        hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
        hp.bottleneck_bits
    ]
    size = size if shape is None else shape
    # Sample in [-1, 1] as the bottleneck is under tanh.
    return 2.0 * tf.random_uniform(size) - 1.0

  def encode(self, x):
    """Auto-encode x and return the bottleneck."""
    features = {"targets": x}
    self(features)  # pylint: disable=not-callable
    res = tf.maximum(0.0, self._cur_bottleneck_tensor)  # Be 0/1 and not -1/1.
    self._cur_bottleneck_tensor = None
    return res

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    """Produce predictions from the model by sampling."""
    del args, kwargs
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Sample and decode.
    num_channels = self.num_channels
    if "targets" not in features:
      features["targets"] = tf.zeros(
          [self.hparams.batch_size, 1, 1, num_channels], dtype=tf.int32)
    samples, _ = self(features)  # pylint: disable=not-callable

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return samples.
    return samples

  def decode(self, bottleneck):
    """Auto-decode from the bottleneck and return the result."""
    # Get the shape from bottleneck and num channels.
    shape = common_layers.shape_list(bottleneck)
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    dummy_targets = tf.zeros(shape[:-1] + [num_channels])
    # Set the bottleneck to decode.
    if len(shape) > 4:
      bottleneck = tf.squeeze(bottleneck, axis=[1])
    bottleneck = 2 * bottleneck - 1  # Be -1/1 instead of 0/1.
    self._cur_bottleneck_tensor = bottleneck
    # Run decoding.
    res = self.infer({"targets": dummy_targets})
    self._cur_bottleneck_tensor = None
    return res

  def _get_kernel_and_strides(self):
    hparams = self.hparams
    kernel = (hparams.kernel_height, hparams.kernel_width)
    strides = (2, 2)
    return (kernel, strides)

@registry.register_hparams
def continuous_autoencoder_basic():
  """Basic autoencoder model."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 5
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.dropout = 0.05
  hparams.add_hparam("max_hidden_size", 1024)
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_shared_bits", 0)
  hparams.add_hparam("bottleneck_shared_bits_start_warmup", 0)
  hparams.add_hparam("bottleneck_shared_bits_stop_warmup", 0)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("bottleneck_warmup_steps", 2000)
  hparams.add_hparam("sample_height", 32)
  hparams.add_hparam("sample_width", 32)
  hparams.add_hparam("discriminator_batchnorm", True)
  hparams.add_hparam("num_sliced_vecs", 20000)
  hparams.add_hparam("sliced_do_tanh", int(True))
  hparams.add_hparam("code_loss_factor", 1.0)
  hparams.add_hparam("gan_codes_warmup_steps", 16000)
  hparams.add_hparam("gan_loss_factor", 0.0)
  hparams.add_hparam("bottleneck_l2_factor", 0.05)
  return hparams
