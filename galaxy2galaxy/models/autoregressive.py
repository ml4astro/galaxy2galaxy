"""Autoregressive models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from pixel_cnn_pp.model import model_spec

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images

@registry.register_model
class Img2imgPixelCnn(t2t_model.T2TModel):

  def image_summary(self, name, image_logits, max_outputs=1, rows=8, cols=8):
    """Helper for image summaries that are safe on TPU."""
    if len(image_logits.get_shape()) != 4:
      tf.logging.info("Not generating image summary, maybe not an image.")
      return
    return tf.summary.image(
        name, pack_images(image_logits, rows, cols),
        #common_layers.tpu_safe_image_summary(pack_images(tensor, rows, cols)),
        max_outputs=max_outputs)

  def body(self, features):
    hparams = self.hparams
    model_opt = { 'nr_resnet': hparams.nr_resnet,
                  'nr_filters': hparams.hidden_size,
                  'nr_logistic_mix': 1,
                  'resnet_nonlinearity': 'concat_elu',
                  'energy_distance': False,
                  'dropout_p': hparams.dropout}

    def pixel_cnn_fn(input_layer):
      model = tf.make_template('model', model_spec)
      out = model(input_layer, None, ema=None, **model_opt)
      out = tf.layers.dense(out, 2, activation=None)
      loc, scale = tf.split(out, num_or_size_splits=2, axis=-1)
      scale = tf.nn.softplus(scale) + 1e-4
      distribution = tfp.distributions.Independent( tfp.distributions.Normal(loc=loc, scale=scale))
      sample = distribution.sample()
      log_prob = distribution.log_prob(input_layer)
      grads = tf.gradients(log_prob, input_layer)[0]
      print(grads)
      output = {'sample': sample, 'log_prob': log_prob,
                'logits':out, 'mu': loc, 'scale': scale, 'grads': grads}
      return output

    # During predict, we use a tf hub module, otherwise we stick to normal
    # behavior to allow for multi gpu training
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      def make_model_spec():
        input_layer = tf.placeholder(tf.float32, shape=features["inputs"].get_shape())
        outputs = pixel_cnn_fn(input_layer)
        hub.add_signature(inputs=input_layer, outputs=outputs)
      spec = hub.create_module_spec(make_model_spec, drop_collections=['checkpoints'])
      pixelcnn = hub.Module(spec, name="pixelcnn_module", trainable=True)
      hub.register_module_for_export(pixelcnn, "pixelcnn")
      output = pixelcnn(features["inputs"], as_dict=True)
    else:
      with tf.variable_scope('pixelcnn_module'):
        output = pixel_cnn_fn(features["inputs"])

    self.image_summary("inputs", features["inputs"])
    self.image_summary("samples", output["sample"])

    return output["logits"], {"training": - output["log_prob"]}

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """ TODO: Switch to parent inference function
    """
    return self(features)[0]

@registry.register_hparams
def pixelcnnpp_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 64
  hparams.batch_size = 16
  hparams.dropout = 0.5
  hparams.clip_grad_norm = 1.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.001
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 0.2
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.bottom["targets"] = modalities.identity_bottom
  hparams.top["targets"] = modalities.identity_top
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  # PixelCNN model opt
  hparams.add_hparam("nr_resnet", 2)
  hparams.add_hparam("num_channels", 1)
  return hparams
