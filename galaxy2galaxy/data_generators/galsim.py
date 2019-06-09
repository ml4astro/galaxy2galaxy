"""HST/ACS Cosmos generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import galsim_utils

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

import tensorflow as tf
import galsim


class GalsimProblem(problem.Problem):
  """Base class for image problems generated with GalSim.

  Subclasses need only implement the `galsim_generator` function used to draw
  postage stamps with GalSim.
  """

  # START: Subclass interface
  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.stamp_size = 64

  @property
  def num_bands(self):
    """Number of bands."""
    return 1

  def generator(self, tmp_dir, task_id=-1):
    """
    Function to implement to generate galaxy postage stamps
    """
    raise NotImplementedError
  # END: Subclass interface

  @property
  def num_train_shards(self):
    return self.dataset_splits[0]["shards"]

  @property
  def num_dev_shards(self):
    return self.dataset_splits[1]["shards"]

  def example_reading_spec(self):
    """Define how data is serialized to file and read back.

    Returns:
      data_fields: A dictionary mapping data names to its feature type.
      data_items_to_decoders: A dictionary mapping data names to TF Example
         decoders, to be used when reading back TF examples from disk.
    """
    p = self.get_hparams()

    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),

        "psf/encoded": tf.FixedLenFeature((), tf.string),
        "psf/format": tf.FixedLenFeature((), tf.string),
    }

    data_items_to_decoders = {
        "inputs": tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_bands,
                #shape=[p.pixel_size, p.pixel_size, self.num_channels],
                dtype=tf.float32),

        "psf": tf.contrib.slim.tfexample_decoder.Image(
                image_key="psf/encoded",
                format_key="psf/format",
                channels=self.num_bands,
                #shape=[p.pixel_size, p.pixel_size, self.num_channels],
                dtype=tf.float32),
    }

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.IMAGE_RMSE, metrics.Metrics.IMAGE_SUMMARY]
    return eval_metrics

  @property
  def decode_hooks(self):
    return [image_utils.convert_predictions_to_image_summaries]

  @property
  def multiprocess_generate(self):
    """Whether to generate the data in multiple parallel processes."""
    return True

  @property
  def num_generate_tasks(self):
    """Needed if multiprocess_generate is True."""
    return self.num_train_shards + self.num_dev_shards

  def prepare_to_generate(self, data_dir, tmp_dir):
    """Prepare to generate data in parallel on different processes.

    This function is called if multiprocess_generate is True.

    Some things that might need to be done once are downloading the data
    if it is not yet downloaded, and building the vocabulary.

    Args:
      data_dir: a string
      tmp_dir: a string
    """
    galsim_utils.maybe_download_cosmos(tmp_dir)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """
    Generates training/dev data.

    Args:
      data_dir: a string
      tmp_dir: a string
      task_id: an optional integer
    Returns:
      shard or shards for which data was generated.
    """
    tf.logging.info("generate_data task_id=%s" % task_id)
    assert task_id >= 0 and task_id < self.num_generate_tasks
    if task_id < self.num_train_shards:
      out_file = self.training_filepaths(
          data_dir, self.num_train_shards, shuffled=False)[task_id]
    else:
      out_file = self.dev_filepaths(
          data_dir, self.num_dev_shards,
          shuffled=False)[task_id - self.num_train_shards]
    generator_utils.generate_files(
        self.generator(tmp_dir, task_id), [out_file])
    generator_utils.shuffle_dataset([out_file])


@registry.register_problem
class GalsimCosmos(GalsimProblem):
  """
  Subclass of GalSim problem implementing drawing galaxies from the COSMOS
  dataset.
  """

  def galsim_generator(self, tmp_dir, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    catalog = galsim.COSMOSCatalog(dir=tmp_dir)
    index = [1, 2, 3]

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.galaxy(index, noise_pad_size=p.stamp_size * p.pixel_scale)

      psf = gal.original_psf

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.stamp_size,
                                               pixel_scale=p.pixel_scale)
