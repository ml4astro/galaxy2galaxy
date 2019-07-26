"""HST/ACS Cosmos generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import galsim_utils

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
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
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
    """
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
    p.img_len = 64

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
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),

        "psf": tf.contrib.slim.tfexample_decoder.Image(
                image_key="psf/encoded",
                format_key="psf/format",
                channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),
    }

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.RMSE]
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
  25.2 sample.
  """
  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
       We are going to split the GalSim data into shards of 1000 galaxies each,
       with 80 shards for training, 2 shards for validation.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 80,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 2,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.img_len = 64
    p.example_per_shard = 1000

  @property
  def num_bands(self):
    """Number of bands."""
    return 1

  def generator(self, tmp_dir, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog()
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'/COSMOS_25.2_training_sample')

    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog.getNObjects()))

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale)

      # We apply the orginal psf
      psf = gal.original_psf

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale)

@registry.register_problem
class GalsimCosmos32(GalsimCosmos):

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.pixel_scale = 0.06
    p.img_len = 32
    p.example_per_shard = 1000

@registry.register_problem
class Img2imgGalsimCosmos32(GalsimCosmos32):

  def preprocess_example(self, example, unused_mode, unused_hparams):
    image = example["inputs"]

    # Clip to 1 the values of the image
    image = tf.clip_by_value(image, -1, 1)

    example["inputs"] = image
    example["targets"] = image
    return example

@registry.register_problem
class GalsimCosmosParametric(GalsimCosmos):

  def generator(self, tmp_dir, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    catalog_real = galsim.COSMOSCatalog(dir=tmp_dir+'/COSMOS_25.2_training_sample')
    catalog_param = galsim.COSMOSCatalog(ir=tmp_dir+'/COSMOS_25.2_training_sample', use_real=False)
    
    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog_param.getNObjects()))

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here (can be used with parametric galaxies) Do we need to add noise ?
      gal_real = catalog_real.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale)
      gal_param = catalog_param.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale)

      # Generate parameter dict.
      param_dict = catalog_param.getParametricRecord(ind)

      # We apply the orginal psf / cannot be used with parametric galaxies, use a gaussian profile ?
      psf = gal_real.original_psf

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal_param, psf,
                                               stamp_size=p.img_len, pixel_scale=p.pixel_scale)
