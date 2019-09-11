""" Utilities for astronomical images problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

import tensorflow as tf



from tensorflow.contrib.slim.python.slim.data import data_decoder
from tensorflow.contrib.slim.python.slim.data.tfexample_decoder import ItemHandler
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops

class Image(ItemHandler):
  """ Item Handler copied from
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/tfexample_decoder.py
  There is a bug in tf 1.14 that prevents reading raw images with arbitry numbers
  of channels. This is has a fix for that.
  TODO: Switch back to upstream data handler code at next release
  """

  def __init__(self,
               image_key=None,
               format_key=None,
               shape=None,
               channels=3,
               dtype=dtypes.uint8,
               repeated=False,
               dct_method=''):
    """Initializes the image.
    Args:
      image_key: the name of the TF-Example feature in which the encoded image
        is stored.
      format_key: the name of the TF-Example feature in which the image format
        is stored.
      shape: the output shape of the image as 1-D `Tensor`
        [height, width, channels]. The image is reshaped
        accordingly.
      channels: the number of channels in the image.
      dtype: images will be decoded at this bit depth. Different formats
        support different bit depths.
          See tf.image.decode_image,
              tf.io.decode_raw,
      repeated: if False, decodes a single image. If True, decodes a
        variable number of image strings from a 1D tensor of strings.
      dct_method: An optional string. Defaults to empty string. It only takes
        effect when image format is jpeg, used to specify a hint about the
        algorithm used for jpeg decompression. Currently valid values
        are ['INTEGER_FAST', 'INTEGER_ACCURATE']. The hint may be ignored, for
        example, the jpeg library does not have that specific option.
    """
    if not image_key:
      image_key = 'image/encoded'
    if not format_key:
      format_key = 'image/format'

    super(Image, self).__init__([image_key, format_key])
    self._image_key = image_key
    self._format_key = format_key
    self._shape = shape
    self._channels = channels
    self._dtype = dtype
    self._repeated = repeated
    self._dct_method = dct_method

  def tensors_to_item(self, keys_to_tensors):
    """See base class."""
    image_buffer = keys_to_tensors[self._image_key]
    image_format = keys_to_tensors[self._format_key]

    if self._repeated:
      return map_fn.map_fn(lambda x: self._decode(x, image_format),
                           image_buffer, dtype=self._dtype)
    else:
      return self._decode(image_buffer, image_format)

  def _decode(self, image_buffer, image_format):
    """Decodes the image buffer.
    Args:
      image_buffer: The tensor representing the encoded image tensor.
      image_format: The image format for the image in `image_buffer`. If image
        format is `raw`, all images are expected to be in this format, otherwise
        this op can decode a mix of `jpg` and `png` formats.
    Returns:
      A tensor that represents decoded image of self._shape, or
      (?, ?, self._channels) if self._shape is not specified.
    """
    # TODO: Assert that the image format is raw
    image = parsing_ops.decode_raw(image_buffer, out_type=self._dtype)
    image = array_ops.reshape(image, self._shape)
    return image

class AstroImageProblem(problem.Problem):
  """Base class for image problems on astronomical images.
  """

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 64

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel if parallel generation
       is used
    """
    return [{
                "split": problem.DatasetSplit.TRAIN,
                "shards": 8,
            }, {
                "split": problem.DatasetSplit.EVAL,
                "shards": 2,
            }]


  @property
  def is_generate_per_split(self):
    """A single call to `generate_samples` generates for all `dataset_splits`.

    Set to True if you already have distinct subsets of data for each dataset
    split specified in `self.dataset_splits`. `self.generate_samples` will be
    called once for each split.

    Set to False if you have a unified dataset that you'd like to have split out
    into training and evaluation data automatically. `self.generate_samples`
    will be called only once and the data will be sharded across the dataset
    splits specified in `self.dataset_splits`.

    Returns:
      bool
    """
    raise NotImplementedError

  @property
  def num_bands(self):
    """Number of bands."""
    raise NotImplementedError

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """Generate examples of images.

    Each yielded dict will be made into a single example.

    This method is typically called once per split in `self.dataset_splits`
    unless `self.is_generate_per_split=False`.

    Args:
      data_dir: final data directory.
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).
      task_id: number of the task, if parallel execution.

    Yields:
      dict
    """
    raise NotImplementedError

  def prepare_to_generate(self, data_dir, tmp_dir):
    """Prepare to generate data in parallel on different processes.

    This function is called if multiprocess_generate is True.

    Some things that might need to be done once are downloading the data
    if it is not yet downloaded, and building the vocabulary.

    Args:
      data_dir: a string
      tmp_dir: a string
    """
    pass

  @property
  def multiprocess_generate(self):
    """Whether to generate the data in multiple parallel processes."""
    return False

  @property
  def already_shuffled(self):
    return False
  # END: Subclass interface

  @property
  def num_train_shards(self):
    return self.dataset_splits[0]["shards"]

  @property
  def num_dev_shards(self):
    return self.dataset_splits[1]["shards"]

  @property
  def num_generate_tasks(self):
    """Needed if multiprocess_generate is True."""
    return self.num_train_shards + self.num_dev_shards

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

    # In case of parallel execution, each shard is generated by a different
    # process
    if self.multiprocess_generate:
      tf.logging.info("generate_data task_id=%s" % task_id)
      assert task_id >= 0 and task_id < self.num_generate_tasks
      if task_id < self.num_train_shards:
        out_file = self.training_filepaths(
            data_dir, self.num_train_shards, shuffled=False)[task_id]
        dataset_split = problem.DatasetSplit.TRAIN
      else:
        out_file = self.dev_filepaths(
            data_dir, self.num_dev_shards,
            shuffled=False)[task_id - self.num_train_shards]
        dataset_split = problem.DatasetSplit.EVAL
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, dataset_split, task_id), [out_file])
      generator_utils.shuffle_dataset([out_file])
    else:
      filepath_fns = {
            problem.DatasetSplit.TRAIN: self.training_filepaths,
            problem.DatasetSplit.EVAL: self.dev_filepaths,
            problem.DatasetSplit.TEST: self.test_filepaths,
            }

      split_paths = [(split["split"], filepath_fns[split["split"]](
            data_dir, split["shards"], shuffled=self.already_shuffled))
                       for split in self.dataset_splits]

      all_paths = []
      for _, paths in split_paths:
        all_paths.extend(paths)

      if self.is_generate_per_split:
        for split, paths in split_paths:
          generator_utils.generate_files(
              self.generator(data_dir, tmp_dir, split), paths)
      else:
        generator_utils.generate_files(
            self.generator(
                data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

      generator_utils.shuffle_dataset(all_paths)

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
    }

    # Adds additional attributes to be decoded as specified in the configuration
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_fields['attrs/'+k] = tf.FixedLenFeature([], tf.float32, -1)

    data_items_to_decoders = {
        "inputs": Image( # TODO: switch back to tf.contrib.slim.tfexample_decoder.
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),
    }

    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = []
    return eval_metrics

  @property
  def decode_hooks(self):
    return [image_utils.convert_predictions_to_image_summaries]
