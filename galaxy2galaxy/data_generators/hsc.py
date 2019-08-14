""" HSC Datasets """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import hsc_utils
from . import astroimage_utils

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities, common_layers
from tensor2tensor.utils import metrics

from galaxy2galaxy.utils import registry

from scipy.ndimage import gaussian_filter

import tensorflow as tf
import numpy as np
import fits2hdf.pyhdfits as fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb
import h5py
import os

# HSC default pixel scale TODO: Check what's the correct scale
_HSC_PIXEL_SCALE=0.17501 #arcsec
# Path to sql files for HSC samples
_HSC_SAMPLE_SQL_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hsc_utils')

def _resize_image(im, size):
  centh = im.shape[0]/2
  centw = im.shape[1]/2
  lh, rh = int(centh-size/2), int(centh+size/2)
  lw, rw = int(centw-size/2), int(centw+size/2)
  cropped = im[lh:rh, lw:rw, :]
  assert cropped.shape[0]==size and cropped.shape[1]==size, f"Wrong size! Still {cropped.shape}"
  return cropped

@registry.register_problem
class Img2imgHSC(astroimage_utils.AstroImageProblem):
  """Base class for image problems created from HSC Public Data Release.
  """

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 64
    p.filters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z']
    p.sql_file = os.path.join(_HSC_SAMPLE_SQL_DIR, 'hsc_pdr2_wide_img2img.sql')
    p.data_release = 'pdr2'
    p.rerun = 'pdr2_wide'

  @property
  def num_bands(self):
    """Number of bands."""
    p = self.get_hparams()
    return len(p.filters)

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """ Generator yielding individual postage stamps.
    """
    p = self.get_hparams()

    # Step 1: Maybe regenerate the dataset
    self.maybe_build_dataset(tmp_dir)
    import timeit

    # Step 2: Extract postage stamps, resize them to requested size
    catalog = Table.read(os.path.join(tmp_dir, 'catalog.fits'))
    with h5py.File(os.path.join(tmp_dir, 'cutouts.hdf'),'r') as cutouts:
        # Loop through the examples, resize cutout to desired size
        for row in catalog:
          cutout = cutouts[str(row['object_id'])]
          im = [cutout[f]['HDU0']['DATA'][:] for f in p.filters]

          try:
              im = np.stack(im, axis=-1).astype('float32')
          except:
              print('Failure to stack bands', [i.shape for i in im])
              continue

          # Images may not have exactly the right number of pixels
          im = _resize_image(im, p.img_len)

          serialized_output = {"image/encoded": [im.tostring()],
                               "image/format": ["raw"]}

          # If attributes are requested, let's add them to the dataset
          if hasattr(p, 'attributes'):
            for k in p.attributes:
              serialized_output['attrs/'+k] = [np.asscalar(row[k])]

          yield serialized_output

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image = example["inputs"]

    # TODO: apply some pre-processing/normalization to the images

    if hasattr(p, 'attributes'):
      example["attributes"] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image
    example["targets"] = image
    return example

  # END: Subclass interface
  @property
  def is_generate_per_split(self):
    return False

  def maybe_build_dataset(self, tmp_dir):
    """ Check that the HSC dataset is downloaded and ready in tmp_dir, otherwise
    redownload it from the server.
    """
    p = self.get_hparams()
    if (not os.path.isfile(os.path.join(tmp_dir, 'catalog.fits')) or
        not os.path.isfile(os.path.join(tmp_dir, 'cutouts.hdf')) ):
      hsc_utils.build_hsc_sample(p.sql_file,
                                 out_dir=tmp_dir,
                                 tmp_dir=os.path.join(tmp_dir,'tmp'),
                                 cutout_size=p.img_len*_HSC_PIXEL_SCALE/2,
                                 filters=p.filters,
                                 data_release=p.data_release,
                                 rerun=p.rerun)

@registry.register_problem
class Img2imgHSCAnomaly(Img2imgHSC):
  """ Dataset for anomaly detection on HSC data.
  """

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.filters = ['HSC-G', 'HSC-R', 'HSC-I']
    p.sql_file = os.path.join(_HSC_SAMPLE_SQL_DIR, 'hsc_pdr2_wide_anomaly.sql')
    p.data_release = 'pdr2'
    p.rerun = 'pdr2_wide'
    p.attributes = ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag']
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Luptonize the examples, so that we can use t2t models easily
    """
    p = self.get_hparams()
    image = example["inputs"]

    # Apply Luptonic Asinh stretch, and return uint8 rgb images
    def my_func(x):
      return make_lupton_rgb(x[...,2], x[...,1], x[...,0], Q=15, stretch=0.5, minimum=0)

    int_image = tf.py_func(my_func, [image], tf.uint8)
    int_image.set_shape(image.shape)
    image = common_layers.convert_rgb_to_symmetric_real(int_image)

    if hasattr(p, 'attributes'):
      example["attributes"] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image
    example["targets"] = image
    return example
