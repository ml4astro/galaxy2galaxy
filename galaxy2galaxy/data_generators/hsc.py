""" HSC Datasets """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import hsc_utils
from . import astro_image_utils

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

from galaxy2galaxy.utils import registry

import tensorflow as tf
import numpy as np
import fits2hdf.pyhdfits as fits
import h5py
import os

# HSC default pixel scale TODO: Check what's the correct scale
_HSC_PIXEL_SCALE=0.2 #arcsec
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
class HSCProblem(astro_image_utils.AstroImageProblem):
  """Base class for image problems created from HSC Public Data Release.
  """

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 64
    p.filters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z']
    p.sql_file = os.path.join(_HSC_SAMPLE_SQL_DIR, 'hsc_pdr2_anomaly_test.sql')
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

    # Step 2: Extract postage stamps, resize them to requested size
    with h5py.File(os.path.join(tmp_dir, 'cutouts.hdf'),'r') as cutouts:
        # Loop through the examples, resize cutout to desired size
        for object_id in cutouts.keys():
          cutout = cutouts[object_id]
          im = [fits.open(cutout[f])[3].data for f in p.filters]
          im = np.stack(im, axis=-1).astype('float32')
          # Images may not have exactly the right number of pixels
          im = _resize_image(im, p.img_len)

          yield {"image/encoded": [im.tostring()],
                 "image/format": ["raw"]}

  # END: Subclass interface
  @property
  def is_generate_per_split(self):
    return False

  def maybe_build_dataset(self, tmp_dir):
    """ Check that the HSC dataset is downloaded and ready in tmp_dir, otherwise
    redownload it from the server.
    """
    p = self.get_hparams()
    if (not os.path.isfile(os.path.join(tmp_dir, 'catalog.fits')) and
        not os.path.isfile(os.path.join(tmp_dir, 'cutouts.hdf')) ):
      hsc_utils.build_hsc_sample(p.sql_file,
                                 out_dir=tmp_dir,
                                 tmp_dir=os.path.join(tmp_dir,'tmp'),
                                 cutout_size=p.img_len*_HSC_PIXEL_SCALE/2,
                                 filters=p.filters,
                                 data_release=p.data_release,
                                 rerun=p.rerun)


@registry.register_problem
class HSCAnomaly(HSCProblem):
  """ Dataset for anomaly detection on HSC data.
  """

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 96
    p.filters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z']
    p.sql_file = os.path.join(_HSC_SAMPLE_SQL_DIR, 'hsc_pdr2_anomaly.sql')
    p.data_release = 'pdr2'
    p.rerun = 'pdr2_wide'
