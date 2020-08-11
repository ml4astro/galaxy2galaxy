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
import glob
import os


# HSC default pixel scale
_HSC_PIXEL_SCALE=0.168 #arcsec
# Path to sql files for HSC samples
_HSC_SAMPLE_SQL_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def _resize_image(im, size):
  centh = im.shape[0]/2
  centw = im.shape[1]/2
  lh, rh = int(centh-size/2), int(centh+size/2)
  lw, rw = int(centw-size/2), int(centw+size/2)
  cropped = im[lh:rh, lw:rw, :]
  assert cropped.shape[0]==size and cropped.shape[1]==size, f"Wrong size! Still {cropped.shape}"
  return cropped


@registry.register_problem
class Img2imgCandelsAll(astroimage_utils.AstroImageProblem):
  """Base class for image problems created from HSC Public Data Release.
  """

#   @property
#   def dataset_splits(self):
#     """Splits of data to produce and number of output shards for each.
#        Note that each shard will be produced in parallel.
#        We are going to split the GalSim data into shards of 1000 galaxies each,
#        with 80 shards for training, 2 shards for validation.
#     """
#     return [{
#         "split": problem.DatasetSplit.TRAIN,
#         "shards": 1,
#     }, {
#         "split": problem.DatasetSplit.EVAL,
#         "shards": 0,
#     }]

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.filters = ['wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f850w']
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf", None)

  @property
  def num_bands(self):
    """Number of bands."""
    p = self.get_hparams()
    return len(p.filters)

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """ Generator yielding individual postage stamps.
    """
    p = self.get_hparams()
    all_cat = Table.read(os.path.join(tmp_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='egs   ')] = 'EGS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='UDS   ')] = 'UDS'
    
    cube_psf = np.zeros((167, 167, len(p.filters)))
    
    for i, filter in enumerate(p.filters):
        cube_psf[:, :, i] = fits.open(tmp_dir + 'psf_' + filter +'.fits')[0].data

#     # Step 1: Maybe regenerate the dataset
#     self.maybe_build_dataset(tmp_dir)
#     import timeit

    # Step 2: Extract postage stamps, resize them to requested size
    ''' Genrate GDS and GDN'''
    for field in ['GDS', 'GDN'] :
        print(f"\n generating{field}\n")

        n=0
        index = 0
        im = np.zeros((128, 128, 5))
        sub_cat = all_cat[np.where(all_cat["FIELD_1"]==field)[0]]
        for i in sub_cat['RB_ID']:
                if i == index :
                    continue
                index = i
                print(field, i)
#             try:
                gal = glob.glob(os.path.join(tmp_dir, field, p.filters[0])+'/galaxy_'+str(index)+'_*')[0]
                im[:,:, 0] = fits.open(gal)[0].data
                for i, filter in enumerate(p.filters[1:4]):
                    try :
                        tmp_file = glob.glob(os.path.join(tmp_dir, field, filter)+'/galaxy_'+str(index)+'_*')[0]
                        im[:,:, i+1] = fits.open(tmp_file)[0].data
                    except Exception:
                        print('Galaxy not seen in every filter')
                        continue
                im = _resize_image(im, p.img_len)

                n+=1
#                 attributes = {k: all_cat[k][0] for k in p.attributes}
                flag = [1, 1, 1, 1, 0]


    #           imCp = np.fft.fft2(psf, s=stamp_size, stamp_size)
                ps = np.random.normal(0, 1e-4, (p.img_len, p.img_len))
                im_psf = cube_psf

#                 serialized_output = {"image/encoded": [im.tostring()],
#                 "image/format": ["raw"],
#                 "psf/encoded": [im_psf.tostring()],
#                 "psf/format": ["raw"],
#                 "ps/encoded": [ps.tostring()],
#                 "ps/format": ["raw"],
#                 "flag/filters":flag}
                
                serialized_output = {"image/encoded": [im.tostring()],
                "image/format": ["raw"],
                "psf/encoded": [im_psf.tostring()],
                "psf/format": ["raw"]}
        
        
                if n > 20:
                    break
                yield serialized_output
#             except Exception:
#                 print('No image corresponding to the Index')
#                 n+=1
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

#         "ps/encoded": tf.FixedLenFeature((), tf.string),
#         "ps/format": tf.FixedLenFeature((), tf.string),
    }


    # Adds additional attributes to be decoded as specified in the configuration
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_fields['attrs/'+k] = tf.FixedLenFeature([], tf.float32, -1)

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

#         "ps": tf.contrib.slim.tfexample_decoder.Image(
#                 image_key="ps/encoded",
#                 format_key="ps/format",
#                 channels=self.num_bands,
#                 shape=[p.img_len, p.img_len // 2 + 1],
#                 dtype=tf.float32),
    }
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)

@registry.register_problem
class Img2imgCandelsGoods(astroimage_utils.AstroImageProblem):
  """Base class for image problems created from HSC Public Data Release.
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
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.filters = ['wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w']#, 'acs_f850w']
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf", None)

  @property
  def num_bands(self):
    """Number of bands."""
    p = self.get_hparams()
    return len(p.filters)

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """ Generator yielding individual postage stamps.
    """
    p = self.get_hparams()
    all_cat = Table.read(os.path.join(tmp_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    
    cube_psf = np.zeros((167, 167, len(p.filters)))
    
    for i, filter in enumerate(p.filters):
        cube_psf[:, :, i] = fits.open(tmp_dir + 'psf_' + filter +'.fits')[0].data

#     # Step 1: Maybe regenerate the dataset
#     self.maybe_build_dataset(tmp_dir)
#     import timeit

    # Step 2: Extract postage stamps, resize them to requested size
    ''' Genrate GDS and GDN'''
    for field in ['GDS', 'GDN'] :
        print(f"\n generating{field}\n")
        n=0
        index = 0
        im = np.zeros((128, 128, len(p.filters)))
        sub_cat = all_cat[np.where(all_cat["FIELD_1"]==field)[0]]
        for i in sub_cat['RB_ID']:
                if i == index :
                    continue
                index = i
                print(field, i)
#             try:
                for i, filter in enumerate(p.filters):
#                     try :
                        tmp_file = glob.glob(os.path.join(tmp_dir, field, filter)+'/galaxy_'+str(index)+'_*')[0]
                        im[:,:, i] = fits.open(tmp_file)[0].data
#                     except Exception:
#                         print('Galaxy not seen in every filter')
#                         continue
                im = _resize_image(im, p.img_len)

                n+=1
                if hasattr(p, 'attributes'):
                    attributes = {k: float(all_cat[k][index]) for k in p.attributes}
                    print(attributes)
                else:
                    attributes=None


    #           imCp = np.fft.fft2(psf, s=stamp_size, stamp_size)
                ps = np.random.normal(0, 1e-4, (p.img_len, p.img_len))
                im_psf = cube_psf
                im_psf = _resize_image(im_psf, p.img_len)

                print(f"\n shapes : {im.shape}, {im_psf.shape}, !!!!! \n")

                serialized_output = {"image/encoded": [im.tostring()],
                "image/format": ["raw"],
                "psf/encoded": [im_psf.tostring()],
                "psf/format": ["raw"],
                "ps/encoded": [ps.tostring()],
                "ps/format": ["raw"]}
                
                if attributes is not None:
                    for k in attributes:
                        serialized_output['attrs/'+k] = [attributes[k]]
                if n > 5:
                    break
                yield serialized_output
#             except Exception:
#                 print('No image corresponding to the Index')
#                 n+=1

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image = example["inputs"]

    # Clip to 1 the values of the image
    # image = tf.clip_by_value(image, -1, 1)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image
    example["targets"] = image
    return example

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

        "ps/encoded": tf.FixedLenFeature((), tf.string),
        "ps/format": tf.FixedLenFeature((), tf.string),
    }


    # Adds additional attributes to be decoded as specified in the configuration
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_fields['attrs/'+k] = tf.FixedLenFeature([], tf.float32, -1)

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

        "ps": tf.contrib.slim.tfexample_decoder.Image(
                image_key="ps/encoded",
                format_key="ps/format",
                channels=self.num_bands,
                shape=[p.img_len, p.img_len // 2 + 1],
                dtype=tf.float32),
    }
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)

    return data_fields, data_items_to_decoders
#     END: Subclass interface
    
    
  @property
  def is_generate_per_split(self):
    return False


@registry.register_problem
class Attrs2imgCandelsAll64Euclid(Img2imgCandelsAll):
  """
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.1
    p.img_len = 128
    p.example_per_shard = 1
    p.filters = ['wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f850w']

    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = []#['mag', 're', 'q']
    
    
@registry.register_problem
class Attrs2imgCandelsGoods64Euclid(Img2imgCandelsGoods):
  """
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.1
    p.img_len = 128
    p.example_per_shard = 1
    p.filters = ['wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w']#, 'acs_f850l']

    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
#     p.attributes = ['mag', 're', 'q']
    
