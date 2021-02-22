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
import sys
from skimage.transform import resize
from scipy.ndimage import binary_dilation  # type: ignore
from astropy.table import Table
from scipy.ndimage import rotate
from scipy.spatial import KDTree
import sep


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
class Img2imgCandelsOnefilter(astroimage_utils.AstroImageProblem):
  """ Base class for image problems created from CANDELS GOODS (North and South) fields, in 7 bands.
  """

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
      .
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 0,
    }]

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf", None)
    
  @property
  def num_bands(self):
    """Number of bands."""
    return 1

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """ 
    Generator yielding individual postage stamps.
    """
    
    p = self.get_hparams()
    
    '''Load the catalogue containing every fields and every filter'''
    all_cat = Table.read(os.path.join(tmp_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='egs   ')] = 'EGS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='UDS   ')] = 'UDS'
    
    ''' Load the psfs for each filter and resize'''
    psf = np.expand_dims(fits.open(tmp_dir + '/psfs/psf_acs_f606w.fits')[0].data, -1)
    psf = _resize_image(psf, p.img_len)
    # Step 2: Extract postage stamps, resize them to requested size
    ''' Loop on the fields'''
    for n_field, field in enumerate(['GDS', 'GDN', 'EGS', 'UDS', 'COSMOS']):
        
        print(f"\n generating{field}\n")
        n_gal_creat = 0
        index = 0
        ''' Create a subcat containing only the galaxies (in every filters) of the current field'''
        sub_cat = all_cat[np.where(all_cat["FIELD_1"]==field)[0]]

        ''' Loop on all the galaxies of the field '''
        for gal in sub_cat['RB_ID']:
                print(f"\n generating{gal}\n")

                if gal == index :     # To take care of the redudency inside the cat
                    continue
                index = gal

#             try:

                tmp_file = glob.glob(os.path.join(tmp_dir, field, 'acs_f606w')+ '/galaxy_'+str(index)+'_*')[0]
                im = np.zeros((128, 128, 1))
                im[:, :, 0] = fits.open(tmp_file)[0].data

                        
                ''' Resize the image to the wanted size'''
                im = _resize_image(im, p.img_len)
                
                ''' Load the wanted physical parameters of the galaxy '''
                if hasattr(p, 'attributes'):
                    attributes = {k: float(all_cat[k][index]) for k in p.attributes}

                else:
                    attributes=None
                
                ''' Add a flag corresponding to the field '''
                field_info = np.asarray(n_field)

                ''' Create the output to match T2T format '''
                serialized_output = {"image/encoded": [im.astype('float32').tostring()],
                "image/format": ["raw"],
                "psf/encoded": [psf.astype('float32').tostring()],
                "psf/format": ["raw"],
                "sigma_noise/encoded": [str(4.8e-3)],
                "sigma_noise/format": ["raw"],
                "field/encoded": [nb_field.astype('float32').tostring()],
                "field/format": ["raw"]}

                if attributes is not None:
                    for k in attributes:
                        serialized_output['attrs/'+k] = [attributes[k]]
#                     serialized_output['attrs/field'] = [attributes['field']]
                
                ''' Increment the number of galaxy created on the shard '''
                n_gal_creat += 1
                
                if n_gal_creat > p.example_per_shard:
                    break
                yield serialized_output
#             except Exception:
#                 print('No image corresponding to the Index')
#                 n+=1

  @property
  def is_generate_per_split(self):
    return False

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

        "sigma_noise/encoded": tf.FixedLenFeature((), tf.string),
        "sigma_noise/format": tf.FixedLenFeature((), tf.string),
        
        "field/encoded": tf.FixedLenFeature((), tf.string),
        "field/format": tf.FixedLenFeature((), tf.string)
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

        "sigma": tf.contrib.slim.tfexample_decoder.Image(
                image_key="sigma_noise/encoded",
                format_key="sigma_noise/format",
                channels=self.num_bands,
                shape=[self.num_bands],
                dtype=tf.float32),
        
        "field": tf.contrib.slim.tfexample_decoder.Image(
                image_key="field/encoded",
                format_key="field/format",
                shape=[1],
                dtype=tf.float32),
    }
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)
#         data_items_to_decoders['field'] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/field')

    return data_fields, data_items_to_decoders
#     END: Subclass interface

@registry.register_problem
class Img2imgCandelsAll(astroimage_utils.AstroImageProblem):
  """ Base class for image problems created from CANDELS GOODS (North and South) fields, in 7 bands.
  """

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
      .
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 0,
    }]

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.sigmas = [6.7e-3, 5.4e-3, 4.0e-3, 2.5e-3, 4.8e-3, 3.4e-3, 1.5e-3]
    p.filters = ['f105w', 'f125w', 'wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f814w', 'acs_f850l']
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
    """ 
    Generator yielding individual postage stamps.
    """
    
    p = self.get_hparams()
    
    '''Load the catalogue containing every fields and every filter'''
    all_cat = Table.read(os.path.join(tmp_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='egs   ')] = 'EGS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='UDS   ')] = 'UDS'
    
    ''' Load the psfs for each filter and resize'''
    cube_psf = np.zeros((167, 167, len(p.filters)))
    for i, filter in enumerate(p.filters):
        cube_psf[:, :, i] = fits.open(tmp_dir + '/psfs/psf_' + filter +'.fits')[0].data
                

    # Step 2: Extract postage stamps, resize them to requested size
    ''' Loop on the fields'''
    for n_field, field in enumerate(['GDS', 'GDN', 'EGS', 'UDS', 'COSMOS']):
        
        print(f"\n generating{field}, {n_field}\n")
        n_gal_creat = 0
        index = 0
        im = np.zeros((128, 128, len(p.filters)))
        sigmas = np.zeros(len(p.filters)) + 10
        im_psf = np.zeros((167, 167, len(p.filters)))
        ''' Create a subcat containing only the galaxies (in every filters) of the current field'''
        sub_cat = all_cat[np.where(all_cat["FIELD_1"]==field)[0]]
        
        if field in ['GDS', 'GDN']:
            filters_field = ['f105w', 'f125w', 'wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f850l']
            bands = [0, 1, 2, 3, 4, 5, 7]
        else :
            filters_field = ['f125w', 'wfc3_f160w', 'acs_f606w', 'acs_f814w']
            bands = [1, 2, 4, 6]

        sigmas[bands] = p.sigmas[bands]
        im_psf[bands] = cube_psf[bands]
        ''' Loop on all the galaxies of the field '''
        for gal in sub_cat['RB_ID']:
                if gal == index :     # To take care of the redudency inside the cat
                    continue
                index = gal

#             try:
                ''' Loop on the filters '''
                for n_filter, filter in enumerate(filters_field):
                    try :
                        ''' Open the image corresponding to the index of the current galaxy'''
                        tmp_file = glob.glob(os.path.join(tmp_dir, field, filter)+'/galaxy_'+str(index)+'_*')[0]
                        im[:, :, bands[n_filter]] = fits.open(tmp_file)[0].data
                    except Exception:
                        print('Galaxy not seen in every filter')
                        continue
                        
                ''' Resize the image to the wanted size'''
                im = _resize_image(im, p.img_len)
                print(self.num_bands)
                
                ''' Load the wanted physical parameters of the galaxy '''
                if hasattr(p, 'attributes'):
                    attributes = {k: float(all_cat[k][index]) for k in p.attributes}

                else:
                    attributes=None
                
#                 ''' Create the power spectrum '''
#                 ps = np.zeros((p.img_len, p.img_len//2+1, len(p.filters)))
#                 for i in range(len(p.filters)):
#                     ps[:, :, n_filter] = np.random.normal(0, p.sigmas[n_filter], (p.img_len, p.img_len // 2 + 1))
                
                ''' Add a flag corresponding to the field '''
                field_info = np.asarray(n_field)

                ''' Create the output to match T2T format '''
                serialized_output = {"image/encoded": [im.astype('float32').tostring()],
                "image/format": ["raw"],
                "psf/encoded": [im_psf.astype('float32').tostring()],
                "psf/format": ["raw"],
                "sigma_noise/encoded": [np.asarray(sigmas).astype('float32').tostring()],
                "sigma_noise/format": ["raw"],
                "field/encoded": [field_info.astype('float32').tostring()],
                "field/format": ["raw"]}
                
                if attributes is not None:
                    for k in attributes:
                        serialized_output['attrs/'+k] = [attributes[k]]
#                     serialized_output['attrs/field'] = [attributes['field']]
                
                ''' Increment the number of galaxy created on the shard '''
                n_gal_creat += 1
                
                if n_gal_creat > p.example_per_shard:
                    print('out')
                    break
                yield serialized_output
#             except Exception:
#                 print('No image corresponding to the Index')
#                 n+=1


@registry.register_problem
class Img2imgCandelsGoods(astroimage_utils.AstroImageProblem):
  """ Base class for image problems created from CANDELS GOODS (North and South) fields, in 7 bands.
  """

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
      .
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 0,
    }]

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.sigmas = [6.7e-3, 5.4e-3, 4.0e-3, 2.5e-3, 4.8e-3, 3.4e-3, 1.5e-3]
    p.filters = ['f105w', 'f125w', 'wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f850l']
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
    """ 
    Generator yielding individual postage stamps.
    """
    
    p = self.get_hparams()
    
    '''Load the catalogue containing every fields and every filter'''
    all_cat = Table.read(os.path.join(tmp_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    
    ''' Load the psfs for each filter and resize'''
    cube_psf = np.zeros((167, 167, len(p.filters)))
    for i, filter in enumerate(p.filters):
        cube_psf[:, :, i] = fits.open(tmp_dir + '/psfs/psf_' + filter +'.fits')[0].data
    
    scaling = p.pixel_scale/0.06
    im_psf = resize(cube_psf, (np.ceil(128/scaling)+1, np.ceil(128/scaling)+1,len(p.filters)))
    im_psf = _resize_image(cube_psf, p.img_len)
    
    im_tmp = np.zeros((128, 128, len(p.filters)))
    sigmas = np.asarray(p.sigmas)

    # Step 2: Extract postage stamps, resize them to requested size
    ''' Loop on the two fields'''
    for n_field, field in enumerate(['GDS']):#, 'GDN']):
        
        print(f"\n generating{field}, {n_field}\n")
        n_gal_creat = 0
        index = 0
        
        ''' Create a subcat containing only the galaxies (in every filters) of the current field'''
        sub_cat = all_cat[np.where(all_cat["FIELD_1"]==field)[0]]
        
        ''' Loop on all the galaxies of the field '''
        for gal in sub_cat['RB_ID']:
                if gal == index :     # To take care of the redudency inside the cat
                    continue
                index = gal

#             try:
                ''' Loop on the filters '''
                for n_filter, filter in enumerate(p.filters):
                    try :
                        ''' Open the image corresponding to the index of the current galaxy'''
                        tmp_file = glob.glob(os.path.join(tmp_dir, field, filter)+'/galaxy_'+str(index)+'_*')[0]
                        if np.max(fits.open(tmp_file)[0].data) == 0.:
                            sigmas[n_filter] = 10
                        im_tmp[:, :, n_filter] = fits.open(tmp_file)[0].data

                    except Exception:
                        print('Galaxy not seen in every filter')
                        continue
                        
                ''' Resize the image to the wanted size'''
                im = resize(im_tmp, (np.ceil(128/scaling)+1, np.ceil(128/scaling)+1, len(p.filters)))
                
                im = _resize_image(im, p.img_len)
                
                ''' Load the wanted physical parameters of the galaxy '''
                if hasattr(p, 'attributes'):
                    attributes = {k: float(all_cat[k][index]) for k in p.attributes}

                else:
                    attributes=None
                
#                 ''' Create the power spectrum '''
#                 ps = np.zeros((p.img_len, p.img_len//2+1, len(p.filters)))
#                 for i in range(len(p.filters)):
#                     ps[:, :, n_filter] = np.random.normal(0, p.sigmas[n_filter], (p.img_len, p.img_len // 2 + 1))
                
                ''' Add a flag corresponding to the field '''
                field_info = np.asarray(n_field)

                ''' Create the output to match T2T format '''
                serialized_output = {"image/encoded": [im.astype('float32').tostring()],
                "image/format": ["raw"],
                "psf/encoded": [im_psf.astype('float32').tostring()],
                "psf/format": ["raw"],
                "sigma_noise/encoded": [sigmas.astype('float32').tostring()],
                "sigma_noise/format": ["raw"],
                "field/encoded": [field_info.astype('float32').tostring()],
                "field/format": ["raw"]}
                
                if attributes is not None:
                    for k in attributes:
                        serialized_output['attrs/'+k] = [attributes[k]]
                
                ''' Increment the number of galaxy created on the shard '''
                n_gal_creat += 1
                
                if n_gal_creat > p.example_per_shard:
                    print('out')
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

        "sigma_noise/encoded": tf.FixedLenFeature((), tf.string),
        "sigma_noise/format": tf.FixedLenFeature((), tf.string),

        "field/encoded": tf.FixedLenFeature((), tf.string),
        "field/format": tf.FixedLenFeature((), tf.string),
    }

    # Adds additional attributes to be decoded as specified in the configuration
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_fields['attrs/'+k] = tf.FixedLenFeature([], tf.float32, -1)
    data_items_to_decoders = {
        "inputs": tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
#                 channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),

        "psf": tf.contrib.slim.tfexample_decoder.Image(
                image_key="psf/encoded",
                format_key="psf/format",
#                 channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),

        "sigma": tf.contrib.slim.tfexample_decoder.Image(
                image_key="sigma_noise/encoded",
                format_key="sigma_noise/format",
#                 channels=self.num_bands,
                shape=[self.num_bands],
                dtype=tf.float32),
        
        "field": tf.contrib.slim.tfexample_decoder.Image(
                image_key="field/encoded",
                format_key="field/format",
                shape=[1],
                dtype=tf.float32),
    }
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)
#         data_items_to_decoders['field'] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/field')

    return data_fields, data_items_to_decoders
#     END: Subclass interface
    
    
  @property
  def is_generate_per_split(self):
    return False
  
@registry.register_problem
class Attrs2imgCandelsOnefilter64Euclid(Img2imgCandelsOnefilter):
  """
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.1
    p.img_len = 64
    p.example_per_shard = 10
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag', 're', 'q']
    

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
    p.img_len = 64
    p.sigmas = [6.7e-3, 5.4e-3, 4.0e-3, 2.5e-3, 4.8e-3, 3.4e-3, 1.5e-3]
    p.example_per_shard = 5
    p.filters = ['f105w', 'f125w', 'wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f850l']

    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag', 're', 'q']
    
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
    p.img_len = 64
    p.sigmas = [6.7e-3, 5.4e-3, 4.0e-3, 2.5e-3, 4.8e-3, 3.4e-3, 1e-4, 1.5e-3]
    p.example_per_shard = 5
    p.filters = ['f105w', 'f125w', 'wfc3_f160w', 'acs_f435w', 'acs_f606w', 'acs_f775w', 'acs_f814w', 'acs_f850l']

    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag', 're', 'q']



@registry.register_problem
class Img2imgCandelsGoodsMultires(astroimage_utils.AstroImageProblem):
  """ Base class for image problems created from CANDELS GOODS (North and South) fields, in 7 bands.
  """

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
      .
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 0,
    }]

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.sigmas = {"high" : [1e-4], "low" : [6.7e-3, 5.4e-3, 4.0e-3]}
    p.filters = {"high" : ['acs_f814w'], "low" : ['f105w', 'f125w', 'wfc3_f160w']}
    p.resolutions = ["high","low"]
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf", None)

  @property
  def num_bands(self):
    """Number of bands."""
    p = self.get_hparams()
    return np.sum([len(p.filters[res]) for res in p.resolutions])

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """ 
    Generator yielding individual postage stamps.
    """
    
    p = self.get_hparams()
    band_num = np.sum([len(p.filters[res]) for res in p.resolutions])
    scalings = {}
    for res in p.resolutions:
        scalings[res] = p.pixel_scale[res]/p.base_pixel_scale[res]
    target_pixel_scale = p.pixel_scale[p.resolutions[0]]
    target_scaling = target_pixel_scale/p.base_pixel_scale[p.resolutions[0]]
    print("scalings and all ",scalings,target_pixel_scale,target_scaling)
    
    '''Load the catalogue containing every fields and every filter'''
    all_cat = Table.read(os.path.join(data_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    
    ''' Load the psfs for each filter and resize'''
    cube_psf = np.zeros((167, 167, band_num))
    k = 0
    for res in p.resolutions:
        cube_psf_tmp = np.zeros((167, 167, len(p.filters[res])))
        for i, filt in enumerate(p.filters[res]):
            cube_psf_tmp[:, :, i] = fits.open(data_dir + '/psfs/psf_' + filt +'.fits')[0].data
        cube_psf_tmp = resize(cube_psf_tmp, (167,167,len(p.filters[res])))
        cube_psf[:,:,k:k+len(p.filters[res])] = cube_psf_tmp

        k += len(p.filters[res])
    
    im_psf = _resize_image(cube_psf, p.img_len)
    
    sigmas = p.sigmas

    # Step 2: Extract postage stamps, resize them to requested size
    ''' Loop on the two fields'''
    for n_field, field in enumerate(['GDS']):
        
        print(f"\n generating{field}, {n_field}\n")
        n_gal_creat = 0
        index = 0
        
        ''' Create a subcat containing only the galaxies (in every filters) of the current field'''
        sub_cat = all_cat[np.where(all_cat["FIELD_1"]==field)[0]]

        ''' Loop on all the galaxies of the field '''
        for gal in sub_cat['RB_ID']:
            if gal == index or gal == 15431 :     # To take care of the redudency inside the cat
                continue
            index = gal
            print(index)

            try:
                ''' Loop on the filters '''
                target_size = int(np.ceil(128/target_scaling)+1)
                print("target ",target_size)
                im = np.zeros((target_size, target_size, band_num))

                k = 0
                for res in p.resolutions:
                    im_tmp = np.zeros((128, 128, len(p.filters[res])))
                    for n_filter, filt in enumerate(p.filters[res]):
                        print(filt)
                        # try :
                        ''' Open the image corresponding to the index of the current galaxy'''

                        tmp_file = glob.glob(os.path.join(data_dir, field, filt)+'/galaxy_'+str(index)+'_*')[0]
                        if np.max(fits.open(tmp_file)[0].data) == 0.:
                            sigmas[res][n_filter] = 10
                        im_import = fits.open(tmp_file)[0].data
                        im_tmp[:, :, n_filter] = clean_rotate_stamp(im_import,sigma_sex=1.5)

                        # except Exception:
                        #     print('Galaxy not seen in every filter')
                        #     continue
                            
                    ''' Resize the image to the low resolution'''
                    new_size = np.ceil(128/scalings[res])+1
                    im_tmp = resize(im_tmp, (new_size, new_size, len(p.filters[res])))

                    ''' Resize the image to the highest resolution to get consistent array sizes'''
                    im_tmp = resize(im_tmp, (target_size, target_size, len(p.filters[res])))

                    im[:,:,k:k+len(p.filters[res])] = im_tmp
                    k += len(p.filters[res])
                
                im = _resize_image(im, p.img_len)
                

                ''' Load the wanted physical parameters of the galaxy '''
                if hasattr(p, 'attributes'):
                    attributes = {k: float(all_cat[k][index]) for k in p.attributes}

                else:
                    attributes=None
                
#                 ''' Create the power spectrum '''
#                 ps = np.zeros((p.img_len, p.img_len//2+1, len(p.filters)))
#                 for i in range(len(p.filters)):
#                     ps[:, :, n_filter] = np.random.normal(0, p.sigmas[n_filter], (p.img_len, p.img_len // 2 + 1))
                
                ''' Add a flag corresponding to the field '''
                field_info = np.asarray(n_field)

                sigmas_array = []
                for res in p.resolutions:
                    sigmas_array += sigmas[res]
                sigmas_array = np.array(sigmas_array)

                ''' Create the output to match T2T format '''
                serialized_output = {"image/encoded": [im.astype('float32').tostring()],
                "image/format": ["raw"],
                "psf/encoded": [im_psf.astype('float32').tostring()],
                "psf/format": ["raw"],
                "sigma_noise/encoded": [sigmas_array.astype('float32').tostring()],
                "sigma_noise/format": ["raw"],
                "field/encoded": [field_info.astype('float32').tostring()],
                "field/format": ["raw"]}
                
                if attributes is not None:
                    for k in attributes:
                        serialized_output['attrs/'+k] = [attributes[k]]
                
                ''' Increment the number of galaxy created on the shard '''
                n_gal_creat += 1
                
                if n_gal_creat > p.example_per_shard:
                    print('out')
                    break
                yield serialized_output
            except ValueError:
                print(sys.exc_info()[0], sys.exc_info()[1])
                continue

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

        "sigma_noise/encoded": tf.FixedLenFeature((), tf.string),
        "sigma_noise/format": tf.FixedLenFeature((), tf.string),

        "field/encoded": tf.FixedLenFeature((), tf.string),
        "field/format": tf.FixedLenFeature((), tf.string),
    }

    # Adds additional attributes to be decoded as specified in the configuration
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_fields['attrs/'+k] = tf.FixedLenFeature([], tf.float32, -1)
    data_items_to_decoders = {
        "inputs": tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
#                 channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),

        "psf": tf.contrib.slim.tfexample_decoder.Image(
                image_key="psf/encoded",
                format_key="psf/format",
#                 channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),

        "sigma": tf.contrib.slim.tfexample_decoder.Image(
                image_key="sigma_noise/encoded",
                format_key="sigma_noise/format",
#                 channels=self.num_bands,
                shape=[self.num_bands],
                dtype=tf.float32),
        
        "field": tf.contrib.slim.tfexample_decoder.Image(
                image_key="field/encoded",
                format_key="field/format",
                shape=[1],
                dtype=tf.float32),
    }
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)
#         data_items_to_decoders['field'] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/field')

    return data_fields, data_items_to_decoders
#     END: Subclass interface
    
    
  @property
  def is_generate_per_split(self):
    return False



@registry.register_problem
class Attrs2imgCandelsGoodsEuclid64(Img2imgCandelsGoodsMultires):
  """
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = {'high' : 0.1, 'low' : 0.3}
    p.base_pixel_scale = {'high' : 0.03,'low' : 0.13}
    p.img_len = 64
    p.sigmas = {"high" : [1e-4], "low" : [6.7e-3, 5.4e-3, 4.0e-3]}
    p.filters = {"high" : ['acs_f814w'], "low" : ['f105w', 'f125w', 'wfc3_f160w']}
    p.resolutions = ["high","low"]
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag', 're', 'q']



@registry.register_problem
class Attrs2imgCandelsGoodsEuclid64Test(Img2imgCandelsGoodsMultires):
  """
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = {'high' : 0.1, 'low' : 0.3}
    p.base_pixel_scale = {'high' : 0.05,'low' : 0.13}
    p.img_len = 64
    p.sigmas = {"high" : [3.4e-4], "low" : [6.7e-3, 5.4e-3, 4.0e-3]}
    p.filters = {"high" : ['acs_f775w'], "low" : ['f105w', 'f125w', 'wfc3_f160w']}
    p.resolutions = ["high","low"]
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag', 're', 'q']






def find_central(sex_cat):

    n_detect = len(sex_cat)
    
    ''' Match the pred and true cat'''
    pred_pos = np.zeros((n_detect, 2))
    pred_pos[:, 0] = sex_cat['x']
    pred_pos[:, 1] = sex_cat['y']

    true_pos = np.zeros((1, 2))
    true_pos[:, 0] = 64
    true_pos[:, 1] = 64

    _, match_index = KDTree(pred_pos).query(true_pos)
    
    return match_index


import re 

def sort_nicely( l ): 
    """ Sort the given list in the way that humans expect. 
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    l.sort( key=alphanum_key ) 
    return l


def mask_out_pixels(img, segmap, segval,
                    n_iter: int = 5, shuffle: bool = False,
                    noise_factor: int = 1):
    """
    Replace central galaxy neighbours with background noise

    Basic recipe to replace the detected sources around the central galaxy
    with either randomly selected pixels from the background, or a random
    realisation of the background noise.

    """
    masked_img = img.copy()
    # Create binary masks of all segmented sources
    sources = binary_dilation(segmap, iterations=n_iter)

    background_mask = np.logical_not(sources)
    # Create binary mask of the central galaxy
    central_source = binary_dilation(np.where(segmap == segval, 1, 0),
                                     iterations=n_iter)
    # Compute the binary mask of all sources BUT the central galaxy
    sources_except_central = np.logical_xor(sources, central_source)

    if shuffle:
        # Select random pixels from the noise in the image
        n_pixels_to_fill_in = sources_except_central.sum()
        random_background_pixels = np.random.choice(
            img[background_mask],
            size=n_pixels_to_fill_in
        )
        # Fill in the voids with these pixels
        masked_img[sources_except_central] = random_background_pixels
    else:
        # Create a realisation of the background for the std value
        background_std = np.std(img[background_mask])
        # background_std = 0.007220430274502116
        random_background = np.random.normal(scale=background_std, size=img.shape)
        masked_img[sources_except_central] = random_background[sources_except_central]
        masked_img[np.where(masked_img==0.0)] = random_background[np.where(masked_img==0.0)]
        
    return masked_img.astype(img.dtype), sources, background_mask, central_source, sources_except_central

def clean_rotate_stamp(img, eps=5, sigma_sex=2):

    ''' Sex for clean'''
    img = img.byteswap().newbyteorder()
    bkg = sep.Background(img)
   
    cat,sex_seg = sep.extract(img-bkg,sigma_sex,err=bkg.globalrms,segmentation_map=True)
    
    if len(cat) == 0:
        raise ValueError('No galaxy detected in the field')
    
    middle_pos = [cat[find_central(cat)[0]]['x'],cat[find_central(cat)[0]]['y']]
    
    distance = np.sqrt((middle_pos[0]-64)**2 + (middle_pos[1]-64)**2)
    if distance > 10 :
        raise ValueError('No galaxy detected in the center')

    middle = np.max(sex_seg[int(round(middle_pos[0]))-eps:int(round(middle_pos[0]))+eps, int(round(middle_pos[1]))-eps:int(round(middle_pos[1]))+eps])
    if middle == 0:
        raise ValueError('No galaxy detected in the center')

    cleaned, _, _, central, _ = mask_out_pixels(img, sex_seg, middle,n_iter=5)
    
    if np.any(np.logical_and(np.not_equal(sex_seg[central],0),np.not_equal(sex_seg[central],middle))):
       raise ValueError('Blending suspected')


    '''Rotate'''
    PA = cat[find_central(cat)[0]][4]
    img_rotate = rotate(cleaned, PA, reshape=False)
    
    '''Add noise'''
    background_mask = np.where(sex_seg == 0, 1, 0)
    background_std = np.std(img * background_mask)
    random_background = np.random.normal(scale=background_std, size=img_rotate.shape)
    rotated = np.where(img_rotate == 0, random_background, img_rotate)

    return rotated