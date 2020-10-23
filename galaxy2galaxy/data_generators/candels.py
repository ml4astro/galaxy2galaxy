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
from skimage.transform import resize



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