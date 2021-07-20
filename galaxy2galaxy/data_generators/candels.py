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
import galsim
from skimage.transform import resize,rescale
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
class Img2imgCandelsMultires(astroimage_utils.AstroImageProblem):
  """ Base class for image problems with the CANDELS catalog, with multiresolution images.
  """

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
      .
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 20,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 2,
    }]

  @property
  def multiprocess_generate(self):
    """Whether to generate the data in multiple parallel processes."""
    return True

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.img_len = 128
    p.sigmas = {"high" : [1e-4], "low" : [4.0e-3]}
    p.filters = {"high" : ['acs_f814w'], "low" : ['wfc3_f160w']}
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
    print(task_id)
    
    p = self.get_hparams()
    band_num = np.sum([len(p.filters[res]) for res in p.resolutions])
    scalings = {}
    for res in p.resolutions:
        scalings[res] = p.pixel_scale[res]/p.base_pixel_scale[res]
    target_pixel_scale = p.pixel_scale[p.resolutions[0]]
    target_scaling = target_pixel_scale/p.base_pixel_scale[p.resolutions[0]]
    target_size = p.img_len
    
    '''Load the catalogue containing every fields and every filter'''
    all_cat = Table.read(os.path.join(data_dir, 'CANDELS_morphology_v8_3dhst_galfit_ALLFIELDS.fit'))
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='gdn   ')] = 'GDN'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='egs   ')] = 'EGS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='GDS   ')] = 'GDS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='UDS   ')] = 'UDS'
    all_cat['FIELD_1'][np.where(all_cat['FIELD_1']=='COSMOS   ')] = 'COSMOS'
    
    ''' Load the psfs for each filter and resize'''
    cube_psf = np.zeros((2*p.img_len, 2*p.img_len // 2 + 1, band_num))
    interp_factor=2
    padding_factor=1
    Nk = p.img_len*interp_factor*padding_factor
    bounds = galsim.bounds._BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
    k = 0
    for res in p.resolutions:
        cube_psf_tmp = np.zeros((2*p.img_len, 2*p.img_len // 2 + 1, len(p.filters[res])))
        for i, filt in enumerate(p.filters[res]):
            psf = galsim.InterpolatedImage(data_dir + '/psfs/psf_' + filt +'.fits',scale=0.06)
            
            imCp = psf.drawKImage(bounds=bounds,
                                 scale=2.*np.pi/(Nk * p.pixel_scale[res] / interp_factor),
                                 recenter=False)

            # Transform the psf array into proper format, remove the phase
            im_psf = np.abs(np.fft.fftshift(imCp.array, axes=0)).astype('float32')
            cube_psf_tmp[:, :, i] = im_psf
        cube_psf_tmp = resize(cube_psf_tmp, (2*p.img_len, 2*p.img_len // 2 + 1,len(p.filters[res])))
        cube_psf[:,:,k:k+len(p.filters[res])] = cube_psf_tmp
        k += len(p.filters[res])

    psf = cube_psf
    
    sigmas = p.sigmas

    # Step 2: Extract postage stamps, resize them to requested size
    n_gal_creat = 0
    index = 0
   
    ''' Create a subcat containing only the galaxies (in every filters) of the current field'''
    sub_cat = all_cat[np.where(np.isin(list(all_cat["FIELD_1"]),["GDS","GDN","EGS","COSMOS","UDS"]))]
    sub_cat = sub_cat[np.where(sub_cat['mag'] <= 25.3)]
    assert(task_id > -1)
    n_shards = self.dataset_splits[0]["shards"] + self.dataset_splits[1]["shards"]
    indexes = list(range(task_id*len(sub_cat)//n_shards,
                  min((task_id+1)*len(sub_cat)//n_shards, len(sub_cat))))
    sub_cat = sub_cat[indexes]

    ''' Loop on all the galaxies of the field '''
    for m,gal in enumerate(sub_cat['RB_ID']):
        if gal == index or gal == 15431 or sub_cat["mag"][m] < 0:     # To take care of the redudency inside the cat
            continue
        index = gal
        print(index)
        target_flux_main_band = 10**(-0.4*(sub_cat['mag'][m]-p.zeropoint))

        try:
            ''' Loop on the filters '''
            im = np.zeros((target_size, target_size, band_num))

            k = 0  
            for res in p.resolutions:
                im_tmp = np.zeros((128, 128, len(p.filters[res])))
                for n_filter, filt in enumerate(p.filters[res]):
                    print(filt)
                    ''' Open the image corresponding to the index of the current galaxy'''

                    tmp_file = glob.glob(os.path.join(data_dir, sub_cat["FIELD_1"][m], filt)+'/galaxy_'+str(index)+'_*')[0]
                    im_import = fits.open(tmp_file)[0].data
                    cleaned_image = clean_rotate_stamp(im_import,sigma_sex=1.5)#,noise_level=p.sigmas[res][n_filter])

                    if res == p.resolutions[0] and n_filter == 0:
                        flux_ratio = 1/np.max(cleaned_image) if np.max(cleaned_image) != 0 else 1

                    im_tmp[:, :, n_filter] = cleaned_image * flux_ratio
                    if np.max(cleaned_image) <= 5*10**(-3):
                        raise ValueError("Very weak image")
                        
                ''' Resize the image to the low resolution'''
                new_size = np.ceil(128/scalings[res])+1
                im_tmp = resize(im_tmp, (new_size, new_size, len(p.filters[res])))
                ''' Resize the image to the highest resolution to get consistent array sizes'''
                im_tmp = rescale(im_tmp,p.pixel_scale[res]/target_pixel_scale,multichannel=True,preserve_range=True)
                im_tmp = _resize_image(im_tmp,target_size)

                im[:,:,k:k+len(p.filters[res])] = im_tmp
                k += len(p.filters[res])
            
            im = _resize_image(im, p.img_len)
            
            # Check that there is still a galaxy
            img_s = im[:,:,0]
            img_s = img_s = img_s.copy(order='C')
            bkg = sep.Background(img_s)
            cat_s = sep.extract(img_s-bkg,2,err=bkg.globalrms)  
            if len(cat_s) == 0:
                raise ValueError('No galaxy detected in the field')

            ''' Load the wanted physical parameters of the galaxy '''
            if hasattr(p, 'attributes'):
                attributes = {k: float(sub_cat[k][m]) for k in p.attributes}

            else:
                attributes=None
            
            ''' Create the power spectrum '''
            k = 0
            noise_im = np.zeros((p.img_len, p.img_len, band_num))
            for res in p.resolutions:
                for n_filter in range(len(p.filters[res])):
                    if False:
                        noise_im[:, :, n_filter+k] = np.random.normal(0, bkg.globalrms, (p.img_len, p.img_len))
                    else:
                        noise_im[:, :, n_filter+k] = np.random.normal(0, p.sigmas[res][n_filter], (p.img_len, p.img_len))
                k+=1
            noise_im = np.transpose(noise_im,[2,0,1])
            ps = np.abs(np.fft.rfft2(noise_im))
            ps = np.transpose(ps,[1,2,0])

            ''' Add a flag corresponding to the field '''
            field_info = np.asarray(1 if sub_cat["FIELD_1"][m] == "GDS" else 0)

            sigmas_array = []
            for res in p.resolutions:
                sigmas_array += sigmas[res]
            sigmas_array = np.array(sigmas_array)

            ''' Create the output to match T2T format '''
            serialized_output = {"image/encoded": [im.astype('float32').tostring()],
            "image/format": ["raw"],
            "psf/encoded": [psf.astype('float32').tostring()],
            "psf/format": ["raw"],
            "ps/encoded": [ps.astype('float32').tostring()],
            "ps/format": ["raw"],
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
                print('out ',n_gal_creat)
                break
            yield serialized_output
        except Exception:
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
    """
    Define how data is serialized to file and read back.

    Returns:
      data_fields: A dictionary mapping data names to its feature type.
      data_items_to_decoders: A dictionary mapping data names to TF Example
                              decoders, to be used when reading back TF examples 
                              from disk.
    """
    p = self.get_hparams()

    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),

        "psf/encoded": tf.FixedLenFeature((), tf.string),
        "psf/format": tf.FixedLenFeature((), tf.string),

        "ps/encoded": tf.FixedLenFeature((), tf.string),
        "ps/format": tf.FixedLenFeature((), tf.string),

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
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),

        "psf": tf.contrib.slim.tfexample_decoder.Image(
                image_key="psf/encoded",
                format_key="psf/format",
                shape=[2*p.img_len, 2*p.img_len // 2 + 1, self.num_bands],
                dtype=tf.float32),

        "ps": tf.contrib.slim.tfexample_decoder.Image(
                image_key="ps/encoded",
                format_key="ps/format",
                shape=[p.img_len, p.img_len//2+1, self.num_bands],
                dtype=tf.float32),

        "sigma_noise": tf.contrib.slim.tfexample_decoder.Image(
                image_key="sigma_noise/encoded",
                format_key="sigma_noise/format",
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

    return data_fields, data_items_to_decoders
#     END: Subclass interface
    
    
  @property
  def is_generate_per_split(self):
    return False



@registry.register_problem
class Attrs2imgCandelsEuclid64(Img2imgCandelsMultires):
  """For generating images with the Euclid bands
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = {'high' : 0.1, 'low' : 0.3}
    p.base_pixel_scale = {'high' : 0.06,'low' : 0.06}
    p.img_len = 64
    p.sigmas = {"high" : [1e-4], "low" : [0.003954237367399534, 0.003849901319445, 0.004017507500562]}
    p.filters = {"high" : ['acs_f814w'], "low" : ['f105w', 'f125w', 'wfc3_f160w']}
    p.resolutions = ["high","low"]
    p.example_per_shard = 2000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag', 're', 'q']



@registry.register_problem
class Attrs2imgCandelsEuclid64TwoBands(Img2imgCandelsMultires):
  """ For generating two-band images (visible and infrared)
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = {'high' : 0.1, 'low' : 0.1}
    p.base_pixel_scale = {'high' : 0.06,'low' : 0.06}
    p.img_len = 64
    p.sigmas = {"high" : [0.004094741966557142], "low" : [0.004017507500562]}
    p.filters = {"high" : ['acs_f606w'], "low" : ['wfc3_f160w']}
    p.zeropoint = 26.49
    p.resolutions = ["high","low"]
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag','re', 'q','ZPHOT','F_IRR','F_SPHEROID','F_DISK']


def find_central(sex_cat,center_coords=64):
    """Find the central galaxy in a catalog provided by SExtractor
    """
    n_detect = len(sex_cat)
    
    ''' Match the pred and true cat'''
    pred_pos = np.zeros((n_detect, 2))
    pred_pos[:, 0] = sex_cat['x']
    pred_pos[:, 1] = sex_cat['y']

    true_pos = np.zeros((1, 2))
    true_pos[:, 0] = center_coords
    true_pos[:, 1] = center_coords

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
                    noise_factor: int = 1, noise_level=None):
    """
    Replace central galaxy neighbours with background noise

    Basic recipe to replace the detected sources around the central galaxy
    with either randomly selected pixels from the background, or a random
    realisation of the background noise.

    """
    masked_img = img.copy()
    # Create binary masks of all segmented sources
    sources = binary_dilation(segmap, iterations=n_iter)

    background_mask = np.logical_and(np.logical_not(sources),np.array(img,dtype=bool))
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
        if noise_level == None:
            background_std = np.std(img[background_mask])
        else:
            background_std = noise_level
        random_background = np.random.normal(scale=background_std, size=img.shape)
        masked_img[sources_except_central] = random_background[sources_except_central]
        masked_img[np.where(masked_img==0.0)] = random_background[np.where(masked_img==0.0)]
        
    return masked_img.astype(img.dtype), sources, background_mask, central_source, sources_except_central

def clean_rotate_stamp(img, eps=5, sigma_sex=2, noise_level=None, rotate_b=False, blend_threshold=0.1):
    """Clean images by removing galaxies other than the central one.
    """

    # Detect galaxies with SExtractor
    img = img.byteswap().newbyteorder()
    im_size = img.shape[0]
    bkg = sep.Background(img)
   
    cat,sex_seg = sep.extract(img-bkg,sigma_sex,err=bkg.globalrms,segmentation_map=True)
    
    if len(cat) == 0:
        raise ValueError('No galaxy detected in the field')
    
    middle_pos = [cat[find_central(cat,im_size[0]//2)[0]]['x'],cat[find_central(cat,im_size[0]//2)[0]]['y']]
    
    distance = np.sqrt((middle_pos[0]-im_size[0]//2)**2 + (middle_pos[1]-im_size[0]//2)**2)
    if distance > 10 :
        raise ValueError('No galaxy detected in the center')

    middle = np.max(sex_seg[int(round(middle_pos[0]))-eps:int(round(middle_pos[0]))+eps, int(round(middle_pos[1]))-eps:int(round(middle_pos[1]))+eps])
    if middle == 0:
        raise ValueError('No galaxy detected in the center')

    cleaned, _, _, central, _ = mask_out_pixels(img, sex_seg, middle,n_iter=5,noise_level=noise_level)
    
    blended_pixels = np.logical_and(np.not_equal(sex_seg,0),np.not_equal(sex_seg,middle))*central
    blend_flux = np.sum(img[np.nonzero(blended_pixels)])
    if np.any(blended_pixels):
        loc = np.argwhere(blended_pixels==True)
        blended_galaxies = np.unique(sex_seg[loc])
        for blended_galaxy in blended_galaxies:
            blended_galaxy_flux = np.sum(img[np.where(sex_seg==blended_galaxy)])
            if blend_flux/blended_galaxy_flux > blend_threshold:
              raise ValueError('Blending suspected')

    # Rotate
    if rotate_b:
        PA = cat[find_central(cat)[0]][4]
        img_rotate = rotate(cleaned, PA, reshape=False)
    else:
        img_rotate = cleaned

    # Add noise
    background_mask = np.logical_and(np.logical_not(sex_seg==0),np.array(img,dtype=bool))
    if noise_level == None:
        background_std = np.std(img * background_mask)
    else:
        background_std = noise_level
    random_background = np.random.normal(scale=background_std, size=img_rotate.shape)
    rotated = np.where(img_rotate == 0, random_background, img_rotate)

    return rotated