import galsim
import numpy as np
import os
import sys
import tensorflow as tf
from collections import namedtuple

__all__ =['draw_and_encode_stamp']


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def draw_and_encode_stamp(gal, psf, stamp_size, pixel_scale, parameters=None):
    """
    Draws the galaxy, psf and noise power spectrum on a postage stamp and
    encodes it to be exported in a TFRecord.

    Parameters
    ----------
    parameters: `dict`
        Additional scalar parameters to save for that galaxy
    """
    # Apply the PSF
    gal = galsim.Convolve(gal, psf)

    # Draw the Fourier domain image of the galaxy
    imC = galsim.ImageCF(stamp_size, stamp_size, scale=2. *
                         np.pi / (pixel_scale * stamp_size))

    imCp = galsim.ImageCF(stamp_size, stamp_size, scale=2. *
                          np.pi / (pixel_scale * stamp_size))

    gal.drawKImage(image=imC)
    psf.drawKImage(image=imCp)

    # Keep track of the pixels with 0 value
    mask = ~(np.fft.fftshift(imC.array)[:, :(stamp_size) // 2 + 1] == 0)

    # Inverse Fourier transform of the image
    # TODO: figure out why we need 2 fftshifts....
    im = np.fft.fftshift(np.fft.ifft2(
        np.fft.fftshift(imC.array))).real.astype('float32')

    # Transform the psf array into proper format for Theano
    im_psf = np.fft.fftshift(np.fft.ifft2(
            np.fft.fftshift(imCp.array))).real.astype('float32')

    # Compute noise power spectrum
    try:
        ps = gal.noise._get_update_rootps((stamp_size, stamp_size),
                                      wcs=galsim.PixelScale(pixel_scale))

        # The following comes from correlatednoise.py
        rt2 = np.sqrt(2.)
        shape = (stamp_size, stamp_size)
        ps[0, 0] = rt2 * ps[0, 0]
        # Then make the changes necessary for even sized arrays
        if shape[1] % 2 == 0:  # x dimension even
            ps[0, shape[1] // 2] = rt2 * ps[0, shape[1] // 2]
        if shape[0] % 2 == 0:  # y dimension even
            ps[shape[0] // 2, 0] = rt2 * ps[shape[0] // 2, 0]
            # Both dimensions even
            if shape[1] % 2 == 0:
                ps[shape[0] // 2, shape[1] // 2] = rt2 * \
                    ps[shape[0] // 2, shape[1] // 2]

    except:
        # In case there is no noise in the images, flat PS
        ps = np.ones_like(mask)

    # Apply mask to power spectrum so that it is very large outside maxk
    ps = np.where(mask, np.log(ps**2), 10).astype('float32')

    serialized_output = {"image/encoded": [im.tostring()],
            "image/format": ["raw"],
            "psf/encoded": [im_psf.tostring()],
            "psf/format": ["raw"],
            "ps/encoded": [ps.tostring()],
            "ps/format": ["raw"]}

    # Adding the parameters provided
    if parameters is not None:
        for k in parameters:
            serialized_output['params/'+k] = [parameters[k]]

    return serialized_output
