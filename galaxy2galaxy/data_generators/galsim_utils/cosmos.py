import galsim
import numpy as np
import os
import sys
import tensorflow as tf
from collections import namedtuple

__all__ = ['maybe_download_cosmos']


def maybe_download_cosmos(target_dir, sample="25.2"):
    """
    Checks for already accessible cosmos data, downloads it somewhere otherwise
    """
    import logging
    logging_level = logging.INFO

    # Setup logging to go to sys.stdout or (if requested) to an output file
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    logger = logging.getLogger('galaxy2galaxy:galsim')

    url = "http://great3.jb.man.ac.uk/leaderboard/data/public/COSMOS_%s_training_sample.tar.gz"%(sample)
    file_name = os.path.basename(url)
    target = os.path.join(target_dir, file_name)
    unpack_dir = target[:-len('.tar.gz')]
    args = {'quiet': True, 'force': False, 'verbosity': 2}
    args = namedtuple('Args', args.keys())(*args.values())

    # Download the tarball
    new_download, target, meta = galsim.download_cosmos.download(url, target,
                                                                 unpack_dir,
                                                                 args, logger)
    # Usually we unpack if we downloaded the tarball
    do_unpack = new_download

    # If the unpack dir is missing, then need to unpack
    if not os.path.exists(unpack_dir):
        do_unpack = True

    # But of course if there is no tarball, we can't unpack it
    if not os.path.isfile(target):
        do_unpack = False

    # Unpack the tarball
    if do_unpack:
        galsim.download_cosmos.unpack(target, target_dir, unpack_dir, meta,
                                      args, logger)

    # Usually, we remove the tarball if we unpacked it and command line doesn't specify to save it.
    do_remove = do_unpack

    # Remove the tarball
    if do_remove:
        logger.info("Removing the tarball to save space")
        os.remove(target)
