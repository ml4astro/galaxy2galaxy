"""Scripts to generate HSC samples from the HSC archive."""
import argparse
from unagi import hsc
from unagi import task

import os
import shutil
import astropy.units as u
from astropy.table import Table

def build_hsc_sample(sql_file,
                     out_dir,
                     tmp_dir,
                     filters='i',
                     cutout_size=10.0, # in arcsec
                     data_release='pdr2',
                     rerun='pdr2_wide',
                     nproc=2):
    """
    This function runs an sql query to extract a catalog, then
    proceeds to download cutouts in requested bands for all
    catalog entries.
    """
    # Creates output directory if doesn't already exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Create archive instance and login
    archive = hsc.Hsc(dr=data_release, rerun=rerun)

    catalog_filename = os.path.join(out_dir, 'catalog.fits')
    if os.path.isfile(catalog_filename):
        print("Readding hsc catalog from %s"%catalog_filename)
        catalog = Table.read(catalog_filename)
    else:
        # Query the database
        catalog = archive.sql_query(sql_file,
                                from_file=True,
                                verbose=True)

        print("Saving hsc catalog to %s"%catalog_filename)
        catalog.write(catalog_filename)

    # Query corresponding postage stamps
    cutouts_filename = task.hsc_bulk_cutout(catalog,
                                            cutout_size=cutout_size* u.Unit('arcsec'),
                                            filters=filters,
                                            archive=archive,
                                            nproc=nproc,
                                            tmp_dir=tmp_dir,
                                            output_dir=out_dir)
    # Rename the cutout file
    shutil.move(cutouts_filename, os.path.join(out_dir, 'cutouts.hdf'))
