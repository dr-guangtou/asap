#!/usr/bin/env python3
"""Script that make a galaxy catalog for ASAP model based on the UM output.
"""

import os
import argparse

import numpy as np

from astropy.table import Table, vstack

from asap.vagc import prep_um_catalog, value_added_mock


def main(um_file, box_size=400., mhalo_lim=11.5, mhalo_type='peak',
         save_fits=True, verbose=True):
    """Main function."""
    if not os.path.isfile(um_file):
        raise IOError("# Can not find the UM catalog :" % um_file)

    # Separate the prefix and file extention
    um_pre, um_ext = os.path.splitext(um_file)
    um_pre_out = um_pre + '_vagc_m%s_%4.1f' % (mhalo_type.strip(), mhalo_lim)

    if um_ext.strip() is not 'npz':
        raise IOError("# We need the .npz format UM catalog")

    # Read in the catalog and separate the centrals and satellites
    um_final = np.load(um_file)
    um_cen, um_sat = Table(um_final['centrals']), Table(um_final['satellites'])
    if verbose:
        print("# There are %d centrals and %d satellites" % (len(um_cen), len(um_sat)))

    # Combine the central and satellites
    um_all = vstack([um_cen, um_sat])

    # Value added the catalogs
    um_vagc = value_added_smdpl_mock(um_all, box_size=box_size)

    # Organize the catalog, add additional information, and make Mpeak cut
    um_use = prep_um_catalog(um_vagc, um_min_mvir=mhalo_lim,
                             mhalo_col='logmh_%s' % mhalo_type.strip())

    # Save the results
    if save_fits:
        if verbose:
            print("# Will save fits catalog : %s" % (um_pre_out + '.fits'))
        um_use.write(um_pre_out + '.fits', overwrite=True, format='fits')

    # Save the numpy file
    np.save(um_pre_out + '.npy', np.asarray(um_use))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'um_file', type=str,
        help=('UniverseMachine snapshot file in .txt format'))

    parser.add_argument(
        '-s', '--boxsize', dest='box_size',
        help='Box size of the DM simulation. Default: 400 Mpc/h (SMDPL)',
        type=float, default=400.)

    parser.add_argument(
        '-t', '--type', dest='mhalo_type',
        help='Which halo mass to use in the selection',
        type=str, default='peak')

    parser.add_argument(
        '-m', '--mhalo', dest='mhalo_lim',
        help='Lower limit of Mpeak',
        type=float, default=11.0)

    parser.add_argument(
        '-v', '--verbose', dest='verbose',
        action="store_true", default=False)

    parser.add_argument(
        '-f', '--fits', dest='save_fits',
        action="store_true", default=False)

    args = parser.parse_args()

    main(args.um_file, mhalo_lim=args.mhalo_lim, mhalo_type=args.mhalo_type,
         verbose=args.verbose, save_fits=args.save_fits)
