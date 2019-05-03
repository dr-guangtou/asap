#!/usr/bin/env python
"""This script will read the dark matter particle table for the SMDPL
simulation, and downsample it for our model.
"""

import os
import argparse

import numpy as np
import pandas as pd


S18A_RAND_COLS = [
    'object_id', 'ra', 'dec', 'coord', 'skymap_id', 'tract', 'patch', 'patch_s', 'parent_id',
    'nchild', 'isprimary', 'adjust_density', 'detect_ispatchinner', 'detect_istractinner',
    'g_pix_variance', 'g_sky_mean', 'g_sky_std', 'g_inputcount_value', 'g_inputcount_flag',
    'g_inputcount_flag_noinputs', 'g_inputcount_flag_badcentroid', 'g_pixelflags',
    'g_pixelflags_offimage', 'g_pixelflags_edge', 'g_pixelflags_bad',
    'g_pixelflags_interpolatedcenter', 'g_pixelflags_saturatedcenter', 'g_pixelflags_crcenter',
    'g_pixelflags_suspectcenter', 'g_pixelflags_bright_objectcenter', 'g_sdssshape_psf_shape11',
    'g_sdssshape_psf_shape22', 'g_sdssshape_psf_shape12',
    'r_pix_variance', 'r_sky_mean', 'r_sky_std', 'r_inputcount_value', 'r_inputcount_flag',
    'r_inputcount_flag_noinputs', 'r_inputcount_flag_badcentroid', 'r_pixelflags',
    'r_pixelflags_offimage', 'r_pixelflags_edge', 'r_pixelflags_bad',
    'r_pixelflags_interpolatedcenter', 'r_pixelflags_saturatedcenter', 'r_pixelflags_crcenter',
    'r_pixelflags_suspectcenter', 'r_pixelflags_bright_objectcenter', 'r_sdssshape_psf_shape11',
    'r_sdssshape_psf_shape22', 'r_sdssshape_psf_shape12',
    'i_pix_variance', 'i_sky_mean', 'i_sky_std', 'i_inputcount_value', 'i_inputcount_flag',
    'i_inputcount_flag_noinputs', 'i_inputcount_flag_badcentroid', 'i_pixelflags',
    'i_pixelflags_offimage', 'i_pixelflags_edge', 'i_pixelflags_bad',
    'i_pixelflags_interpolatedcenter', 'i_pixelflags_saturatedcenter', 'i_pixelflags_crcenter',
    'i_pixelflags_suspectcenter', 'i_pixelflags_bright_objectcenter', 'i_sdssshape_psf_shape11',
    'i_sdssshape_psf_shape22', 'i_sdssshape_psf_shape12',
    'z_pix_variance', 'z_sky_mean', 'z_sky_std', 'z_inputcount_value', 'z_inputcount_flag',
    'z_inputcount_flag_noinputs', 'z_inputcount_flag_badcentroid', 'z_pixelflags',
    'z_pixelflags_offimage', 'z_pixelflags_edge', 'z_pixelflags_bad',
    'z_pixelflags_interpolatedcenter', 'z_pixelflags_saturatedcenter', 'z_pixelflags_crcenter',
    'z_pixelflags_suspectcenter', 'z_pixelflags_bright_objectcenter', 'z_sdssshape_psf_shape11',
    'z_sdssshape_psf_shape22', 'z_sdssshape_psf_shape12',
    'y_pix_variance', 'y_sky_mean', 'y_sky_std', 'y_inputcount_value', 'y_inputcount_flag',
    'y_inputcount_flag_noinputs', 'y_inputcount_flag_badcentroid', 'y_pixelflags',
    'y_pixelflags_offimage', 'y_pixelflags_edge', 'y_pixelflags_bad',
    'y_pixelflags_interpolatedcenter', 'y_pixelflags_saturatedcenter', 'y_pixelflags_crcenter',
    'y_pixelflags_suspectcenter', 'y_pixelflags_bright_objectcenter', 'y_sdssshape_psf_shape11',
    'y_sdssshape_psf_shape22', 'y_sdssshape_psf_shape12'
]

S18A_RAND_USE = [1, 2, 3, 6, 7, 11, 12, 18, 37, 56, 75, 94]

S18A_RAND_DTYPE = [
    ("object_id", "int32"), ("ra", "float64"), ("dec", "float64"),
    ("tract", "int16"), ("patch", "int16"), ("isprimary", "bool"),
    ("adjust_density", "float64"), ("g_inputcount_value", "int16"),
    ("r_inputcount_value", "int16"), ("i_inputcount_value", "int16"),
    ("z_inputcount_value", "int16"), ("y_inputcount_value", "int16")
]


def downsample_randoms(rand_file, chunksize=1e6, seed=95064, csv=True,
                       downsample=False, verbose=True):
    """Down-sample the random catalogs from the HSC S18A data."""
    if not os.path.isfile(rand_file):
        raise IOError("# Can not find the particle table : %s" % rand_file)
    rand_pre, _ = os.path.splitext(rand_file)

    # Reduce the number of colunms and save as a numpy array
    rand_out = rand_pre + ".npy"
    rand_out_downsample = rand_pre + "_downsample.npy"

    if verbose:
        print("# Save the downsampled catalog to : %s" % rand_out)

    # Read the data
    rand_pchunks = pd.read_csv(
        rand_file, usecols=S18A_RAND_COLS, delim_whitespace=csv,
        names=S18A_RAND_USE, dtype=S18A_RAND_DTYPE, index_col=False,
        chunksize=chunksize)

    rand_pdframe = pd.concat(rand_pchunks)
    rand_array = rand_pdframe.values.ravel().view(dtype=S18A_RAND_DTYPE)

    # Save the result
    np.save(rand_out, rand_array)

    # Downsample
    if downsample:
        np.random.seed(seed)
        n_rand = int(len(rand_array) / 10)
        rand_downsample = np.random.choice(rand_array, n_rand, replace=False)

        # Save the result
        np.save(rand_out_downsample, rand_downsample)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'rand_file', type=str,
        help=('The particle catalog.'))

    parser.add_argument(
        '-c', '--chunk', dest='chunksize', type=int, default=1e6,
        help=('Size of the chunk when reading in the catalog.'))

    parser.add_argument(
        '-s', '--seed', dest='seed', help='Random seed',
        type=int, default=95064)

    parser.add_argument(
        '-v', '--verbose', dest='verbose',
        action="store_true", default=False)

    args = parser.parse_args()

    downsample_randoms(args.rand_file, chunksize=args.chunksize,
        seed=args.seed, verbose=args.verbose)
