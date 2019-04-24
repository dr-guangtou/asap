#!/usr/bin/env python
"""This script will read the dark matter particle table for the SMDPL
simulation, and downsample it for our model.
"""

import os
import argparse

import numpy as np
import pandas as pd

def downsample_particles(ptbl_file, n_million, seed=95064, csv=False, verbose=True):
    """Down-sample the partile files from the DM simulation."""
    if not os.path.isfile(ptbl_file):
        raise IOError("# Can not find the particle table : %s" % ptbl_file)
    ptbl_pre, ptbl_ext = os.path.splitext(ptbl_file)

    # Reduce the number of colunms and save as a numpy array
    ptbl_out = ptbl_pre + "_downsample_%.1fm.npy" % n_million
    if verbose:
        print("# Save the downsampled catalog to : %s" % ptbl_out)

    # Data format for output
    particle_table_dtype = [
        ("x", "float64"), ("y", "float64"), ("z", "float64")]

    if csv or ptbl_ext == '.csv':
        use_csv = True
    else:
        use_csv = False

    # Read the data
    chunksize = 1000000
    ptbl_pchunks = pd.read_csv(
        ptbl_file, usecols=[0, 1, 2], delim_whitespace=use_csv,
        names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'id'],
        dtype=particle_table_dtype, index_col=False,
        chunksize=chunksize)

    ptbl_pdframe = pd.concat(ptbl_pchunks)
    ptbl_array = ptbl_pdframe.values.ravel().view(dtype=particle_table_dtype)

    # Downsample
    np.random.seed(seed)
    ptbl_downsample = np.random.choice(ptbl_array, int(n_million * 1e6), replace=False)

    # Save the result
    np.save(ptbl_out, ptbl_downsample)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ptbl_file', type=str,
        help=('The particle catalog.'))

    parser.add_argument(
        'n_million', type=float,
        help=('Downsample the catalog to N x millions particles.'))

    parser.add_argument(
        '-s', '--seed', dest='seed',
        help='Random seed',
        type=int, default=95064)

    parser.add_argument(
        '-v', '--verbose', dest='verbose',
        action="store_true", default=False)

    parser.add_argument(
        '-c', '--csv', dest='csv',
        action="store_true", default=False)

    args = parser.parse_args()

    downsample_particles(args.ptbl_file, args.n_million,
                         csv=args.csv, seed=args.seed, verbose=args.verbose)
