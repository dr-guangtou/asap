#!/usr/bin/env python
"""
Precompute lensing pairs using UniverseMachine SMDPL catalog.
"""

import os
import argparse

import numpy as np

from astropy.table import Table

from asap import vagc


def main(um_file, ptl_file, wl_min_r=0.08, wl_max_r=50.0, wl_n_bins=22, verbose=True):
    """Pre-compute the particles pairs to calculate WL profile."""
    # Read in the UM mock catalog
    um_mock = Table(np.load(um_file))
    if verbose:
        print("# Load in UM mock catalog: {}".format(um_file))
        print("# Dealing with {} galaxies".format(len(um_mock)))
    # Read in the particle table
    sim_particles = Table(np.load(ptl_file))
    if verbose:
        print("# Load in particle table: {}".format(ptl_file))
        print("# Dealing with {} particles".format(len(sim_particles)))

    # Output file name
    um_pre, _ = os.path.splitext(um_file)
    ptl_pre, _ = os.path.splitext(ptl_file)
    n_ptl = ptl_pre.split('_')[-1]
    precompute_out = "{}_{}_r_{:4.2f}_{:4.1f}_{:2d}bins.npy".format(
        um_pre, n_ptl, wl_min_r, wl_max_r, wl_n_bins
    )
    if verbose:
        print("# Output file name : {}".format(precompute_out))

    # Run precompute
    if 'smdpl' in ptl_file:
        mass_encl = vagc.precompute_wl_smdpl(
            um_mock, sim_particles, wl_min_r=wl_min_r, wl_max_r=wl_max_r,
            wl_n_bins=wl_n_bins)
    elif 'mdpl2' in ptl_file:
        mass_encl = vagc.precompute_wl_mdpl2(
            um_mock, sim_particles, wl_min_r=wl_min_r, wl_max_r=wl_max_r,
            wl_n_bins=wl_n_bins)
    else:
        raise NameError("# Wrong simulation: [smdpl/mdpl2]")

    np.save(precompute_out, mass_encl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'um_file', type=str,
        help=('UniverseMachine snapshot file in .npy format'))

    parser.add_argument(
        'ptl_file', type=str,
        help=('Simulation particle table in .npy format'))

    parser.add_argument(
        '-l', '--r_low', dest='wl_min_r',
        help='Lower limit of the radial bin',
        type=float, default=0.08)

    parser.add_argument(
        '-u', '--r_upp', dest='wl_max_r',
        help='Upper limit of the radial bin',
        type=float, default=50.0)

    parser.add_argument(
        '-n', '--n_bins', dest='wl_n_bins',
        help='Number of the radial bin',
        type=int, default=22)

    args = parser.parse_args()

    main(args.um_file, args.ptl_file,
         wl_min_r=args.wl_min_r, wl_max_r=args.wl_max_r,
         wl_n_bins=args.wl_n_bins)
