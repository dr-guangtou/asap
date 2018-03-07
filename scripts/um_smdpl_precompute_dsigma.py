#!/usr/bin/env python
"""
Precompute lensing pairs using UniverseMachine SMDPL catalog.
"""

import os

import numpy as np

from time import time
from astropy.table import Table

from model_predictions import precompute_lensing_pairs


def precompute_wl_smdpl(um_mock, sim_particles, um_min_mvir=None,
                        m_particle=9.63E7, n_particles_per_dim=3840,
                        box_size=400, h0=0.6777,
                        wl_min_r=0.08, wl_max_r=50.0, wl_n_bins=22,
                        verbose=True):
    """
    Precompute lensing pairs using UniverseMachine SMDPL catalog.

    Parameters:
    -----------

    sim_particles : astropy.table, optional
        External particle data catalog.

    um_min_mvir : float, optional
        Minimum halo mass used in computation.
        Default: None

    wl_min_r : float, optional
        Minimum radius for WL measurement.
        Default: 0.08

    wl_max_r : float, optional
        Maximum radius for WL measurement.
        Default: 50.0

    wl_n_bins : int, optional
        Number of bins in log(R) space.
        Default: 22
    """
    n_particles_tot = (n_particles_per_dim ** 3)
    if verbose:
        print("#   The simulation particle mass is %f" % m_particle)
        print("#   The number of particles is %d" %
              n_particles_tot)

    sim_downsampling_factor = (n_particles_tot /
                               float(len(sim_particles)))

    if um_min_mvir is not None:
        sample = um_mock[um_mock['logmh_peak'] >= um_min_mvir]
    else:
        sample = um_mock

    # Radius bins
    rp_bins = np.logspace(np.log10(wl_min_r),
                          np.log10(wl_max_r),
                          wl_n_bins)
    # Box size
    sim_period = box_size

    start = time()
    sim_mass_encl = precompute_lensing_pairs(
        sample['x'], sample['y'], sample['z'],
        sim_particles['x'], sim_particles['y'], sim_particles['z'],
        m_particle, sim_downsampling_factor,
        rp_bins, sim_period)
    end = time()
    runtime = (end - start)

    msg = ("Total runtime for {0} galaxies and {1:.1e} particles "
           "={2:.2f} seconds")
    print(msg.format(len(sample), len(sim_particles), runtime))

    return sim_mass_encl


# Input table
um_smdpl_dir = '/Users/song/astro5/massive/dr16a/um2/um2_new'
um_smdpl_ptbl = os.path.join(um_smdpl_dir,
                             'um_smdpl_particles_0.7124_10m.npy')
um_smdpl_gtbl = os.path.join(um_smdpl_dir,
                             'um_smdpl_0.7124_new_vagc_mpeak_11.5.npy')

sim_particles = Table(np.load(um_smdpl_ptbl))
um_mock = Table(np.load(um_smdpl_gtbl))

# Output file
um_smdpl_precompute = os.path.join(
    um_smdpl_dir,
    'um_smdpl_0.7124_new_vagc_mpeak_11.5_10m_r_0.08_50_22bins.npy'
    )

# Run precompute
um_smdpl_mass_encl = precompute_wl_smdpl(um_mock, sim_particles)

np.save(um_smdpl_precompute, um_smdpl_mass_encl)
