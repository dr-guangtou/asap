"""Functions for prepare the catalogs of UniverseMachine model.

This includes codes to value added the catalog and prepare the particle
table.
"""

from __future__ import print_function

from time import time
from astropy.table import Column

import numpy as np

from model_predictions import (precompute_lensing_pairs,
                               total_stellar_mass_including_satellites)


def prep_um_catalog(um_mock, um_min_mvir=None, use_mvir=False):
    """Prepare the UniverseMachine mock catalog.

    The goal is to prepare a FITS catalog that include all
    necessary information.
    During the modeling part, we just need to load this catalog once.
    """
    # Sort the catalog based on the host halo ID
    um_mock.sort('halo_hostid')

    # Make a mask for central galaxies
    mask_central = um_mock['upid'] == -1
    um_mock.add_column(Column(data=mask_central,
                              name='mask_central'))

    # Add a column as the BCG+ICL mass
    um_mock.add_column(Column(data=(um_mock['sm'] + um_mock['icl']),
                              name='mtot_galaxy'))

    # Total stellar masses within a halo, including the satellites
    mstar_mhalo = total_stellar_mass_including_satellites(um_mock,
                                                          'mtot_galaxy')
    um_mock.add_column(Column(data=mstar_mhalo,
                              name='mstar_mhalo'))

    # Add log10(Mass)
    # Stellar mass
    um_mock.add_column(Column(data=np.log10(um_mock['sm']),
                              name='logms_gal'))
    um_mock.add_column(Column(data=np.log10(um_mock['icl']),
                              name='logms_icl'))
    um_mock.add_column(Column(data=np.log10(um_mock['mtot_galaxy']),
                              name='logms_tot'))
    um_mock.add_column(Column(data=np.log10(um_mock['mstar_mhalo']),
                              name='logms_halo'))
    # Halo mass
    um_mock.add_column(Column(data=np.log10(um_mock['mvir']),
                              name='logmh_vir'))
    um_mock.add_column(Column(data=np.log10(um_mock['mpeak']),
                              name='logmh_peak'))
    um_mock.add_column(Column(data=np.log10(um_mock['host_halo_mvir']),
                              name='logmh_host'))

    um_mock.rename_column('host_halo_mvir', 'mhalo_host')

    if um_min_mvir is not None:
        if use_mvir:
            um_mock_use = um_mock[um_mock['logmh_vir'] >= um_min_mvir]
        else:
            um_mock_use = um_mock[um_mock['logmh_host'] >= um_min_mvir]
    else:
        um_mock_use = um_mock

    return um_mock_use


def precompute_wl_smdpl(um_mock, sim_particles,
                        m_particle=9.63E7, n_particles_per_dim=3840,
                        box_size=400, wl_min_r=0.08, wl_max_r=50.0,
                        wl_n_bins=22, verbose=True):
    """Precompute lensing pairs using UniverseMachine SMDPL catalog.

    Parameters
    ----------
    sim_particles : astropy.table, optional
        External particle data catalog.

    um_min_mvir : float, optional
        Minimum halo mass used in computation.
        Default: None

    wl_min_r : float, optional
        Minimum radius for WL measurement.
        Default: 0.1

    wl_max_r : float, optional
        Maximum radius for WL measurement.
        Default: 40.0

    wl_n_bins : int, optional
        Number of bins in log(R) space.
        Default: 11

    """
    n_particles_tot = (n_particles_per_dim ** 3)
    if verbose:
        print("#   The simulation particle mass is %f" % m_particle)
        print("#   The number of particles is %d" %
              n_particles_tot)

    sim_downsampling_factor = (n_particles_tot /
                               float(len(sim_particles)))

    # Radius bins
    rp_bins = np.logspace(np.log10(wl_min_r),
                          np.log10(wl_max_r),
                          wl_n_bins)
    # Box size
    start = time()
    sim_mass_encl = precompute_lensing_pairs(
        um_mock['x'], um_mock['y'], um_mock['z'],
        sim_particles['x'], sim_particles['y'], sim_particles['z'],
        m_particle, sim_downsampling_factor,
        rp_bins, box_size)
    runtime = (time() - start)

    msg = ("Total runtime for {0} galaxies and {1:.1e} particles "
           "={2:.2f} seconds")
    print(msg.format(len(um_mock), len(sim_particles), runtime))

    return sim_mass_encl
