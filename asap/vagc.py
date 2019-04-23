"""Value added UM catalog."""
from __future__ import print_function, division, absolute_import

import copy

from time import time

import numpy as np
from astropy.table import Column, Table

from halotools.utils import crossmatch
from halotools.utils import group_member_generator
from halotools.mock_observables import total_mass_enclosed_per_cylinder

__all__ = ['value_added_mdpl2_mock', 'value_added_smdpl_mock',
           'total_stellar_mass_including_satellites', 'precompute_lensing_pairs',
           'prep_um_catalog', 'precompute_wl_smdpl']


def value_added_mdpl2_mock(fname):
    """Value added the UniverseMachine model catalog for MDPL2 simulation."""
    mdpl2_mock = Table(np.load(fname))

    mdpl2_mock.rename_column('m', 'mvir')
    mdpl2_mock.rename_column('mp', 'mpeak')
    mdpl2_mock.rename_column('id', 'halo_id')

    #  Apply periodic boundary conditions
    mdpl2_mock['x'] = np.where(mdpl2_mock['x'] < 0, 1000. + mdpl2_mock['x'],
                               mdpl2_mock['x'])
    mdpl2_mock['x'] = np.where(mdpl2_mock['x'] > 1000.,
                               mdpl2_mock['x'] - 1000., mdpl2_mock['x'])
    mdpl2_mock['y'] = np.where(mdpl2_mock['y'] < 0, 1000. + mdpl2_mock['y'],
                               mdpl2_mock['y'])
    mdpl2_mock['y'] = np.where(mdpl2_mock['y'] > 1000.,
                               mdpl2_mock['y'] - 1000., mdpl2_mock['y'])
    mdpl2_mock['z'] = np.where(mdpl2_mock['z'] < 0, 1000. + mdpl2_mock['z'],
                               mdpl2_mock['z'])
    mdpl2_mock['z'] = np.where(mdpl2_mock['z'] > 1000.,
                               mdpl2_mock['z'] - 1000., mdpl2_mock['z'])

    mdpl2_mock['halo_hostid'] = mdpl2_mock['halo_id']
    satmask = mdpl2_mock['upid'] != -1
    mdpl2_mock['halo_hostid'][satmask] = mdpl2_mock['upid'][satmask]

    idxA, idxB = crossmatch(mdpl2_mock['halo_hostid'], mdpl2_mock['halo_id'])
    mdpl2_mock['host_halo_mvir'] = mdpl2_mock['mvir']
    mdpl2_mock['host_halo_mvir'][idxA] = mdpl2_mock['mvir'][idxB]

    return mdpl2_mock


def value_added_smdpl_mock(smdpl_mock):
    """Value added the UniverseMachine model catalog for SMDPL simulation.

    This is designed for the short catalog, and the size of the SMDPL box
    is 400/h Mpc.
    """

    smdpl_mock.rename_column('m', 'mvir')
    smdpl_mock.rename_column('mp', 'mpeak')
    smdpl_mock.rename_column('id', 'halo_id')

    #  Apply periodic boundary conditions
    smdpl_mock['x'] = np.where(smdpl_mock['x'] < 0, 400. + smdpl_mock['x'],
                               smdpl_mock['x'])
    smdpl_mock['x'] = np.where(smdpl_mock['x'] > 400.,
                               smdpl_mock['x'] - 400., smdpl_mock['x'])
    smdpl_mock['y'] = np.where(smdpl_mock['y'] < 0, 400. + smdpl_mock['y'],
                               smdpl_mock['y'])
    smdpl_mock['y'] = np.where(smdpl_mock['y'] > 400.,
                               smdpl_mock['y'] - 400., smdpl_mock['y'])
    smdpl_mock['z'] = np.where(smdpl_mock['z'] < 0, 400. + smdpl_mock['z'],
                               smdpl_mock['z'])
    smdpl_mock['z'] = np.where(smdpl_mock['z'] > 400.,
                               smdpl_mock['z'] - 400., smdpl_mock['z'])

    smdpl_mock['halo_hostid'] = smdpl_mock['halo_id']
    satmask = smdpl_mock['upid'] != -1
    smdpl_mock['halo_hostid'][satmask] = smdpl_mock['upid'][satmask]

    idxA, idxB = crossmatch(smdpl_mock['halo_hostid'], smdpl_mock['halo_id'])
    smdpl_mock['host_halo_mvir'] = smdpl_mock['mvir']
    smdpl_mock['host_halo_mvir'][idxA] = smdpl_mock['mvir'][idxB]

    return smdpl_mock


def total_stellar_mass_including_satellites(gals, colname, hostid='halo_hostid'):
    """Sum up all the stellar mass in each host halo, including mass bound up in satellites.

    Whatever quantity is stored in the ``colname`` column will be summed within
    each host halo.

    Parameters
    -----------
    gals : Astropy Table
        Table storing the mock galaxy data,
        sorted in ascending order of the ``halo_hostid`` column.
        So make sure ``gals`` is sorted in the same order as results from
        when gals.sort('halo_hostid') has been called.

        Usually the ``gals`` table is the output of the `value_added_mock`
        function implemented in umachine_pyio.load_mock,
        which precomputes the ``halo_hostid`` column.

    colname : string
        Name of the column storing the stellar mass-like variable to be summed
        over.
        Typically this will be the name of the column storing ``sm`` + ``icl``.

    Returns
    -------
    total_stellar_mass : ndarray
        Numpy array of shape (num_gals, ) storing the total amount of stellar
        mass inside the host halo that each mock galaxy belongs to.
    """
    #  Bounds check the input galaxy table
    msg = "Input ``gals`` table must have a ``{0}`` column"
    assert colname in list(gals.keys()), msg.format(colname)

    msg = ("Input ``gals`` table must have a %s column\n" % hostid +
           "This column is pre-computed by the `value_added_mock` function" +
           "in the `umachine_pyio` package\n")
    assert hostid in list(gals.keys()), msg

    msg = ("Input ``gals`` table must be pre-sorted in ascending order " +
           "of ``halo_hostid`` column\n" +
           "You must call np.sort(gals) prior to calling this function.\n")
    assert np.all(np.diff(gals[hostid]) >= 0), msg

    #  Initialize the array to be filled in the loop
    total_stellar_mass = np.zeros(len(gals))

    #  Calculate generator that will yield our grouped mock data
    requested_columns = [colname]
    group_gen = group_member_generator(gals, hostid, requested_columns)

    #  Iterate over the generator and calculate total stellar mass in each halo
    for first, last, member_props in group_gen:
        stellar_mass_of_members = member_props[0]
        total_stellar_mass[first:last] = sum(stellar_mass_of_members)

    return total_stellar_mass


def precompute_lensing_pairs(galx, galy, galz, ptclx, ptcly, ptclz, particle_masses,
                             downsampling_factor, rp_bins, period):
    """Calculate the precomputed array to be used as the DeltaSigma kernels.

    Parameters
    -----------
    galx, galy, galz : xyz position of the entire galaxy sample
        in comoving units of Mpc/h, each array with shape (num_gals, ).
        All values must be between 0 <= xyz <= period.

    ptclx, ptcly, ptclz : xyz position of the downsampled particles
        in comoving units of Mpc/h, each array with shape (num_ptcl, ).
        All values must be between 0 <= xyz <= period.

    particle_masses : float or ndarray
        Single number or array of shape (num_ptcl, ) storing the
        mass of the downsampled particles in units of Msun/h

    downsampling_factor : float
        Factor by which the particles have been downsampled.
        For example, if the total number of particles in the entire simulation
        is num_ptcl_tot = 2048**3, then
        downsampling_factor = num_ptcl_tot/float(len(ptclx))

    rp_bins : ndarray
        Array of shape (num_bins, ) storing the bins in projected separation
        of the lensing calculation in comoving units of Mpc/h

    period : float
        Simulation box size in comoving units of Mpc/h

    Returns
    -------
    total_mass_encl : ndarray
        Numpy array of shape (num_gals, num_bins) storing the total enclosed
        mass in units of Msun/h within each of the num_bins cylinders
        surrounding each galaxy.
        This array can be used as the input to the halotools function
        `delta_sigma_from_precomputed_pairs`
    """
    galaxy_positions = np.vstack((galx, galy, galz)).T
    particle_positions = np.vstack((ptclx, ptcly, ptclz)).T

    return total_mass_enclosed_per_cylinder(
        galaxy_positions, particle_positions, particle_masses,
        downsampling_factor, rp_bins, period)


def prep_um_catalog(um_input, um_min_mvir=None, mhalo_col='logmh_peak'):
    """Prepare the UniverseMachine mock catalog.

    The goal is to prepare a FITS catalog that include all
    necessary information.
    During the modeling part, we just need to load this catalog once.
    """
    # Make a new copy, so don't modify the original catalog.
    um_mock = copy.deepcopy(um_input)

    # Sort the catalog based on the host halo ID
    um_mock.sort('halo_hostid')

    # Make a mask for central galaxies
    mask_central = um_mock['upid'] == -1
    um_mock.add_column(Column(data=mask_central, name='mask_central'))

    # Add a column as the BCG+ICL mass
    um_mock.add_column(Column(data=(um_mock['sm'] + um_mock['icl']),
                              name='mtot_galaxy'))

    # Total stellar masses within a halo, including the satellites
    mstar_mhalo = total_stellar_mass_including_satellites(um_mock, 'mtot_galaxy')
    um_mock.add_column(Column(data=mstar_mhalo, name='mstar_mhalo'))

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
        um_mock_use = um_mock[um_mock[mhalo_col] >= um_min_mvir]
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
        print("#   The number of particles is %d" % n_particles_tot)

    sim_downsampling_factor = (n_particles_tot / float(len(sim_particles)))

    # Radius bins
    rp_bins = np.logspace(np.log10(wl_min_r), np.log10(wl_max_r), wl_n_bins)

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
