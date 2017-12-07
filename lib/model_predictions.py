""" Module storing a few frequently used
derived quantities from the mock galaxies, like the
total stellar mass in each halo, and the lensing signal precomputation.
"""
import numpy as np
from halotools.utils import group_member_generator
from halotools.mock_observables import total_mass_enclosed_per_cylinder


def total_stellar_mass_including_satellites(gals, colname):
    """
    Sum up all the stellar mass in each host halo, including mass bound up in satellites.

    Whatever quantity is stored in the ``colname`` column will be summed within each host halo.

    Parameters
    -----------
    gals : Astropy Table
        Table storing the mock galaxy data,
        sorted in ascending order of the ``halo_hostid`` column.
        So make sure ``gals`` is sorted in the same order as results from
        when gals.sort('halo_hostid') has been called.

        Usually the ``gals`` table is the output of the `value_added_mock` function
        implemented in umachine_pyio.load_mock,
        which precomputes the ``halo_hostid`` column.

    colname : string
        Name of the column storing the stellar mass-like variable to be summed over.
        Typically this will be the name of the column storing ``sm`` + ``icl``.

    Returns
    -------
    total_stellar_mass : ndarray
        Numpy array of shape (num_gals, ) storing the total amount of stellar mass
        inside the host halo that each mock galaxy belongs to.
    """
    #  Bounds check the input galaxy table
    msg = "Input ``gals`` table must have a ``{0}`` column"
    assert colname in list(gals.keys()), msg.format(colname)

    msg = ("Input ``gals`` table must have a ``halo_hostid`` column\n"
        "This column is pre-computed by the `value_added_mock` function in the `umachine_pyio` package\n")
    assert 'halo_hostid' in list(gals.keys()), msg

    msg = ("Input ``gals`` table must be pre-sorted in ascending order of ``halo_hostid`` column\n"
        "You must call np.sort(gals) prior to calling this function.\n")
    assert np.all(np.diff(gals['halo_hostid']) >= 0), msg

    #  Initialize the array to be filled in the loop
    total_stellar_mass = np.zeros(len(gals))

    #  Calculate generator that will yield our grouped mock data
    grouping_key = 'halo_hostid'
    requested_columns = [colname]
    group_gen = group_member_generator(gals, grouping_key, requested_columns)

    #  Iterate over the generator and calculate total stellar mass in each halo
    for first, last, member_props in group_gen:
        stellar_mass_of_members = member_props[0]
        total_stellar_mass[first:last] = sum(stellar_mass_of_members)

    return total_stellar_mass


def precompute_lensing_pairs(galx, galy, galz, ptclx, ptcly, ptclz,
        particle_masses, downsampling_factor, rp_bins, period):
    """
    Calculate the precomputed array to be used as
    the Delta Sigma kernels in a lensing MCMC.

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
        is num_ptcl_tot = 2048**3, then downsampling_factor = num_ptcl_tot/float(len(ptclx))

    rp_bins : ndarray
        Array of shape (num_bins, ) storing the bins in projected separation
        of the lensing calculation in comoving units of Mpc/h

    period : float
        Simulation box size in comoving units of Mpc/h

    Returns
    -------
    total_mass_encl : ndarray
        Numpy array of shape (num_gals, num_bins) storing the total enclosed mass
        in units of Msun/h within each of the num_bins cylinders surrounding each galaxy.
        This array can be used as the input to the halotools function
        `delta_sigma_from_precomputed_pairs`
    """
    galaxy_positions = np.vstack((galx, galy, galz)).T
    particle_positions = np.vstack((ptclx, ptcly, ptclz)).T

    return total_mass_enclosed_per_cylinder(galaxy_positions, particle_positions,
        particle_masses, downsampling_factor, rp_bins, period)

