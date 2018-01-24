"""
Module storing functions used to paint values of M10, M100 and M*_tot onto
model galaxies
"""

import numpy as np

from sm_halo_model import logms_halo_from_logmh_log_linear
from sm_minn_model import logms_inn_cam
from sm_mtot_model import logms_tot_from_logms_halo


def sigma_logms_from_logmh(logMhalo, sigms_a, sigms_b,
                           min_scatter=0.01):
    """Scatter of stellar mass at fixed halo mass.

    Assuming a simple log-log linear relation.

    Parameters
    ----------
    logMhalo : ndarray
        log10(Mhalo), for UM, use the true host halo mass.

    sigms_a : float
        Slope of the SigMs = a x logMh + b relation.

    sigms_b : float
        Normalization of the SigMs = a x logMh + b relation.

    min_scatter : float, optional
        Minimum allowed scatter.
        Sometimes the relation could lead to negative or super tiny
        scatter at high-mass end.  Use min_scatter to replace the
        unrealistic values.
        Default: 0.05

    """
    logSigMs = sigms_a * (np.array(logMhalo) - 15.3) + sigms_b

    logSigMs = np.where(logSigMs <= min_scatter, min_scatter, logSigMs)

    return logSigMs


def determine_logms_bins(logms, min_logms, max_logms, n_bins,
                         constant_bin=False, min_nobj_per_bin=30):
    """Bins for log stellar mass."""
    if constant_bin:
        # Constant log-linear bin size
        logms_bins = np.linspace(min_logms, max_logms, n_bins)
    else:
        # Try equal number object bin
        nobj_per_bin = np.ceil(len(logms) / n_bins)
        nobj_per_bin = (nobj_per_bin if nobj_per_bin > min_nobj_per_bin
                        else min_nobj_per_bin)

        logms_sort = np.sort(logms)
        logms_bins = logms_sort[
            np.where(np.arange(len(logms_sort)) % nobj_per_bin == 0)]
        logms_bins[-1] = logms_sort[-1]

    return logms_bins


def mass_prof_model_simple(um_mock,
                           logms_tot_obs,
                           logms_inn_obs,
                           parameters,
                           min_logms=11.5,
                           max_logms=12.2,
                           n_bins=10,
                           constant_bin=False,
                           logmh_col='logmh_vir',
                           logms_col='logms_tot',
                           min_scatter=0.02,
                           min_nobj_per_bin=30):
    """Mtot and Minn prediction using simple model."""
    # Model parameters
    shmr_a, shmr_b, sigms_a, sigms_b = parameters
    # Scatter of logMs_tot based on halo mass
    sig_logms_tot = sigma_logms_from_logmh(um_mock[logmh_col],
                                           sigms_a, sigms_b,
                                           min_scatter=min_scatter)

    # Get the bins for total stellar mass
    logms_tot_bins = determine_logms_bins(logms_tot_obs,
                                          min_logms, max_logms, n_bins,
                                          constant_bin=constant_bin,
                                          min_nobj_per_bin=min_nobj_per_bin)

    # Fraction of 'sm' + 'icl' to the total stellar mass of the halo
    # (including satellites)
    frac_tot_by_halo = 10.0 ** (um_mock['logms_tot'] - um_mock['logms_halo'])
    frac_inn_by_tot = 10.0 ** (um_mock['logms_gal'] - um_mock['logms_tot'])

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod = logms_halo_from_logmh_log_linear(um_mock[logmh_col],
                                                      shmr_a,
                                                      shmr_b,
                                                      sig_logms_tot,
                                                      log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(logms_halo_mod,
                                                  frac_tot_by_halo,
                                                  log_mass=True)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    # Given the modelled fraction of Ms,cen/Ms,tot from UM2,
    # predict the stellar mass in the inner region using
    # conditional abundance matching method.
    logms_inn_mod = logms_inn_cam(logms_tot_obs,
                                  logms_inn_obs,
                                  logms_tot_mod_all[mask_tot],
                                  frac_inn_by_tot[mask_tot],
                                  logms_tot_bins,
                                  sigma=0,
                                  num_required_gals_per_massbin=5)

    return (logms_inn_mod,
            logms_tot_mod_all,
            logms_halo_mod[mask_tot],
            mask_tot,
            um_mock[mask_tot])


def mass_prof_model_frac1(um_mock,
                          logms_tot_obs,
                          logms_inn_obs,
                          parameters,
                          min_logms=11.5,
                          max_logms=12.2,
                          n_bins=10,
                          constant_bin=False,
                          logmh_col='logmh_vir',
                          logms_col='logms_tot',
                          min_scatter=0.02,
                          min_nobj_per_bin=30):
    """Mtot and Minn prediction using simple model."""
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins, frac_exs) = parameters
    # Scatter of logMs_tot based on halo mass
    sig_logms_tot = sigma_logms_from_logmh(um_mock[logmh_col],
                                           sigms_a, sigms_b,
                                           min_scatter=min_scatter)

    # Get the bins for total stellar mass
    logms_tot_bins = determine_logms_bins(logms_tot_obs,
                                          min_logms, max_logms, n_bins,
                                          constant_bin=constant_bin,
                                          min_nobj_per_bin=min_nobj_per_bin)

    # Fraction of 'sm' + 'icl' to the total stellar mass of the halo
    # (including satellites)
    frac_tot_by_halo = 10.0 ** (um_mock['logms_tot'] - um_mock['logms_halo'])

    um_mins_predict = np.log10(frac_ins * um_mock['sm'] +
                               frac_exs * um_mock['icl'])

    frac_inn_by_tot = 10.0 ** (um_mins_predict - um_mock['logms_tot'])

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod = logms_halo_from_logmh_log_linear(um_mock[logmh_col],
                                                      shmr_a,
                                                      shmr_b,
                                                      sig_logms_tot,
                                                      log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (in-situ + ex-situ).
    logms_tot_mod_all = logms_tot_from_logms_halo(logms_halo_mod,
                                                  frac_tot_by_halo,
                                                  log_mass=True)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    # Given the modelled fraction of Ms,cen/Ms,tot from UM2,
    # predict the stellar mass in the inner region using
    # conditional abundance matching method.
    logms_inn_mod = logms_inn_cam(logms_tot_obs,
                                  logms_inn_obs,
                                  logms_tot_mod_all[mask_tot],
                                  frac_inn_by_tot[mask_tot],
                                  logms_tot_bins,
                                  sigma=0.,
                                  num_required_gals_per_massbin=5)

    return (logms_inn_mod,
            logms_tot_mod_all,
            logms_halo_mod[mask_tot],
            mask_tot,
            um_mock[mask_tot])


def mass_prof_model_frac2(um_mock,
                          logms_tot_obs,
                          logms_inn_obs,
                          parameters,
                          min_logms=11.5,
                          max_logms=12.2,
                          n_bins=10,
                          constant_bin=False,
                          logmh_col='logmh_vir',
                          logms_col='logms_tot',
                          min_scatter=0.02,
                          min_nobj_per_bin=30):
    """Mtot and Minn prediction using simple model.

    Without using the conditional abundance matching method.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins, frac_exs) = parameters
    # Scatter of logMs_tot based on halo mass
    sig_logms_tot = sigma_logms_from_logmh(um_mock[logmh_col],
                                           sigms_a, sigms_b,
                                           min_scatter=min_scatter)

    # Get the bins for total stellar mass
    logms_tot_bins = determine_logms_bins(logms_tot_obs,
                                          min_logms, max_logms, n_bins,
                                          constant_bin=constant_bin,
                                          min_nobj_per_bin=min_nobj_per_bin)

    # Fraction of 'sm' + 'icl' to the total stellar mass of the halo
    # (including satellites)
    frac_tot_by_halo = 10.0 ** (um_mock['logms_tot'] - um_mock['logms_halo'])

    frac_ins_by_tot = um_mock['sm'] / (um_mock['sm'] + um_mock['icl'])
    frac_exs_by_tot = um_mock['icl'] / (um_mock['sm'] + um_mock['icl'])

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod = logms_halo_from_logmh_log_linear(um_mock[logmh_col],
                                                      shmr_a,
                                                      shmr_b,
                                                      sig_logms_tot,
                                                      log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (in-situ + ex-situ).
    logms_tot_mod_all = logms_tot_from_logms_halo(logms_halo_mod,
                                                  frac_tot_by_halo,
                                                  log_mass=True)

    mtot_mod_all = 10.0 ** logms_tot_mod_all
    logms_inn_mod_all = np.log10(mtot_mod_all * frac_ins_by_tot * frac_ins +
                                 mtot_mod_all * frac_exs_by_tot * frac_exs)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    return (logms_inn_mod_all[mask_tot],
            logms_tot_mod_all,
            logms_halo_mod[mask_tot],
            mask_tot,
            um_mock[mask_tot])


def sm_profile_from_mhalo(logmh,
                          shmr_a,
                          shmr_b,
                          random_scatter_in_dex,
                          frac_tot_by_halo,
                          frac_inn_by_tot,
                          logms_tot_obs,
                          logms_inn_obs,
                          logms_tot_bins,
                          log_mass=True,
                          ngal_min=25):
    """
    Parameters
    ----------
    logmh : float or ndarray
        Float or Numpy array of shape (num_gals, ) of the halo mass of the
        galaxy in units of log10(Msun)
        (*not* in scaled h=1 units, but instead in "straight up" units
        calculated assuming little h equals the value appropriate for
        its cosmology)

    shmr_a : float
        Power law scaling index of smtot with mhalo

    shmr_b : float
        Normalization of the power law scaling between mhalo and smtot.

    random_scatter_in_dex : float
        Dispersion of the log-normal random noise added to smtot at fixed
        mhalo.

    frac_tot_by_halo: ndarray
        Fraction of the stellar mass of (central + ICL) to the total
        stellar mass within the halo (including satellites galaxies).
        (e.g., the ones predicted by UniverseMachine model.)

    frac_inn_by_tot: ndarray
        Fraction of the stellar mass in the inner region of galaxy
        to the total stellar mass of the galaxy (central + ICL).
        (e.g., the ones predicted by UniverseMachine model.)

    log_m100_data : ndarray
        Numpy array of shape (num_gals, ) storing log10(M100) of the
        HSC catalog

    log_m10_data : ndarray
        Numpy array of shape (num_gals, ) storing log10(M10) of the
        HSC catalog

    log_m100_bins : array
        Bins for log_m100_model

    ngal_min : int
        Minimum number of galaxy in each M100 bin.
        Default: 25

    Returns
    -------
    logms_inn_mod : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass in the inner region (e.g., within 10 kpc aperture),
        in unit of log10(Msun).

    logms_tot_mod : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass within a very large aperture (e.g., 100 kpc aperture),
        in unit of log10(Msun).

    logms_halo_mod : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass in the halo, including central galaxy, ICL and total satellite
        galaxy mass in units of log10(Msun).

    mask_tot : boolen
        Flags for useful items in the predicted stellar masses.
    """
    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod = logms_halo_from_logmh_log_linear(logmh,
                                                      shmr_a,
                                                      shmr_b,
                                                      random_scatter_in_dex,
                                                      log_mass=log_mass)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod = logms_tot_from_logms_halo(logms_halo_mod,
                                              frac_tot_by_halo,
                                              log_mass=log_mass)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod >= logms_tot_bins[0]) &
                (logms_tot_mod <= logms_tot_bins[-1]))

    # Given the modelled fraction of Ms,cen/Ms,tot from UM2,
    # predict the stellar mass in the inner region using
    # conditional abundance matching method.
    logms_inn_mod = logms_inn_cam(logms_tot_obs,
                                  logms_inn_obs,
                                  logms_tot_mod[mask_tot],
                                  frac_inn_by_tot[mask_tot],
                                  logms_tot_bins,
                                  sigma=0,
                                  num_required_gals_per_massbin=ngal_min)

    return (logms_inn_mod, logms_tot_mod,
            logms_halo_mod, mask_tot)
