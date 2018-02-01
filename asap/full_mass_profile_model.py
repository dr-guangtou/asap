"""Predict M100 and M10."""

import numpy as np

from sm_halo_model import logms_halo_from_logmh_log_linear
from sm_minn_model import logms_inn_cam
from sm_mtot_model import logms_tot_from_logms_halo

__all__ = ['frac_from_logmh', 'sigma_logms_from_logmh',
           'determine_logms_bins', 'mass_prof_model_simple',
           'mass_prof_model_frac1', 'mass_prof_model_frac2',
           'mass_prof_model_frac3', 'mass_prof_model_frac4',
           'mass_prof_model_frac5']


def frac_from_logmh(logm_halo, frac_a, frac_b,
                    min_frac=0.0, max_frac=1.0):
    """Halo mass dependent fraction."""
    frac = frac_a * (np.array(logm_halo) - 15.3) + frac_b

    frac = np.where(frac <= min_frac, min_frac, frac)
    frac = np.where(frac >= max_frac, max_frac, frac)

    return frac


def sigma_logms_from_logmh(logm_halo, sigms_a, sigms_b,
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
    sigms = sigms_a * (np.array(logm_halo) - 15.3) + sigms_b

    sigms = np.where(sigms <= min_scatter, min_scatter, sigms)

    return sigms


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
    logms_halo_mod_all = logms_halo_from_logmh_log_linear(
        um_mock[logmh_col], shmr_a, shmr_b, sig_logms_tot, log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(
        logms_halo_mod_all, frac_tot_by_halo, log_mass=True)

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
            logms_halo_mod_all,
            mask_tot)


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
    logms_halo_mod_all = logms_halo_from_logmh_log_linear(
        um_mock[logmh_col], shmr_a, shmr_b, sig_logms_tot, log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(
        logms_halo_mod_all, frac_tot_by_halo, log_mass=True)

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
            logms_halo_mod_all,
            mask_tot)


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
    logms_halo_mod_all = logms_halo_from_logmh_log_linear(
        um_mock[logmh_col], shmr_a, shmr_b, sig_logms_tot, log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(
        logms_halo_mod_all, frac_tot_by_halo, log_mass=True)

    mtot_mod_all = 10.0 ** logms_tot_mod_all
    logms_inn_mod_all = np.log10(mtot_mod_all * frac_ins_by_tot * frac_ins +
                                 mtot_mod_all * frac_exs_by_tot * frac_exs)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    return (logms_inn_mod_all,
            logms_tot_mod_all,
            logms_halo_mod_all,
            mask_tot)


def mass_prof_model_frac3(um_mock,
                          logms_tot_obs,
                          logms_inn_obs,
                          parameters,
                          min_logms=11.5,
                          max_logms=12.2,
                          n_bins=10,
                          constant_bin=False,
                          logmh_col='logmh_vir',
                          logms_col='logms_tot',
                          min_scatter=0.01,
                          min_nobj_per_bin=30):
    """Mtot and Minn prediction using simple model.

    Without using the conditional abundance matching method.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins, frac_exs, frac_tot) = parameters
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
    logms_halo_mod_all = logms_halo_from_logmh_log_linear(
        um_mock[logmh_col], shmr_a, shmr_b, sig_logms_tot, log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(
        logms_halo_mod_all, frac_tot_by_halo, log_mass=True)

    mtot_mod_all = 10.0 ** logms_tot_mod_all

    logms_inn_mod_all = np.log10(mtot_mod_all * frac_ins_by_tot * frac_ins +
                                 mtot_mod_all * frac_exs_by_tot * frac_exs)

    logms_out_mod_all = np.log10(mtot_mod_all * frac_tot)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    return (logms_inn_mod_all,
            logms_out_mod_all,
            logms_halo_mod_all,
            mask_tot)


def mass_prof_model_frac4(um_mock,
                          logms_tot_obs,
                          logms_inn_obs,
                          parameters,
                          min_logms=11.5,
                          max_logms=12.2,
                          n_bins=10,
                          constant_bin=False,
                          logmh_col='logmh_vir',
                          logms_col='logms_tot',
                          min_scatter=0.01,
                          min_nobj_per_bin=30):
    """Mtot and Minn prediction using simple model.

    Without using the conditional abundance matching method.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins, frac_exs_a, frac_exs_b) = parameters

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

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod_all = logms_halo_from_logmh_log_linear(
        um_mock[logmh_col], shmr_a, shmr_b, sig_logms_tot, log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(
        logms_halo_mod_all, frac_tot_by_halo, log_mass=True)

    # Fraction of ex-situ component that goes into the inner aperture
    frac_exs = frac_from_logmh(um_mock[logmh_col],
                               frac_exs_a, frac_exs_b)

    ms_ins = (10.0 ** logms_tot_mod_all) * (um_mock['sm'] /
                                            (um_mock['sm'] + um_mock['icl']))
    ms_exs = (10.0 ** logms_tot_mod_all) * (um_mock['icl'] /
                                            (um_mock['sm'] + um_mock['icl']))

    logms_inn_mod_all = np.log10(ms_ins * frac_ins + ms_exs * frac_exs)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    return (logms_inn_mod_all,
            logms_tot_mod_all,
            logms_halo_mod_all,
            mask_tot)


def mass_prof_model_frac5(um_mock,
                          logms_tot_obs,
                          logms_inn_obs,
                          parameters,
                          min_logms=11.5,
                          max_logms=12.2,
                          n_bins=10,
                          constant_bin=False,
                          logmh_col='logmh_vir',
                          logms_col='logms_tot',
                          min_scatter=0.01,
                          min_nobj_per_bin=30):
    """Mtot and Minn prediction using simple model.

    Without using the conditional abundance matching method.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins_a, frac_ins_b,
     frac_exs_a, frac_exs_b) = parameters

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

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod_all = logms_halo_from_logmh_log_linear(
        um_mock[logmh_col], shmr_a, shmr_b, sig_logms_tot, log_mass=True)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_tot_from_logms_halo(
        logms_halo_mod_all, frac_tot_by_halo, log_mass=True)

    # Fraction of ex-situ component that goes into the inner aperture
    frac_ins = frac_from_logmh(um_mock[logmh_col],
                               frac_ins_a, frac_ins_b)
    frac_exs = frac_from_logmh(um_mock[logmh_col],
                               frac_exs_a, frac_exs_b)

    ms_ins = (10.0 ** logms_tot_mod_all) * (um_mock['sm'] /
                                            (um_mock['sm'] + um_mock['icl']))
    ms_exs = (10.0 ** logms_tot_mod_all) * (um_mock['icl'] /
                                            (um_mock['sm'] + um_mock['icl']))

    logms_inn_mod_all = np.log10(ms_ins * frac_ins + ms_exs * frac_exs)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot_mod_all >= logms_tot_bins[0]) &
                (logms_tot_mod_all <= logms_tot_bins[-1]))

    return (logms_inn_mod_all,
            logms_tot_mod_all,
            logms_halo_mod_all,
            mask_tot)
