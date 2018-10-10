"""Module to deal with model predictions"""
from __future__ import print_function, division, unicode_literals

import numpy as np

from scipy import interpolate


def frac_from_logmh(logm_halo, frac_a, frac_b,
                    min_frac=0.0, max_frac=1.0, pivot=15.3):
    """Halo mass dependent fraction."""
    frac = frac_a * (np.array(logm_halo) - pivot) + frac_b

    frac = np.where(frac <= min_frac, min_frac, frac)
    frac = np.where(frac >= max_frac, max_frac, frac)

    return frac


def sigma_logms_from_logmh(logm_halo, sigms_a, sigms_b,
                           min_scatter=0.01, pivot=15.3):
    """Scatter of stellar mass at fixed halo mass.

    Assuming a simple log-log linear relation.

    Parameters
    ----------
    logMhalo: ndarray
        log10(Mhalo), for UM, use the true host halo mass.
    sigms_a: float
        Slope of the SigMs = a x logMh + b relation.
    sigms_b: float
        Normalization of the SigMs = a x logMh + b relation.
    min_scatter: float, optional
        Minimum allowed scatter.
        Sometimes the relation could lead to negative or super tiny
        scatter at high-mass end.  Use min_scatter to replace the
        unrealistic values. Default: 0.01
    pivot: float, optional
        Pivot halo mass.

    """
    sigms = sigms_a * (np.array(logm_halo) - pivot) + sigms_b

    sigms = np.where(sigms <= min_scatter, min_scatter, sigms)

    return sigms


def predict_mstar_basic(um_mock, parameters, random=False, min_logms=11.0,
                        logmh_col='logmh_host', min_scatter=0.01, pivot=0.0):
    """Mtot and Minn prediction using simple model.

    Parameters
    ----------
    um_mock : numpy array
        Galaxy catalog from UniverseMachine.
    parameters : numpy array
        Array of model parameters.
    random: boolen, optional
        Using random realization or not. Default: False
    min_logms: float, optional
        Minimum stellar mass to return. Default: 11.0
    min_scatter: float, optional
        Minimum scatter of logMhalo-logM* relation allowed. Default: 0.01
    logmh_col: string, optional
        Name of the halo mass column. Default: 'logmh_host'
    pivot: float, optional
        Pivot halo mass. Default: 0.0

    Notes
    -----

    This is the default model with 7 free parameters:
        shmr_a, shmr_b:   determines a log-log linear SHMR between the
                          halo mass and total stellar mass within the halo.
        sigms_a, sigms_b: determines the relation between scatter of
                          total stellar mass and the halo mass.
        frac_ins:         fraction of the in-situ stars in the inner aperture.
        frac_exs_a, frac_exs_b:  determine the fraction of the ex-situ stars in
                                 the inner aperture.
    """
    # Model parameters
    assert len(parameters) == 7, "# Wrong parameter combinations."
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins, frac_exs_a, frac_exs_b) = parameters

    # Scatter of logMs_tot based on halo mass
    sig_logms_tot = sigma_logms_from_logmh(
        um_mock[logmh_col], sigms_a, sigms_b, min_scatter=min_scatter)

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    if random:
        logms_halo_mod_all = np.random.normal(
            loc=(shmr_a * (um_mock[logmh_col] - pivot) + shmr_b),
            scale=sig_logms_tot)
    else:
        logms_halo_mod_all = shmr_a * (um_mock[logmh_col] - pivot) + shmr_b

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_halo_mod_all + np.log10(um_mock['frac_cen_tot'])

    # Mask for massive enough galaxies
    mask_use = logms_tot_mod_all >= min_logms
    logms_tot_mod = logms_tot_mod_all[mask_use]
    sig_logms = sig_logms_tot[mask_use]

    # Fraction of ex-situ component that goes into the inner aperture
    frac_exs = frac_from_logmh(um_mock['logmh_peak'][mask_use],
                               frac_exs_a, frac_exs_b)

    # Stellar mass for each component
    logms_ins_inn = (logms_tot_mod +
                     np.log10(um_mock['frac_ins_cen'][mask_use]) +
                     np.log10(frac_ins))
    logms_exs_inn = (logms_tot_mod +
                     np.log10(um_mock['frac_exs_cen'][mask_use]) +
                     np.log10(frac_exs))

    logms_inn_mod = np.log10(10.0 ** logms_ins_inn + 10.0 ** logms_exs_inn)

    if random:
        return logms_inn_mod, logms_tot_mod, mask_use

    return logms_inn_mod, logms_tot_mod, sig_logms, mask_use
