"""Predict M100 and M10."""

import numpy as np

__all__ = ['frac_from_logmh', 'sigma_logms_from_logmh',
           'mass_model_frac4']


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


def mass_model_frac4(um_mock, parameters, random=False, min_logms=None,
                     logmh_col='logmh_vir', logms_col='logms_tot',
                     min_scatter=0.01):
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

    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    if random:
        logms_halo_mod_all = np.random.normal(
            loc=(shmr_a * um_mock[logmh_col] + shmr_b),
            scale=sig_logms_tot)
    else:
        logms_halo_mod_all = shmr_a * um_mock[logmh_col] + shmr_b

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod_all = logms_halo_mod_all + np.log10(um_mock['frac_cen_tot'])

    # Fraction of ex-situ component that goes into the inner aperture
    frac_exs = frac_from_logmh(um_mock[logmh_col],
                               frac_exs_a, frac_exs_b)

    logms_ins_inn = (logms_tot_mod_all + np.log10(um_mock['frac_ins_cen']) +
                     np.log10(frac_ins))
    logms_exs_inn = (logms_tot_mod_all + np.log10(um_mock['frac_exs_cen']) +
                     np.log10(frac_exs))

    logms_inn_mod_all = np.log10(10.0 ** logms_ins_inn +
                                 10.0 ** logms_exs_inn)

    if random:
        if min_logms is not None:
            mask_use = logms_tot_mod_all >= min_logms
            return (logms_inn_mod_all[mask_use],
                    logms_tot_mod_all[mask_use], mask_use)

        return logms_inn_mod_all, logms_tot_mod_all

    if min_logms is not None:
        mask_use = logms_tot_mod_all >= min_logms
        return logms_inn_mod_all, logms_tot_mod_all, sig_logms_tot, mask_use

    return logms_inn_mod_all, logms_tot_mod_all, sig_logms_tot
