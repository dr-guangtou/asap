"""Predict M100 and M10."""

import numpy as np

__all__ = ['frac_from_logmh', 'sigma_logms_from_logmh',
           'mass_model_frac4', 'mass_model_frac5', 'mass_model_frac6',
           'mass_model_frac7']


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


def mass_model_frac4(um_mock, parameters, random=False, min_logms=11.0,
                     logmh_col='logmh_vir', min_scatter=0.01):
    """Mtot and Minn prediction using simple model.

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

    # Mask for massive enough galaxies
    mask_use = logms_tot_mod_all >= min_logms
    logms_tot_mod = logms_tot_mod_all[mask_use]
    sig_logms = sig_logms_tot[mask_use]

    # Fraction of ex-situ component that goes into the inner aperture
    frac_exs = frac_from_logmh(um_mock[logmh_col][mask_use],
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


def mass_model_frac5(um_mock, parameters, random=False, min_logms=11.0,
                     logmh_col='logmh_vir', min_scatter=0.01):
    """Mtot and Minn prediction using simple model.

    This is the model with 8 free parameters:
        shmr_a, shmr_b:   determines a log-log linear SHMR between the 
                          halo mass and total stellar mass within the halo.
        sigms_a, sigms_b: determines the relation between scatter of
                          total stellar mass and the halo mass.
        frac_ins_a, frac_ins_b:  determine the fraction of the in-situ stars 
                                 in the inner 10 kpc.
        frac_exs_a, frac_exs_b:  determine the fraction of the ex-situ stars 
                                 in the inner 10 kpc.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins_a, frac_ins_b, 
     frac_exs_a, frac_exs_b) = parameters

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

    # Mask for massive enough galaxies
    mask_use = logms_tot_mod_all >= min_logms
    logms_tot_mod = logms_tot_mod_all[mask_use]
    sig_logms = sig_logms_tot[mask_use]

    # Fraction of in-situ component that goes into the inner aperture
    # We assume that the fraction depends on halo mass in a log-log linear manner
    frac_ins = frac_from_logmh(um_mock[logmh_col][mask_use],
                               frac_ins_a, frac_ins_b)               

    # Fraction of ex-situ component that goes into the inner aperture
    # We assume that the fraction depends on halo mass in a log-log linear manner
    frac_exs = frac_from_logmh(um_mock[logmh_col][mask_use],
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


def mass_model_frac6(um_mock, parameters, random=False, min_logms=11.0,
                     logmh_col='logmh_vir', min_scatter=0.01):
    """Mtot and Minn prediction using simple model.

    This is the model with 10 free parameters:
        shmr_a, shmr_b:   determines a log-log linear SHMR between the 
                          halo mass and total stellar mass within the halo.
        sigms_a, sigms_b: determines the relation between scatter of
                          total stellar mass and the halo mass.
        frac_ins_a, frac_ins_b:  determine the fraction of the in-situ stars 
                                 in the inner aperture.
        frac_exs_a, frac_exs_b:  determine the fraction of the ex-situ stars 
                                 in the inner aperture.
        frac_tot_a, frac_tot_b:  determine the fraction of the total stellar 
                                 mass in the outer aperture.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins_a, frac_ins_b, 
     frac_exs_a, frac_exs_b,
     frac_tot_a, frac_tot_b) = parameters

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

    # We assume certain fraction of stellar mass of the central galaxy is within 
    # the outer aperture, and we assume that fraction depends on halo mass 
    frac_tot = frac_from_logmh(um_mock[logmh_col], frac_tot_a, frac_tot_b)

    # This is the stellar mass within outer aperture to be compared with observation
    logms_out_mod_all = logms_tot_mod_all + np.log10(frac_tot)

    # Mask for massive enough galaxies
    mask_use = logms_out_mod_all >= min_logms
    logms_tot_mod = logms_tot_mod_all[mask_use]
    logms_out_mod = logms_out_mod_all[mask_use]
    sig_logms = sig_logms_tot[mask_use]

    # Fraction of in-situ component that goes into the inner aperture
    # We assume that the fraction depends on halo mass in a log-log linear manner
    frac_ins = frac_from_logmh(um_mock[logmh_col][mask_use],
                               frac_ins_a, frac_ins_b)               

    # Fraction of ex-situ component that goes into the inner aperture
    # We assume that the fraction depends on halo mass in a log-log linear manner
    frac_exs = frac_from_logmh(um_mock[logmh_col][mask_use],
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
        return logms_inn_mod, logms_out_mod, mask_use

    return logms_inn_mod, logms_out_mod, sig_logms, mask_use


def mass_model_frac7(um_mock, parameters, random=False, min_logms=11.0,
                     logmh_col='logmh_vir', min_scatter=0.01):
    """Mtot and Minn prediction using simple model.

    This is the model with 10 free parameters:
        shmr_a, shmr_b:   determines a log-log linear SHMR between the 
                          halo mass and total stellar mass within the halo.
        sigms_a, sigms_b: determines the relation between scatter of
                          total stellar mass and the halo mass.
        frac_ins:  Fraction of the in-situ stars in the inner aperture.
        frac_exs_a, frac_exs_b:  determine the fraction of the ex-situ stars 
                                 in the inner aperture.
        frac_tot_a, frac_tot_b:  determine the fraction of the total stellar 
                                 mass in the outer aperture.
    """
    # Model parameters
    (shmr_a, shmr_b, sigms_a, sigms_b,
     frac_ins, 
     frac_exs_a, frac_exs_b,
     frac_tot_a, frac_tot_b) = parameters

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

    # We assume certain fraction of stellar mass of the central galaxy is within 
    # the outer aperture, and we assume that fraction depends on halo mass 
    frac_tot = frac_from_logmh(um_mock[logmh_col], frac_tot_a, frac_tot_b)

    # This is the stellar mass within outer aperture to be compared with observation
    logms_out_mod_all = logms_tot_mod_all + np.log10(frac_tot)

    # Mask for massive enough galaxies
    mask_use = logms_out_mod_all >= min_logms
    logms_tot_mod = logms_tot_mod_all[mask_use]
    logms_out_mod = logms_out_mod_all[mask_use]
    sig_logms = sig_logms_tot[mask_use]

    # Fraction of ex-situ component that goes into the inner aperture
    # We assume that the fraction depends on halo mass in a log-log linear manner
    frac_exs = frac_from_logmh(um_mock[logmh_col][mask_use],
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
        return logms_inn_mod, logms_out_mod, mask_use

    return logms_inn_mod, logms_out_mod, sig_logms, mask_use
