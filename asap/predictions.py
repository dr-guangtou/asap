"""Module to deal with model predictions"""
from __future__ import print_function, division, unicode_literals

import numpy as np

from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM

from . import smf
from . import utils
from . import dsigma


__all__ = ['frac_from_logmh', 'sigma_logms_from_logmh', 'predict_mstar_basic',
           'predict_smf']


def frac_from_logmh(logm_halo, frac_a, frac_b,
                    min_frac=0.0, max_frac=1.0, pivot=15.3):
    """Halo mass dependent fraction.

    Parameters
    ----------
    logm_halo: numpy array
        Array of halo mass.
    frac_a: float
        Slope of the scaling relation.
    frac_b: float
        Intercept of scaling relation.
    min_frac: float, optional
        Minimum fraction allowed. Default: 0.0
    max_frac: float, optional
        Maximum fraction allowed. Default: 1.0
    pivot: float, optional
        Pivot halo mass for the scaling relation. Default: 15.3

    Returns
    -------
    frac: numpy array
        Fraction predicted by the scaling relation.

    """
    frac = frac_a * (np.array(logm_halo) - pivot) + frac_b

    frac = np.where(frac <= min_frac, min_frac, frac)
    frac = np.where(frac >= max_frac, max_frac, frac)

    return frac


def sigma_logms_from_logmh(logm_halo, sig_logms_a, sig_logms_b,
                           min_scatter=0.01, pivot=15.3):
    """Scatter of stellar mass at fixed halo mass.

    Assuming a simple log-log linear relation.

    Parameters
    ----------
    logm_halo: ndarray
        log10(Mhalo), for UM, use the true host halo mass.
    sig_logms_a: float
        Slope of the sig_logms = a x logMh + b relation.
    sig_logms_b: float
        Normalization of the sig_logms = a x logMh + b relation.
    min_scatter: float, optional
        Minimum allowed scatter.
        Sometimes the relation could lead to negative or super tiny
        scatter at high-mass end.  Use min_scatter to replace the
        unrealistic values. Default: 0.01
    pivot: float, optional
        Pivot halo mass. Default: 15.3

    Returns
    -------
    sig_logms: numpy array
        Scatter or uncertainties of the predicted stellar mass.

    """
    sig_logms = sig_logms_a * (np.array(logm_halo) - pivot) + sig_logms_b

    sig_logms = np.where(sig_logms <= min_scatter, min_scatter, sig_logms)

    return sig_logms


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

    Returns
    -------
    logms_inn_mod: numpy array
        Predicted stellar mass in the inner aperture.
    logms_tot_mod: numpy array
        Predicted total stellar mass of the galaxy.
    sig_logms: numpy array
        Uncertainties of the stellar masses.
    mask_use: boolen array
        Mask that indicates the galaxies with stellar mass predictions.

    Notes
    -----

    This is the default model with 7 free parameters:
        shmr_a, shmr_b:   determines a log-log linear SHMR between the
                          halo mass and total stellar mass within the halo.
        sig_logms_a, sig_logms_b: determines the relation between scatter of
                          total stellar mass and the halo mass.
        frac_ins:         fraction of the in-situ stars in the inner aperture.
        frac_exs_a, frac_exs_b:  determine the fraction of the ex-situ stars in
                                 the inner aperture.
    """
    # Model parameters
    assert len(parameters) == 7, "# Wrong parameter combinations."
    (shmr_a, shmr_b, sig_logms_a, sig_logms_b,
     frac_ins, frac_exs_a, frac_exs_b) = parameters

    # Scatter of logMs_tot based on halo mass
    sig_logms_tot = sigma_logms_from_logmh(
        um_mock[logmh_col], sig_logms_a, sig_logms_b, min_scatter=min_scatter)

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


def predict_smf(logms_mod_tot, logms_mod_inn, sig_logms, cfg, min_weight=0.4):
    """Predict SMFs weighted by mass uncertainties.

    Parameters
    ----------
    logms_mod_tot: numpy array
        Array of total stellar masses.
    logms_mod_inn: numpy array
        Array of stellar mass in the inner region.
    sig_logms: numpy array
        Uncertainties of stellar mass.
    cfg: dict
        Configuration parameters.
    min_weight: float, optional
        Minimum weight used to get SMF of inner aperture mass. Default: 0.4

    Returns
    -------
    smf_tot: numpy array
        Stellar mass function of the total stellar mass.
    smf_inn: numpy array
        Stellar mass function of the inner aperture mass.

    """
    # SMF of the predicted Mtot (M100 or MMax)

    # Expected number of galaxies in the
    ngal_expect = (cfg['obs']['ngal_use'] * cfg['um']['volume'] / cfg['obs']['volume'])

    # If the number of predicted galaxies with mass in the observed range is smaller
    # than 1/10 of the expected number, don't estimate SMF
    min_num = int(ngal_expect / 10)

    # If the number of useful galaxy is too small, return None
    if (logms_mod_tot > cfg['obs']['smf_tot_min']).sum() <= min_num:
        return None, None

    if (logms_mod_inn > cfg['obs']['smf_inn_min']).sum() <= min_num:
        return None, None

    smf_tot = smf.smf_sigma_mass_weighted(
        logms_mod_tot, sig_logms, cfg['um']['volume'], cfg['obs']['smf_tot_nbin'],
        cfg['obs']['smf_tot_min'], cfg['obs']['smf_tot_max'])

    weight = utils.mass_gaussian_weight(
        logms_mod_tot, sig_logms, cfg['obs']['smf_tot_min'], cfg['obs']['smf_tot_max'])

    # SMF of the predicted inner aperture mass
    smf_inn = smf.smf_sigma_mass_weighted(
        logms_mod_inn[weight > min_weight], sig_logms[weight > min_weight],
        cfg['um']['volume'], cfg['obs']['smf_inn_nbin'],
        cfg['obs']['smf_inn_min'], cfg['obs']['smf_inn_max'])

    return smf_tot, smf_inn


def um_get_dsigma(cfg, mock_use, mass_encl_use, mask=None, weight=None,
                  r_interp=None, mstar_lin=None):
    """Weak lensing dsigma profiles using pre-computed pairs.

    Parameters
    ----------
    cfg: dict
        Configurations of the data and model.
    mock_use: numpy array
        Mock galaxy catalog.
    mass_encl_use: numpy array
        Pre-computed pairs results for mock galaxies.
    mask: numpy array, optional
        Mask array that defines the subsample. Default: None
    weight: numpy array, optional
        Galaxy weight used in the calculation. Default: None
    r_interp : array, optional
        Radius array to interpolate to. Default: None
    mstar_lin : float, optional
        Linear mass for "point source". Default: None

    Returns
    -------
        DeltaSigma profile for UM mock galaxies.

    """
    # Make sure the data have the right shape
    assert mock_use.shape[0] == mass_encl_use.shape[0], ("Mock catalog and pre-compute "
                                                         "results are not consistent")

    # Cosmology
    um_cosmo = FlatLambdaCDM(H0=cfg['um']['h0'] * 100.0, Om0=cfg['um']['omega_m'])

    # Radius bins
    rp_bins = np.logspace(np.log10(cfg['um']['wl_minr']),
                          np.log10(cfg['um']['wl_maxr']), cfg['um']['wl_nbin'])

    #  Use the mask to get subsample positions and pre-computed pairs
    if mask is not None:
        subsample = mock_use[mask]
        subsample_mass_encl_precompute = mass_encl_use[mask, :]
    else:
        subsample = mock_use
        subsample_mass_encl_precompute = mass_encl_use

    # Get the positions of the mock galaxies
    subsample_positions = np.vstack([subsample['x'], subsample['y'], subsample['z']]).T

    rp_ht_units, ds_ht_units = dsigma.delta_sigma_from_precomputed_pairs(
        subsample_positions, subsample_mass_encl_precompute,
        rp_bins, cfg['um']['lbox'], cosmology=um_cosmo,
        weight=weight)

    # Unit conversion
    ds_phys_msun_pc2 = ((1. + cfg['um']['redshift']) ** 2 *
                        (ds_ht_units * um_cosmo.h) / (1e12))

    rp_phys = ((rp_ht_units) / (abs(1. + cfg['um']['redshift']) * um_cosmo.h))

    # Add the point source term
    if mstar_lin is not None:
        # Only add to the inner most data point.
        ds_phys_msun_pc2[0] += (mstar_lin / 1e12 / (np.pi * (rp_phys[0] ** 2.0)))

    if r_interp is not None:
        intrp = interpolate.interp1d(rp_phys, ds_phys_msun_pc2, kind='cubic', bounds_error=False)
        return intrp(r_interp)

    return ds_phys_msun_pc2


def get_single_dsigma_profile(cfg, mock_use, mass_encl_use, obs_prof, logms_mod_tot, logms_mod_inn,
                              sig_logms, min_weight=0.03, min_num=3, add_stellar=False):
    """Weigted delta sigma profiles.

    Parameters
    ----------
    cfg: dict
        Configurations of the data and model.
    mock_use: numpy array
        Mock galaxy catalog.
    mass_encl_use: numpy array
        Pre-computed pairs results for mock galaxies.
    obs_prof: numpy array
        Observed DeltaSigma information for one mass bin.
    logms_mod_tot: numpy array
        Array of total stellar masses.
    logms_mod_inn: numpy array
        Array of stellar mass in the inner region.
    sig_logms: numpy array
        Uncertainties of stellar mass.
    min_weight: float, optional
        Minimum weight value for calculating DeltaSigma profile. Default: 0.03
    min_num: int, optional
        Minimum number of useful mock galaxy to calculate DeltaSigma profile. Default: 3
    add_stellar: boolen, optional
        Whether add stellar mass to the profile. Default: False

    Returns
    -------
        DeltaSigma profile of mock galaxies in a mass bin.

    """
    # "Point source" term for the central galaxy
    weight = np.array(
        utils.mtot_minn_weight(logms_mod_tot, logms_mod_inn, sig_logms,
                               obs_prof['min_logm1'], obs_prof['max_logm1'],
                               obs_prof['min_logm2'], obs_prof['max_logm2']))

    # Exclude the ones with tiny weight values.
    mask = (weight >= min_weight)

    if mask.sum() >= min_num:
        if add_stellar:
            # Use the weighted mean total stellar mass
            mstar_lin = 10.0 ** (np.sum(weight[mask] * logms_mod_tot[mask]) /
                                 np.sum(weight[mask]))
        else:
            mstar_lin = None

        return um_get_dsigma(
            cfg, mock_use[mask], mass_encl_use[mask, :],
            weight=weight[mask], r_interp=obs_prof['r_mpc'], mstar_lin=mstar_lin)

    return np.zeros(len(obs_prof['r_mpc']))


def predict_dsigma_profiles(cfg, obs_dsigma, mock_use, mass_encl_use, logms_mod_tot, logms_mod_inn,
                            mask=None, sig_logms=None, min_num=3, add_stellar=False):
    """WL profiles to compare with observations.

    Parameters
    ----------
    cfg: dict
        Configurations of the data and model.
    obs_dsigma: numpy array
        Observed DeltaSigma profiles.
    mock_use: numpy array
        Mock galaxy catalog.
    mass_encl_use: numpy array
        Pre-computed pairs results for mock galaxies.
    logms_mod_tot : ndarray
        Total stellar mass (e.g. M100) predicted by UM.
    logms_mod_inn : ndarray
        Inner stellar mass (e.g. M10) predicted by UM.
    sig_logms: numpy array, optional
        Uncertainties of stellar mass. Default: None
    mask: bool array, optional
        Mask for the input mock catalog and precomputed WL pairs. Default: None
    min_num: int, optional
        Minimum requred galaxies in each bin to estimate WL profile. Default: 5
    add_stellar: boolen, optional
        Whether add stellar mass to the profile. Default: False

    Returns
    -------
        A list of DeltaSigma profiles for UM mock galaxies in different mass bins.

    """
    # The mock catalog and precomputed mass files for subsamples
    if mask is not None:
        mock_use = mock_use[mask]
        mass_encl_use = mass_encl_use[mask, :]

    if sig_logms is None:
        sig_logms = np.zeros(len(mock_use))

    return [get_single_dsigma_profile(
        cfg, mock_use, mass_encl_use, obs_prof, logms_mod_tot, logms_mod_inn, sig_logms,
        min_num=min_num, add_stellar=add_stellar) for obs_prof in obs_dsigma]
