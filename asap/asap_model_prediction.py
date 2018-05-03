"""Predictions of the A.S.A.P. model."""

from __future__ import print_function, division, unicode_literals

import numpy as np

from scipy import interpolate

from stellar_mass_function import get_smf_bootstrap, \
    smf_sigma_mass_weighted
from full_mass_profile_model import mass_prof_model_simple, \
    mass_prof_model_frac1, mass_prof_model_frac2, \
    mass_prof_model_frac3, mass_prof_model_frac4, \
    mass_prof_model_frac5, mass_prof_model_frac6, mass_prof_model_frac7
from um_model_plot import plot_mtot_minn_smf, plot_dsigma_profiles
from asap_mass_model import mass_model_frac4, mass_model_frac5, \
    mass_model_frac6, mass_model_frac7
from asap_delta_sigma import delta_sigma_from_precomputed_pairs
from asap_utils import mtot_minn_weight, mass_gaussian_weight


__all__ = ['asap_predict_mass', 'asap_predict_smf',
           'asap_single_dsigma', 'asap_um_dsigma',
           'asap_predict_dsigma', 'asap_predict_model',
           'asap_predict_model_prob', 'asap_predict_mass_prob',
           'asap_predict_smf_prob', 'asap_single_dsigma_weight',
           'asap_predict_mhalo']


def asap_predict_mass_prob(parameters, cfg, um_mock, return_all=False):
    """Predict stellar masses in different apertures."""
    if cfg['model_type'] == 'frac4':
        # 7 free parameters
        return mass_model_frac4(
            um_mock, parameters,
            random=False,
            min_logms=cfg['obs_min_mtot'],
            logmh_col=cfg['um_halo_col'],
            min_scatter=cfg['um_min_scatter'])
    elif cfg['model_type'] == 'frac5':
        # 8 free parameters
        # In-situ fraction in inner aperture depends on halo mass
        return mass_model_frac5(
            um_mock, parameters,
            random=False,
            min_logms=cfg['obs_min_mtot'],
            logmh_col=cfg['um_halo_col'],
            min_scatter=cfg['um_min_scatter'])
    elif cfg['model_type'] == 'frac6':
        # 10 free parameters
        # In-situ fraction in inner aperture depends on halo mass
        # Fraction of total stellar mass in outer aperture depends on halo mass 
        return mass_model_frac6(
            um_mock, parameters,
            random=False,
            min_logms=cfg['obs_min_mtot'],
            logmh_col=cfg['um_halo_col'],
            min_scatter=cfg['um_min_scatter'])
    elif cfg['model_type'] == 'frac7':
        # 9 free parameters
        # In-situ fraction is fixed
        # Fraction of total stellar mass in outer aperture depends on halo mass 
        return mass_model_frac7(
            um_mock, parameters,
            random=False,
            min_logms=cfg['obs_min_mtot'],
            logmh_col=cfg['um_halo_col'],
            min_scatter=cfg['um_min_scatter'])
    else:
        raise Exception("!! Wrong model: frac4")


def asap_predict_mass(parameters, cfg, obs_data, um_data,
                      constant_bin=False):
    """
    Predict stellar masses in different apertures.

    Parameters
    ----------
    parameters : array, list, or tuple
        Model parameters.

    cfg : dict
        Configurations of the data and model.

    obs_data: dict
        Dictionary for observed data.

    um_data: dict
        Dictionary for UniverseMachine data.

    constant_bin : boolen
        Whether to use constant bin size for logMs_tot or not.

    """
    if cfg['model_type'] == 'simple':
        return mass_prof_model_simple(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac1':
        return mass_prof_model_frac1(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac2':
        return mass_prof_model_frac2(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac3':
        return mass_prof_model_frac3(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac4':
        return mass_prof_model_frac4(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac5':
        return mass_prof_model_frac5(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac6':
        return mass_prof_model_frac6(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac7':
        return mass_prof_model_frac7(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=cfg['obs_smf_tot_min'],
            max_logms=cfg['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    else:
        raise Exception("# Wrong model choice! ")


def asap_predict_smf_prob(logms_mod_tot, logms_mod_inn, sigms, cfg):
    """Predict SMFs weighted by mass uncertainties."""
    # SMF of the predicted Mtot (M100 or MMax)

    # If the number of useful galaxy is too small, return None
    if (logms_mod_tot > cfg['obs_smf_tot_min']).sum() <= 100:
        return None, None

    if (logms_mod_inn > cfg['obs_smf_inn_min']).sum() <= 100:
        return None, None

    um_smf_tot = smf_sigma_mass_weighted(logms_mod_tot, sigms,
                                         cfg['um_volume'],
                                         cfg['obs_smf_tot_nbin'],
                                         cfg['obs_smf_tot_min'],
                                         cfg['obs_smf_tot_max'])

    weight = mass_gaussian_weight(
        logms_mod_tot, sigms, cfg['obs_smf_tot_min'], cfg['obs_smf_tot_max'])

    # SMF of the predicted Minn (M10x)
    um_smf_inn = smf_sigma_mass_weighted(logms_mod_inn[weight > 0.45],
                                         sigms[weight > 0.45],
                                         cfg['um_volume'],
                                         cfg['obs_smf_inn_nbin'],
                                         cfg['obs_smf_inn_min'],
                                         cfg['obs_smf_inn_max'])

    return um_smf_tot, um_smf_inn


def asap_predict_smf(logms_mod_tot, logms_mod_inn, cfg):
    """Stellar mass functions of Minn and Mtot predicted by UM."""
    # SMF of the predicted Mtot (M100 or MMax)
    um_smf_tot = get_smf_bootstrap(logms_mod_tot,
                                   cfg['um_volume'],
                                   cfg['obs_smf_tot_nbin'],
                                   cfg['obs_smf_tot_min'],
                                   cfg['obs_smf_tot_max'],
                                   n_boots=1)

    # SMF of the predicted Minn (M10)
    um_smf_inn = get_smf_bootstrap(logms_mod_inn,
                                   cfg['um_volume'],
                                   cfg['obs_smf_inn_nbin'],
                                   cfg['obs_smf_inn_min'],
                                   cfg['obs_smf_inn_max'],
                                   n_boots=1)

    return np.array(um_smf_tot['smf']), np.array(um_smf_inn['smf'])


def asap_um_dsigma(cfg, mock_use, mass_encl_use,
                   mask=None, weight=None,
                   verbose=False, r_interp=None, mstar_lin=None):
    """Weak lensing dsigma profiles using pre-computed pairs.

    Parameters
    ----------
    cfg : dict
        Configurations of the data and model.

    mask : ndarray
        Mask array that defines the subsample.

    r_interp : array, optional
        Radius array to interpolate to.
        Default: None

    mstar_lin : float, optional
        Linear mass for "point source".
        Default: None

    """
    # Cosmology
    um_cosmo = cfg['um_cosmo']

    # Radius bins
    rp_bins = np.logspace(np.log10(cfg['um_wl_minr']),
                          np.log10(cfg['um_wl_maxr']), cfg['um_wl_nbin'])

    #  Use the mask to get subsample positions and pre-computed pairs
    if mask is not None:
        subsample = mock_use[mask]
        subsample_mass_encl_precompute = mass_encl_use[mask, :]
    else:
        subsample = mock_use
        subsample_mass_encl_precompute = mass_encl_use

    subsample_positions = np.vstack([subsample['x'],
                                     subsample['y'],
                                     subsample['z']]).T

    rp_ht_units, ds_ht_units = delta_sigma_from_precomputed_pairs(
        subsample_positions, subsample_mass_encl_precompute,
        rp_bins, cfg['um_lbox'], cosmology=um_cosmo,
        weight=weight)

    # Unit conversion
    ds_phys_msun_pc2 = ((1. + cfg['um_redshift']) ** 2 *
                        (ds_ht_units * um_cosmo.h) / (1e12))

    rp_phys = ((rp_ht_units) / (abs(1. + cfg['um_redshift']) * um_cosmo.h))

    # Add the point source term
    if mstar_lin is not None:
        ds_phys_msun_pc2[0] += (
            mstar_lin / 1e12 / (np.pi * (rp_phys ** 2.0))
            )

    if r_interp is not None:
        intrp = interpolate.interp1d(rp_phys, ds_phys_msun_pc2,
                                     kind='cubic', bounds_error=False)
        return intrp(r_interp)

    return ds_phys_msun_pc2


def asap_single_mhalo(mock_use, obs_prof,
                      logms_mod_tot, logms_mod_inn):
    """Mhalo and scatter in one bin."""
    bin_mask = ((logms_mod_tot >= obs_prof.low_mtot) &
                (logms_mod_tot <= obs_prof.upp_mtot) &
                (logms_mod_inn >= obs_prof.low_minn) &
                (logms_mod_inn <= obs_prof.upp_mtot))

    # Just return the logMHalo and its scatter in each bin
    return (np.nanmedian(mock_use['logmh_vir'][bin_mask]),
            np.nanstd(mock_use['logmh_vir'][bin_mask]))


def asap_single_dsigma_weight(cfg, mock_use, mass_encl_use, obs_prof,
                              logms_mod_tot, logms_mod_inn, sig_logms,
                              add_stellar=False):
    """Weigted delta sigma profiles."""
    # "Point source" term for the central galaxy
    weight = np.array(
        mtot_minn_weight(logms_mod_tot, logms_mod_inn, sig_logms,
                         obs_prof.low_mtot, obs_prof.upp_mtot,
                         obs_prof.low_minn, obs_prof.upp_mtot))

    # TODO: Apply a mask
    mask = (weight >= 0.03)

    if mask.sum() >= 2:
        if add_stellar:
            mstar_lin = 10.0 ** (np.sum(weight[mask] * logms_mod_tot[mask]) /
                                 np.sum(weight[mask]))
        else:
            mstar_lin = None

        return asap_um_dsigma(
            cfg, mock_use[mask], mass_encl_use[mask, :],
            weight=weight[mask], r_interp=obs_prof.r, mstar_lin=mstar_lin)

    if add_stellar:
        mstar_lin = 10.0 ** (np.sum(weight * logms_mod_tot) /
                             np.sum(weight))
    else:
        mstar_lin = None

    return asap_um_dsigma(
        cfg, mock_use, mass_encl_use,
        weight=weight, r_interp=obs_prof.r, mstar_lin=mstar_lin)


def asap_single_dsigma(cfg, mock_use, mass_encl_use, obs_prof,
                       logms_mod_tot, logms_mod_inn,
                       um_wl_min_ngal=15, verbose=False,
                       add_stellar=False):
    """Individual WL profile for UM galaxies."""
    bin_mask = ((logms_mod_tot >= obs_prof.low_mtot) &
                (logms_mod_tot <= obs_prof.upp_mtot) &
                (logms_mod_inn >= obs_prof.low_minn) &
                (logms_mod_inn <= obs_prof.upp_mtot))

    # "Point source" term for the central galaxy
    if add_stellar:
        mstar_lin = np.nanmedian(10.0 * logms_mod_tot[bin_mask])
    else:
        mstar_lin = None

    if np.sum(bin_mask) <= um_wl_min_ngal:
        wl_prof = np.zeros(len(obs_prof.r))
        if verbose:
            print("# Not enough UM galaxy in bin %d !" % obs_prof.bin_id)
    else:
        return asap_um_dsigma(
            cfg, mock_use, mass_encl_use, mask=bin_mask,
            r_interp=obs_prof.r, mstar_lin=mstar_lin)


def asap_predict_mhalo(obs_dsigma, mock_use,
                       logms_mod_tot, logms_mod_inn):
    """Halo mass and its scatter in each bin.

    Parameters
    ----------
    logms_mod_tot : ndarray
        Total stellar mass (e.g. M100) predicted by UM.

    logms_mod_inn : ndarray
        Inner stellar mass (e.g. M10) predicted by UM.

    mask_tot : bool array
        Mask for the input mock catalog and precomputed WL pairs.

    um_wl_min_ngal : int, optional
        Minimum requred galaxies in each bin to estimate WL profile.

    """
    # The mock catalog and precomputed mass files for subsamples
    return [asap_single_mhalo(mock_use, obs_prof,
                              logms_mod_tot, logms_mod_inn)
            for obs_prof in obs_dsigma]


def asap_predict_dsigma(cfg, obs_data, mock_use, mass_encl_use,
                        logms_mod_tot, logms_mod_inn,
                        mask=None, sig_logms=None,
                        um_wl_min_ngal=15, verbose=False,
                        add_stellar=False):
    """WL profiles to compare with observations.

    Parameters
    ----------
    logms_mod_tot : ndarray
        Total stellar mass (e.g. M100) predicted by UM.

    logms_mod_inn : ndarray
        Inner stellar mass (e.g. M10) predicted by UM.

    mask_tot : bool array
        Mask for the input mock catalog and precomputed WL pairs.

    um_wl_min_ngal : int, optional
        Minimum requred galaxies in each bin to estimate WL profile.

    """
    # The mock catalog and precomputed mass files for subsamples
    if mask is not None:
        mock_use = mock_use[mask]
        mass_encl_use = mass_encl_use[mask, :]

    if sig_logms is not None:
        return [asap_single_dsigma_weight(cfg, mock_use, mass_encl_use,
                                          obs_prof, logms_mod_tot,
                                          logms_mod_inn, sig_logms,
                                          add_stellar=cfg['um_wl_add_stellar'])
                for obs_prof in obs_data['obs_wl_dsigma']]

    return [asap_single_dsigma(cfg, mock_use, mass_encl_use, obs_prof,
                               logms_mod_tot, logms_mod_inn,
                               um_wl_min_ngal=um_wl_min_ngal,
                               verbose=verbose,
                               add_stellar=cfg['um_wl_add_stellar'])
            for obs_prof in obs_data['obs_wl_dsigma']]


def asap_predict_model_prob(param, cfg, obs_data, um_data,
                            return_all=False, show_smf=False,
                            show_dsigma=False):
    """Return all model predictions."""
    # Predict stellar masses
    (logms_mod_inn, logms_mod_tot,
     sig_logms, mask_tot) = asap_predict_mass_prob(
         param, cfg, um_data['um_mock'], return_all=return_all)

    # TODO: trying to constrain the likelihood better
    ngal_max = (cfg['obs_ngal_use'] *
                (cfg['um_volume'] / cfg['obs_volume']) * 5)
    ngal_min = (cfg['obs_ngal_use'] *
                (cfg['um_volume'] / cfg['obs_volume']) / 10)

    if len(logms_mod_tot) <= ngal_min or len(logms_mod_tot) >= ngal_max:
        um_smf_tot, um_smf_inn = None, None,
        um_dsigma = None
    else:
        # Predict SMFs
        um_smf_tot, um_smf_inn = asap_predict_smf_prob(
            logms_mod_tot, logms_mod_inn, sig_logms, cfg)

        # Predict DeltaSigma profiles
        um_dsigma = asap_predict_dsigma(
            cfg, obs_data, um_data['um_mock'], um_data['um_mass_encl'],
            logms_mod_tot, logms_mod_inn,
            mask=mask_tot, sig_logms=sig_logms,
            add_stellar=cfg['um_wl_add_stellar'])

        if show_smf and um_smf_tot is not None and um_smf_inn is not None:
            um_smf_tot_all = get_smf_bootstrap(logms_mod_tot,
                                               cfg['um_volume'],
                                               20, 10.5, 12.5,
                                               n_boots=1)
            _ = plot_mtot_minn_smf(
                obs_data['obs_smf_tot'], obs_data['obs_smf_inn'],
                obs_data['obs_mtot'], obs_data['obs_minn'],
                um_smf_tot, um_smf_inn,
                logms_mod_tot, logms_mod_inn,
                obs_smf_full=obs_data['obs_smf_full'],
                um_smf_tot_all=um_smf_tot_all,
                not_table=True)

        if show_dsigma:
            um_mhalo_tuple = asap_predict_mhalo(
                obs_data['obs_wl_dsigma'],
                um_data['um_mock'][mask_tot],
                logms_mod_tot, logms_mod_inn)
            _ = plot_dsigma_profiles(obs_data['obs_wl_dsigma'],
                                     um_dsigma, obs_mhalo=None,
                                     um_mhalo=um_mhalo_tuple)

    if return_all:
        return (um_smf_tot, um_smf_inn, um_dsigma,
                logms_mod_inn, logms_mod_tot,
                sig_logms, mask_tot)

    return um_smf_tot, um_smf_inn, um_dsigma


def asap_predict_model(param, cfg, obs_data, um_data,
                       constant_bin=False, return_all=False,
                       show_smf=False, show_dsigma=False):
    """Return all model predictions.

    Parameters
    ----------
    param: list, array, or tuple.
        Input model parameters.

    cfg : dict
        Configurations of the data and model.

    obs_data: dict
        Dictionary for observed data.

    um_data: dict
        Dictionary for UniverseMachine data.

    constant_bin : boolen
        Whether to use constant bin size for logMs_tot or not.

    return_all : bool, optional
        Return all model information.

    show_smf : bool, optional
        Show the comparison of SMF.

    show_dsigma : bool, optional
        Show the comparisons of WL.

    """
    # Predict stellar mass
    if cfg['model_type'] == 'simple' or cfg['model_type'] == 'frac1':
        (logms_mod_inn, logms_mod_tot_all,
         logms_mod_halo_all, mask_mtot) = asap_predict_mass(
             param, cfg, obs_data, um_data, constant_bin=constant_bin)
        logms_mod_tot = logms_mod_tot_all[mask_mtot]
    else:
        (logms_mod_inn_all, logms_mod_tot_all,
         logms_mod_halo_all, mask_mtot) = asap_predict_mass(
             param, cfg, obs_data, um_data, constant_bin=constant_bin)
        logms_mod_inn = logms_mod_inn_all[mask_mtot]
        logms_mod_tot = logms_mod_tot_all[mask_mtot]

    # Predict SMFs
    um_smf_tot, um_smf_inn = asap_predict_smf(
        logms_mod_tot, logms_mod_inn, cfg)

    # Predict DeltaSigma profiles
    um_dsigma_profs = asap_predict_dsigma(
        cfg, obs_data, um_data['um_mock'], um_data['um_mass_encl'],
        logms_mod_tot, logms_mod_inn, mask=mask_mtot,
        add_stellar=cfg['um_wl_add_stellar'])

    if show_smf:
        um_smf_tot_all = get_smf_bootstrap(logms_mod_tot_all,
                                           cfg['um_volume'],
                                           20, cfg['obs_min_mtot'], 12.5,
                                           n_boots=1)
        _ = plot_mtot_minn_smf(
            obs_data['obs_smf_tot'], obs_data['obs_smf_inn'],
            obs_data['obs_mtot'], obs_data['obs_minn'],
            um_smf_tot, um_smf_inn,
            logms_mod_tot, logms_mod_inn,
            obs_smf_full=obs_data['obs_smf_full'],
            um_smf_tot_all=um_smf_tot_all,
            not_table=True)

    if show_dsigma:
        um_mhalo_tuple = asap_predict_mhalo(
            obs_data['obs_wl_dsigma'], um_data['um_mock'][mask_mtot],
            logms_mod_tot, logms_mod_inn)
        _ = plot_dsigma_profiles(obs_data['obs_wl_dsigma'],
                                 um_dsigma_profs, obs_mhalo=None,
                                 um_mhalo=um_mhalo_tuple)

    if return_all:
        if cfg['model_type'] == 'simple' or cfg['model_type'] == 'frac1':
            return (um_smf_tot, um_smf_inn, um_dsigma_profs,
                    logms_mod_inn, logms_mod_tot_all,
                    logms_mod_halo_all, mask_mtot)

        return (um_smf_tot, um_smf_inn, um_dsigma_profs,
                logms_mod_inn_all, logms_mod_tot_all,
                logms_mod_halo_all, mask_mtot)

    return um_smf_tot, um_smf_inn, um_dsigma_profs
