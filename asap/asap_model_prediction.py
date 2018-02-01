"""Predictions of the A.S.A.P. model."""

import numpy as np

from scipy import interpolate

from halotools.mock_observables import delta_sigma_from_precomputed_pairs

from stellar_mass_function import get_smf_bootstrap
from full_mass_profile_model import mass_prof_model_simple, \
    mass_prof_model_frac1, mass_prof_model_frac2, \
    mass_prof_model_frac3, mass_prof_model_frac4, \
    mass_prof_model_frac5
from um_model_plot import plot_mtot_minn_smf, plot_dsigma_profiles


__all__ = ['asap_predict_mass', 'asap_predict_smf',
           'asap_single_dsigma', 'asap_um_dsigma',
           'asap_predict_dsigma', 'asap_predict_model']


def asap_predict_mass(parameters, cfg, obs_data, um_data,
                      constant_bin=False):
    """M100, M10, Mtot using Mvir, M_gal, M_ICL.

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
    else:
        raise Exception("# Wrong model choice! ")


def asap_predict_smf(logms_mod_tot, logms_mod_inn, cfg):
    """Stellar mass functions of Minn and Mtot predicted by UM."""
    # SMF of the predicted Mtot (M1100)
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

    return um_smf_tot, um_smf_inn


def asap_um_dsigma(cfg, mock_use, mass_encl_use, mask,
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
    um_cosmo = cfg['um_cosmo']

    # Radius bins
    rp_bins = np.logspace(np.log10(cfg['um_wl_minr']),
                          np.log10(cfg['um_wl_maxr']),
                          cfg['um_wl_nbin'])

    if verbose:
        print("# Deal with %d galaxies in the subsample" % np.sum(mask))
    #  Use the mask to get subsample positions and pre-computed pairs
    subsample = mock_use[mask]
    subsample_positions = np.vstack([subsample['x'],
                                     subsample['y'],
                                     subsample['z']]).T
    subsample_mass_encl_precompute = mass_encl_use[mask, :]

    rp_ht_units, ds_ht_units = delta_sigma_from_precomputed_pairs(
        subsample_positions, subsample_mass_encl_precompute,
        rp_bins, cfg['um_lbox'], cosmology=um_cosmo)

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
        dsigma = intrp(r_interp)

        return (r_interp, dsigma)

    return (rp_phys, ds_phys_msun_pc2)


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
        _, wl_prof = asap_um_dsigma(
            cfg, mock_use, mass_encl_use, bin_mask,
            r_interp=obs_prof.r, mstar_lin=mstar_lin)

    return wl_prof


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


def asap_predict_dsigma(cfg, obs_data, um_data,
                        logms_mod_tot, logms_mod_inn, mask_mtot,
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
    mock_use = um_data['um_mock'][mask_mtot]
    mass_encl_use = um_data['um_mass_encl'][mask_mtot, :]

    return [asap_single_dsigma(cfg, mock_use, mass_encl_use, obs_prof,
                               logms_mod_tot, logms_mod_inn,
                               um_wl_min_ngal=um_wl_min_ngal,
                               verbose=verbose,
                               add_stellar=cfg['um_wl_add_stellar'])
            for obs_prof in obs_data['obs_wl_dsigma']]


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
    if (cfg['model_type'] == 'simple' or cfg['model_type'] == 'frac1'):
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

    # Predict the SMFs
    um_smf_tot, um_smf_inn = asap_predict_smf(
        logms_mod_tot, logms_mod_inn, cfg)

    um_dsigma_profs = asap_predict_dsigma(
        cfg, obs_data, um_data,
        logms_mod_tot, logms_mod_inn, mask_mtot,
        add_stellar=cfg['um_wl_add_stellar'])

    if show_smf:
        um_smf_tot_all = get_smf_bootstrap(logms_mod_tot_all,
                                           cfg['um_volume'],
                                           20, 10.5, 12.5,
                                           n_boots=1)
        fig_smf = plot_mtot_minn_smf(
            obs_data['obs_smf_tot'], obs_data['obs_smf_inn'],
            obs_data['obs_mtot'], obs_data['obs_minn'],
            um_smf_tot, um_smf_inn,
            logms_mod_tot, logms_mod_inn,
            obs_smf_full=obs_data['obs_smf_full'],
            um_smf_tot_all=um_smf_tot_all)

    if show_dsigma:
        um_mhalo_tuple = asap_predict_mhalo(
            obs_data['obs_wl_dsigma'], um_data['um_mock'][mask_mtot],
            logms_mod_tot, logms_mod_inn)
        fig_dsigma = plot_dsigma_profiles(obs_data['obs_wl_dsigma'],
                                          um_dsigma_profs, obs_mhalo=None,
                                          um_mhalo=um_mhalo_tuple)

    if return_all:
        if (cfg['model_type'] == 'simple' or cfg['model_type'] == 'frac1'):
            return (um_smf_tot, um_smf_inn, um_dsigma_profs,
                    logms_mod_inn, logms_mod_tot_all,
                    logms_mod_halo_all, mask_mtot)
        else:
            return (um_smf_tot, um_smf_inn, um_dsigma_profs,
                    logms_mod_inn_all, logms_mod_tot_all,
                    logms_mod_halo_all, mask_mtot)

    return um_smf_tot, um_smf_inn, um_dsigma_profs
