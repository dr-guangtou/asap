"""Module for likelihood functions for A.S.A.P model."""

import numpy as np
from numpy import linalg

from . import predictions

__all__ = ['ln_likelihood', 'ln_probability', 'lnlike_dsigma', 'lnlike_smf']


def ln_probability(theta, cfg, params, obs_data, um_data, nested=False):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    theta: tuple or list
        One set of model parameters.
    cfg: dict
        Configuration parameters.
    params: asap.Parameters object
        Object for model parameters.
    obs_data: dict
        Dictionary of observations.
    um_data: dict
        Dictionary of UniverseMachine data.
    sep_return: boolen, optional
        Return the likelihood for SMF and DeltaSigma profiles separately when True.
        Default: False

    Returns
    -------
        The ln(likelihood) of the model given the input parameters.

    """
    ln_prior = params.lnprior(theta, nested=nested)

    if not np.isfinite(ln_prior):
        return -np.inf

    return ln_prior + ln_likelihood(theta, cfg, obs_data, um_data)


def lnlike_dsigma(dsigma_obs, dsigma_um, mask=None):
    """Calculate the likelihood for WL profile."""
    if mask is None:
        dsigma_var = (dsigma_obs['dsigma_err'] ** 2)
        dsigma_dif = (dsigma_obs['dsigma'] - dsigma_um) ** 2
    else:
        dsigma_var = (dsigma_obs['dsigma_err'][mask] ** 2)
        dsigma_dif = (dsigma_obs['dsigma'][mask] - dsigma_um[mask]) ** 2

    dsigma_chi2 = (dsigma_dif / dsigma_var).sum()
    dsigma_lnlike = -0.5 * (dsigma_chi2 + np.log(2 * np.pi * dsigma_var).sum())

    return dsigma_lnlike


def lnlike_smf(obs_smf_tot, um_smf_tot, obs_smf_inn, um_smf_inn, obs_smf_cov=None):
    """Calculate the likelihood for SMF."""
    smf_mtot_dif = (np.array(um_smf_tot) - np.array(obs_smf_tot['smf']))
    smf_minn_dif = (np.array(um_smf_inn) - np.array(obs_smf_inn['smf']))

    if obs_smf_cov is not None:
        smf_cov_inv = linalg.inv(obs_smf_cov)
        lnlike_norm = -0.5 * ((np.log(2.0 * np.pi) * len(obs_smf_cov)) +
                              np.log(linalg.det(obs_smf_cov)))
        smf_dif = np.concatenate([smf_mtot_dif, smf_minn_dif])

        smf_chi2 = np.dot(smf_dif, np.dot(smf_cov_inv, smf_dif))

        return -0.5 * smf_chi2 + lnlike_norm

    smf_mtot_var = np.array(obs_smf_tot['smf_err'] ** 2)
    smf_minn_var = np.array(obs_smf_inn['smf_err'] ** 2)

    smf_mtot_chi2 = (smf_mtot_dif ** 2 / smf_mtot_var).sum()
    smf_minn_chi2 = (smf_minn_dif ** 2 / smf_minn_var).sum()

    smf_mtot_lnlike = -0.5 * (
        smf_mtot_chi2 + np.log(2 * np.pi * smf_mtot_var).sum())
    smf_minn_lnlike = -0.5 * (
        smf_minn_chi2 + np.log(2 * np.pi * smf_minn_var).sum())

    return smf_mtot_lnlike + smf_minn_lnlike


def model_chi2(theta, cfg, obs_data, um_data):
    """Chi2 function for the model."""
    return -1.0 * ln_likelihood(theta, cfg, obs_data, um_data)


def ln_likelihood(theta, cfg, obs_data, um_data, sep_return=False):
    """Calculate the lnLikelihood of the model.

    Parameters
    ----------
    theta: tuple or list
        One set of model parameters.
    cfg: dict
        Configuration parameters.
    obs_data: dict
        Dictionary of observations.
    um_data: dict
        Dictionary of UniverseMachine data.
    sep_return: boolen, optional
        Return the likelihood for SMF and DeltaSigma profiles separately when True.
        Default: False

    Returns
    -------
        The ln(likelihood) of the model given the input parameters.

    """
    # Unpack the input parameters
    parameters = list(theta)

    um_smf_tot, um_smf_inn, um_dsigma = predictions.make_model_predictions(
        parameters, cfg, obs_data, um_data)

    if um_smf_tot is None or um_smf_inn is None or um_dsigma is None:
        if sep_return:
            return -np.inf, -np.inf
        return -np.inf

    # Likelihood for SMFs.
    smf_lnlike = lnlike_smf(obs_data['smf_tot'], um_smf_tot, obs_data['smf_inn'], um_smf_inn,
                            obs_smf_cov=obs_data['smf_cov'])

    if cfg['model']['smf_only']:
        return smf_lnlike

    # Likelihood for DeltaSigma
    r_mpc = obs_data['wl_dsigma'][0]['r_mpc']
    r_mask = ((r_mpc >= cfg['model']['dsigma_minr']) & (r_mpc < cfg['model']['dsigma_maxr']))
    dsigma_lnlike = np.array([
        lnlike_dsigma(dsigma_obs, dsigma_um, mask=r_mask) for (dsigma_obs, dsigma_um) in
        zip(obs_data['wl_dsigma'], um_dsigma)]).sum()

    if not np.isfinite(dsigma_lnlike):
        return -np.inf

    if cfg['model']['wl_only']:
        return dsigma_lnlike

    if sep_return:
        return smf_lnlike, dsigma_lnlike

    return smf_lnlike + cfg['model']['wl_weight'] * dsigma_lnlike
