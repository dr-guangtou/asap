"""Likelihood functions for A.S.A.P model."""

import numpy as np
from numpy import linalg

from asap_model_prediction import asap_predict_model, asap_predict_model_prob


__all__ = ['asap_ln_prob', 'asap_ln_like', 'asap_flat_prior',
           'asap_dsigma_lnlike', 'asap_smf_lnlike']


def asap_flat_prior(param_tuple, param_low, param_upp):
    """Priors of parameters."""
    if not np.all([low <= param <= upp for param, low, upp in
                   zip(list(param_tuple), param_low, param_upp)]):
        return -np.inf

    return 0.0


def asap_flat_prior_transform(unit_cube, param_low, param_upp):
    """Transform unit cube into flat priors."""
    return unit_cube * param_upp + (1.0 - unit_cube) * param_low


def asap_ln_prob(param_tuple, cfg, obs_data, um_data):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    param_tuple: tuple of model parameters.

    """
    lp = asap_flat_prior(param_tuple, cfg['param_low'], cfg['param_upp'])

    if not np.isfinite(lp):
        return -np.inf

    return lp + asap_ln_like(param_tuple, cfg, obs_data, um_data)


def asap_dsigma_lnlike(obs_dsigma_prof, dsigma_um):
    """Calculate the likelihood for WL profile."""
    dsigma_obs = obs_dsigma_prof.sig
    dsigma_obs_err = obs_dsigma_prof.err_s

    dsigma_var = (dsigma_obs_err[:-2] ** 2)
    dsigma_dif = (dsigma_obs[:-2] - dsigma_um[:-2]) ** 2

    dsigma_chi2 = (dsigma_dif / dsigma_var).sum()
    dsigma_lnlike = -0.5 * (dsigma_chi2 +
                            np.log(2 * np.pi * dsigma_var).sum())
    # print("DSigma likelihood / chi2: %f, %f" % (dsigma_lnlike, dsigma_chi2))

    return dsigma_lnlike


def asap_smf_lnlike(obs_smf_tot, um_smf_tot, obs_smf_inn, um_smf_inn,
                    obs_smf_cov=None):
    """Calculate the likelihood for SMF."""
    smf_mtot_dif = (np.array(um_smf_tot) -
                    np.array(obs_smf_tot['smf']))
    smf_minn_dif = (np.array(um_smf_inn) -
                    np.array(obs_smf_inn['smf']))

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

    # print("SMF Tot lnlike/chi2: %f,%f" % (smf_mtot_lnlike,
    #                                       smf_mtot_chi2))
    # print("SMF Inn lnlike/chi2: %f,%f" % (smf_minn_lnlike,
    #                                       smf_minn_chi2))

    return smf_mtot_lnlike + smf_minn_lnlike


def asap_chi2(param_tuple, cfg, obs_data, um_data):
    """Chi2 function for the model."""
    return -1.0 * asap_ln_like(param_tuple, cfg, obs_data, um_data)


def asap_ln_like(param_tuple, cfg, obs_data, um_data, chi2=False,
                 sep_return=False):
    """Calculate the lnLikelihood of the model."""
    # Unpack the input parameters
    parameters = list(param_tuple)

    # Generate the model predictions
    if cfg['model_prob']:
        (um_smf_tot, um_smf_inn, um_dsigma_profs) = asap_predict_model_prob(
            parameters, cfg, obs_data, um_data)
    else:
        (um_smf_tot, um_smf_inn, um_dsigma_profs) = asap_predict_model(
            parameters, cfg, obs_data, um_data)

    if um_smf_tot is None or um_smf_inn is None or um_dsigma_profs is None:
        if sep_return:
            return -np.inf, -np.inf
        return -np.inf

    # Likelihood for SMFs.
    smf_lnlike = asap_smf_lnlike(
        obs_data['obs_smf_tot'], um_smf_tot,
        obs_data['obs_smf_inn'], um_smf_inn,
        obs_smf_cov=obs_data['obs_smf_cov'])

    if cfg['mcmc_wl_only']:
        return smf_lnlike

    # Likelihood for DeltaSigma
    dsigma_lnlike = np.array([
        asap_dsigma_lnlike(obs_dsigma_prof, um_dsigma_prof)
        for (obs_dsigma_prof, um_dsigma_prof) in
        zip(obs_data['obs_wl_dsigma'], um_dsigma_profs)]).sum()

    if not np.isfinite(dsigma_lnlike):
        return -np.inf

    if cfg['mcmc_smf_only']:
        return dsigma_lnlike

    if sep_return:
        return smf_lnlike, dsigma_lnlike

    return smf_lnlike + cfg['mcmc_wl_weight'] * dsigma_lnlike
