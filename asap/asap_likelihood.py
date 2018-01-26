"""Likelihood functions for A.S.A.P model."""

import numpy as np
from numpy import linalg

from asap_model_prediction import asap_predict_model


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


def asap_ln_prob(param_tuple, cfg, obs_data, um_data, chi2=False):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    param_tuple: tuple of model parameters.

    """
    lp = asap_flat_prior(param_tuple, cfg['param_low'], cfg['param_upp'])

    if not np.isfinite(lp):
        return -np.inf

    return lp + asap_ln_like(param_tuple, cfg, obs_data, um_data, chi2=chi2)


def asap_dsigma_lnlike(obs_dsigma_prof, dsigma_um, chi2=False):
    """Calculate the likelihood for WL profile."""
    dsigma_obs = obs_dsigma_prof.sig
    dsigma_obs_err = obs_dsigma_prof.err_s

    dsigma_var = (dsigma_obs_err[:-2] ** 2)
    dsigma_dif = (dsigma_obs[:-2] - dsigma_um[:-2]) ** 2

    dsigma_chi2 = (dsigma_dif / dsigma_var).sum()
    dsigma_lnlike = -0.5 * (dsigma_chi2 +
                            np.log(2 * np.pi * dsigma_var).sum())
    # print("DSigma likelihood / chi2: %f, %f" % (dsigma_lnlike, dsigma_chi2))

    if chi2:
        return dsigma_chi2

    return dsigma_lnlike


def asap_smf_lnlike(obs_smf_tot, um_smf_tot, obs_smf_inn, um_smf_inn,
                    obs_smf_cov=None, chi2=False):
    """Calculate the likelihood for SMF."""
    smf_mtot_dif = (obs_smf_tot['smf'] - um_smf_tot['smf']) ** 2
    smf_minn_dif = (obs_smf_inn['smf'] - um_smf_inn['smf']) ** 2

    if obs_smf_cov is not None:
        smf_cov_inv = linalg.inv(obs_smf_cov)
        smf_cov_dim = len(obs_smf_cov)
        lnlike_norm = -0.5 * ((np.log(2.0 * np.pi) * smf_cov_dim) +
                              np.log(linalg.det(obs_smf_cov)))
        smf_dif = np.array([smf_mtot_dif, smf_minn_dif]).flatten()

        if chi2:
            return np.dot(smf_dif, np.dot(smf_cov_inv, smf_dif))

        return (-0.5 * np.dot(smf_dif, np.dot(smf_cov_inv, smf_dif)) +
                lnlike_norm)
    else:
        smf_mtot_var = obs_smf_tot['smf_err'] ** 2
        smf_minn_var = obs_smf_inn['smf_err'] ** 2

        smf_mtot_chi2 = (smf_mtot_dif / smf_mtot_var).sum()
        smf_minn_chi2 = (smf_minn_dif / smf_minn_var).sum()

        smf_mtot_lnlike = -0.5 * (
            smf_mtot_chi2 + np.log(2 * np.pi * smf_mtot_var).sum())
        smf_minn_lnlike = -0.5 * (
            smf_minn_chi2 + np.log(2 * np.pi * smf_minn_var).sum())

        # print("SMF Tot lnlike/chi2: %f,%f" % (smf_mtot_lnlike,
        #                                       smf_mtot_chi2))
        # print("SMF Inn lnlike/chi2: %f,%f" % (smf_minn_lnlike,
        #                                       smf_minn_chi2))

        if chi2:
            return smf_mtot_chi2 + smf_minn_chi2

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
    (um_smf_tot, um_smf_inn, um_dsigma_profs) = asap_predict_model(
        parameters, cfg, obs_data, um_data)

    # Check SMF
    msg = '# UM and observed SMFs should have the same size!'
    assert len(um_smf_inn) == len(obs_data['obs_smf_inn']), msg
    assert len(um_smf_tot) == len(obs_data['obs_smf_tot']), msg

    if not cfg['mcmc_wl_only']:
        smf_lnlike = asap_smf_lnlike(
            obs_data['obs_smf_tot'], um_smf_tot,
            obs_data['obs_smf_inn'], um_smf_inn,
            obs_smf_cov=obs_data['obs_smf_cov'],
            chi2=chi2)
    else:
        smf_lnlike = 0.0

    # Check WL profiles
    msg = '# UM and observed WL profiles should have the same size!'
    assert len(um_dsigma_profs) == len(obs_data['obs_wl_dsigma'])
    assert len(um_dsigma_profs[0]) == len(obs_data['obs_wl_dsigma'][0].r)

    if not cfg['mcmc_smf_only']:
        dsigma_lnlike = np.nansum([
            asap_dsigma_lnlike(obs_dsigma_prof, um_dsigma_prof,
                               chi2=chi2)
            for (obs_dsigma_prof, um_dsigma_prof) in
            zip(obs_data['obs_wl_dsigma'], um_dsigma_profs)])
    else:
        dsigma_lnlike = 0.0

    if sep_return:
        return smf_lnlike, dsigma_lnlike

    return smf_lnlike + cfg['mcmc_wl_weight'] * dsigma_lnlike
