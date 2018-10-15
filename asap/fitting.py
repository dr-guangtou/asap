"""Module for model fitting."""
from __future__ import print_function, division, unicode_literals

from multiprocessing import Pool
from contextlib import closing

import numpy as np 

import emcee

from . import io
from . import config
from . import ensemble
from . import likelihood

__all__ = ['initial_model', 'fit_asap_model']


def ln_probability_global(theta):
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
    ln_prior = params.lnprior(theta, nested=False)

    if not np.isfinite(ln_prior):
        return -np.inf

    return ln_prior + likelihood.ln_likelihood(theta, cfg, obs_data, um_data)


def initial_model(config_file, verbose=True):
    """Initialize the A.S.A.P model.

    Parameters
    ----------
    config_file : str
        Input configuration file.
    verbose : bool, optional
        Blah, blah, blah. Default: True

    Return
    ------
    cfg : dict
        Updated configuration file.
    params : AsapParameter object
        ASAP model parameter object.
    obs_data : dict
        Dictionary for observations.
    um_data : dict
        Dictionary for UniverseMachine data.

    """
    # Parse the configuration file
    configs = config.parse_config(config_file)

    # Update the configuration file, and get the AsapParam object
    configs, parameters = config.update_configuration(configs, verbose=verbose)

    # Load observations, update the configuration
    observations, configs['obs'] = io.load_obs(configs['obs'], verbose=verbose)

    # Load UniverseMachine data
    umachine = io.load_um(configs['um'], verbose=verbose)

    return configs, parameters, observations, umachine


def fit_asap_model(config_file, verbose=True, use_global=False, debug=False):
    """Fit A.S.A.P model.

    Parameters
    ----------
    config_file: str
        Configuration file.
    verbose: boolen, optional
        Blah, blah, blah.  Default: True

    """
    # Make sure we are using the correct emcee version.
    assert emcee.__version__.split('.')[0] == '3', "# Wrong emcee version, should be >3.0"

    if use_global:
        global cfg, params, obs_data, um_data

        # Initialize the model, load the data
        cfg, params, obs_data, um_data = initial_model(config_file, verbose=verbose)

        with closing(Pool(processes=cfg['model']['emcee']['n_thread'])) as pool:
            sample_results, sampler = ensemble.run_emcee_sampler(
                cfg, params, ln_probability_global, 
                pool=pool, verbose=verbose, debug=debug)
    else:
        # Initialize the model, load the data
        cfg, params, obs_data, um_data = initial_model(config_file, verbose=verbose)

        ln_prob_function = likelihood.ln_probability

        with closing(Pool(processes=cfg['model']['emcee']['n_thread'])) as pool:
            sample_results, sampler = ensemble.run_emcee_sampler(
                cfg, params, ln_prob_function,
                postargs=[cfg, params, obs_data, um_data],
                postkwargs={'nested': False}, pool=pool, verbose=verbose,
                debug=debug)

    return sample_results, sampler
