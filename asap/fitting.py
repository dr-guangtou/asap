"""Module for model fitting."""
from __future__ import print_function, division, unicode_literals

from multiprocessing import Pool
#from contextlib import closing
#from schwimmbad import MPIPool

import emcee

from . import io
from . import config
from . import ensemble
from . import likelihood

__all__ = ['initial_model', 'fit_asap_model']


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
    cfg = config.parse_config(config_file)

    # Update the configuration file, and get the AsapParam object
    cfg, params = config.update_configuration(cfg, verbose=verbose)

    # Load observations, update the configuration
    obs_data, cfg['obs'] = io.load_obs(cfg['obs'], verbose=verbose)

    # Load UniverseMachine data
    um_data = io.load_um(cfg['um'], verbose=verbose)

    return cfg, params, obs_data, um_data


def fit_asap_model(config_file, verbose=True, debug=False):
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

    # Initialize the model, load the data
    cfg, params, obs_data, um_data = initial_model(config_file, verbose=verbose)

    ln_prob_function = likelihood.ln_probability

    with Pool() as pool:
    #with closing(Pool(processes=cfg['model']['emcee']['n_thread'])) as pool:
        sample_results, sampler = ensemble.run_emcee_sampler(
            cfg, params, likelihood.ln_probability,
            postargs=[cfg, params, obs_data, um_data],
            postkwargs={'nested': False}, pool=pool, verbose=verbose,
            debug=debug)

    return sample_results, sampler
