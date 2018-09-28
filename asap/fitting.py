"""Module for model fitting."""
from __future__ import print_function, division, unicode_literals

from . import io
from . import config

__all__ = ['initial_model']


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
