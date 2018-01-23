"""Likelihood functions for A.S.A.P model."""

import numpy as np

__all__ = ['asap_ln_prof', 'asap_ln_like', 'asap_flat_prior']


def asap_flat_prior(param_tuple, param_low, param_upp):
    """Priors of parameters."""
    if not np.all(
        [low <= param <= upp for param, low, upp in
         zip(list(param_tuple), param_low, param_upp)]):
        return -np.inf

    return 0.0


def asap_ln_prob(param_tuple):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    param_tuple: tuple of model parameters.

    """
    lp = asap_flat_prior(param_tuple)

    if not np.isfinite(lp):
        return -np.inf

    return lp + asap_ln_like(param_tuple)


def asap_ln_like(param_tuple):
    """Calculate the lnLikelihood of the model."""
