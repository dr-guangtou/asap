"""Model fitting using emcee sampler."""
from __future__ import print_function, division, unicode_literals

import numpy as np

import emcee

from . import io

__all__ = ['emcee_burnin_batch', 'setup_moves', 'setup_walkers', 'run_emcee_sampler']


def setup_moves(cfg_emcee, burnin=False):
    """Choose the Move object for emcee.

    Parameters
    ----------
    cfg_emcee : dict
        Configuration parameters for emcee
    burnin : bool, optional
        Whether this is for burnin

    Return
    ------
    emcee_moves : emcee.moves object
        Move object for emcee walkers.

    """
    move_col = 'moves' if not burnin else 'moves_burnin'

    if cfg_emcee[move_col] == 'snooker':
        emcee_moves = emcee.moves.DESnookerMove()
    elif cfg_emcee[move_col] == 'stretch':
        emcee_moves = emcee.moves.StretchMove(a=cfg_emcee['stretch_a'])
    elif cfg_emcee[move_col] == 'walk':
        emcee_moves = emcee.moves.WalkMove(s=cfg_emcee['walk_s'])
    elif cfg_emcee[move_col] == 'kde':
        emcee_moves = emcee.moves.KDEMove()
    elif cfg_emcee[move_col] == 'de':
        emcee_moves = emcee.moves.DEMove(cfg_emcee['de_sigma'])
    else:
        raise Exception("Wrong option: stretch, walk, kde, de, snooker")

    return emcee_moves


def setup_walkers(cfg_emcee, params, level=0.1):
    """Initialize walkers for emcee.

    Parameters
    ----------
    cfg_emcee: dict
        Configuration parameters for emcee.
    params: asap.Parameter object
        Object for model parameters.
    level: float, optional

    Returns
    -------
    ini_positions: numpy array with (N_walker, N_param) shape
        Initial positions of all walkers.

    """
    # Initialize the walkers
    if cfg_emcee['ini_prior']:
        # Use the prior distributions for initial positions of walkers.
        return params.sample(nsamples=cfg_emcee['nwalkers_burnin'])

    return params.perturb(nsamples=cfg_emcee['nwalkers_burnin'], level=level)


def emcee_burnin_batch(sampler_burnin, ini_positions, n_step, verbose=True):
    """Run the MCMC chain."""
    # Burn-in
    if verbose:
        print("\n# Phase: Burn-in ...")

    burnin_results = sampler_burnin.run_mcmc(ini_positions, n_step, progress=True)

    #mcmc_save_results(burnin_result, mcmc_sampler,
    #                  config['mcmc_burnin_file'], config['mcmc_ndims'],
    #                  verbose=True)

    # Rest the chains
    sampler_burnin.reset()

    return burnin_results


def run_emcee_sampler(cfg, params, ln_probability, postargs=[], postkwargs={}, pool=None):
    """Fit A.S.A.P model using emcee sampler.

    Parameters
    ----------
    config_file: str
        Configuration file.
    verbose: boolen, optional
        Blah, blah, blah.  Default: True

    """
    # Initialize the walkers
    ini_positions = setup_walkers(cfg['model']['emcee'], params, level=0.1)

    # Number of parameters and walkers for burn-in process
    n_dim, n_walkers_burnin, n_step_burnin = params.n_param, ini_positions.shape[0], ini_positions.shape[1]

    # Setup the `Move` for burn-in walkers
    burnin_move = setup_moves(cfg['model']['emcee'], 'moves_burnin')

    # Initialize sampler
    sampler_burnin = emcee.EnsembleSampler(
        n_walkers_burnin, n_dim, ln_probability, args=postargs, kwargs=postkwargs,
        pool=pool, moves=burnin_move)

    # Burn in sampler
    mcmc_burnin_pos, mcmc_burnin_lnp, mcmc_burnin_state = emcee_burnin_batch(
                sampler_burnin, ini_positions, n_step_burnin, verbose=True)

    return mcmc_burnin_pos, mcmc_burnin_lnp, mcmc_burnin_state