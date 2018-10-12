"""Model fitting using emcee sampler."""
from __future__ import print_function, division, unicode_literals

import numpy as np

import emcee

from scipy.stats import gaussian_kde

from . import io

__all__ = ['emcee_burnin_batch', 'setup_moves', 'setup_walkers', 'run_emcee_sampler',
           'mcmc_samples_stats']


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
    move_col = 'sample_move' if not burnin else 'burnin_move'

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
        return params.sample(nsamples=cfg_emcee['burnin_n_walker'])

    return params.perturb(nsamples=cfg_emcee['burnin_n_walker'], level=level)


def mcmc_samples_stats(mcmc_samples):
    """1D marginalized parameter constraints."""
    return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
               zip(*np.percentile(mcmc_samples,
                                  [16, 50, 84], axis=0)))


def emcee_burnin_batch(sampler_burnin, ini_positions, n_step, verbose=True):
    """Run the burn-in process in batch mode."""
    if verbose:
        print("\n# Phase: Burn-in ...")

    burnin_results = sampler_burnin.run_mcmc(
        ini_positions, n_step, progress=True, store=True)

    return burnin_results


def emcee_sample_batch(sampler, ini_positions, n_step, burnin_state=None, verbose=True):
    """Run the sampling process in batch mode."""
    if verbose:
        print("\n# Phase: Sampling Run ...")

    sample_results = sampler.run_mcmc(
        ini_positions, n_step, rstate0=burnin_state, store=True, progress=True)

    return sample_results


def run_emcee_sampler(cfg, params, ln_probability, postargs=[], postkwargs={}, pool=None,
                      verbose=True, debug=False):
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
    n_dim = params.n_param
    n_walkers_burnin, n_step_burnin = ini_positions.shape[0], ini_positions.shape[1]

    # Setup the `Move` for burn-in walkers
    burnin_move = setup_moves(cfg['model']['emcee'], 'burnin_move')

    # Initialize sampler
    sampler_burnin = emcee.EnsembleSampler(
        n_walkers_burnin, n_dim, ln_probability, args=postargs, kwargs=postkwargs,
        pool=pool, moves=burnin_move)

    # Burn-in process
    if cfg['model']['emcee']['burnin_n_repeat'] > 1:
        # TODO: Run burn-in a few times
        pass

    burnin_results = emcee_burnin_batch(
        sampler_burnin, ini_positions, n_step_burnin, verbose=verbose)

    if debug:
        return burnin_results, sampler_burnin

    # Rest the chains
    sampler_burnin.reset()

    burnin_pos, _, burnin_state = burnin_results

    # Save the burn-in results
    io.save_results_to_npz(burnin_results, sampler_burnin, 'burnin.npz', n_dim,
                           verbose=verbose)

    # Number of walkers and steps for the final sampling run
    n_walkers_sample = cfg['model']['emcee']['sample_n_walker']
    n_step_sample = cfg['model']['emcee']['sample_n_sample']

    # Estimate the Kernel density distributions of final brun-in positions
    # Resample the distributions to get starting positions of the actual run
    burnin_kde = gaussian_kde(np.transpose(burnin_pos), bw_method='silverman')
    new_ini_positions = np.transpose(burnin_kde.resample(n_walkers_sample))

    # Decide the Ensemble moves for walkers during the official run
    sample_move = setup_moves(cfg['model']['emcee'], 'sample_move')

    # Initialize sampler
    sampler = emcee.EnsembleSampler(
        n_walkers_sample, n_dim, ln_probability, args=postargs, kwargs=postkwargs,
        pool=pool, moves=sample_move)

    # Final sampling run
    sample_results = emcee_sample_batch(
        sampler, new_ini_positions, n_step_sample, burnin_state=burnin_state, verbose=True)

    # Save the sampling results
    io.save_results_to_npz(sample_results, sampler, 'sample.npz', n_dim,
                           verbose=verbose)


    return sample_results, sampler
