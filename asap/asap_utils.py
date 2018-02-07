"""Utilities for A.S.A.P model."""

from __future__ import print_function, division, unicode_literals

import pickle
import emcee

from scipy.stats import mvn, norm

import numpy as np


__all__ = ["mcmc_save_pickle", "mcmc_save_results", "mcmc_load_pickle",
           "mcmc_initial_guess", "mcmc_samples_stats", "mcmc_load_results",
           "mcmc_setup_moves", "mass_gaussian_weight_2d",
           "mass_gaussian_weight"]


def mcmc_save_pickle(mcmc_pickle_file, mcmc_results):
    """Pickle the chain to a file."""
    pickle_file = open(mcmc_pickle_file, 'wb')
    pickle.dump(mcmc_results, pickle_file)
    pickle_file.close()

    return None


def mcmc_load_pickle(mcmc_pickle_file):
    """Load the pickled pickle."""
    pickle_file = open(mcmc_pickle_file, 'rb')
    mcmc_pickle = pickle.load(pickle_file)
    pickle_file.close()

    return mcmc_pickle


def mcmc_initial_guess(param_ini, param_sig, n_walkers, n_dims):
    """Initialize guesses for the MCMC run."""
    mcmc_position = np.zeros([n_walkers, n_dims])

    for ii, param_0 in enumerate(param_ini):
        mcmc_position[:, ii] = (
            param_0 + param_sig[ii] * np.random.randn(n_walkers))

    return mcmc_position


def mcmc_samples_stats(mcmc_samples):
    """1D marginalized parameter constraints."""
    return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
               zip(*np.percentile(mcmc_samples,
                                  [16, 50, 84], axis=0)))


def mcmc_load_results(mcmc_file):
    """Retrieve the MCMC results from .npz file."""
    mcmc_data = np.load(mcmc_file)

    return (mcmc_data['samples'], mcmc_data['chains'],
            mcmc_data['lnprob'], mcmc_data['best'],
            mcmc_data['position'], mcmc_data['acceptance'])


def mcmc_save_results(mcmc_results, mcmc_sampler, mcmc_file,
                      mcmc_ndims, verbose=True):
    """Save the MCMC run results."""
    (mcmc_position, mcmc_lnprob, _) = mcmc_results

    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
        (-1, mcmc_ndims))
    mcmc_chains = mcmc_sampler.chain
    mcmc_lnprob = mcmc_sampler.lnprobability.reshape(-1, 1)
    mcmc_best = mcmc_samples[np.argmax(mcmc_lnprob)]
    mcmc_params_stats = mcmc_samples_stats(mcmc_samples)

    np.savez(mcmc_file,
             samples=mcmc_samples, lnprob=np.array(mcmc_lnprob),
             best=np.array(mcmc_best), chains=mcmc_chains,
             position=np.asarray(mcmc_position),
             acceptance=np.array(mcmc_sampler.acceptance_fraction))

    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(mcmc_sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" % np.max(mcmc_lnprob))
        print(mcmc_best)
        print("#------------------------------------------------------")
        for param_stats in mcmc_params_stats:
            print(param_stats)
        print("#------------------------------------------------------")

    return


def mcmc_setup_moves(cfg):
    """Choose the Move object for emcee."""
    if cfg['mcmc_moves'] == 'redblue':
        emcee_moves = emcee.moves.RedBlueMove(randomize_split=False)
    elif cfg['mcmc_moves'] == 'stretch':
        emcee_moves = emcee.moves.StretchMove(a=cfg['mcmc_stretch_a'])
    elif cfg['mcmc_moves'] == 'walk':
        emcee_moves = emcee.moves.WalkMove(s=cfg['mcmc_walk_s'])
    elif cfg['mcmc_moves'] == 'kde':
        emcee_moves = emcee.moves.KDEMove()
    else:
        raise Exception("Wrong option: redblue, stretch, walk, kde")

    return emcee_moves


def mass_gaussian_weight_2d(logms1, logms2, sigms1, sigms2,
                            bin1_l, bin1_r, bin2_l, bin2_r):
    """Weight of galaaxy using two stellar masses."""
    low = np.array([bin1_l, bin2_l])
    upp = np.array([bin1_r, bin2_r])
    mu = np.array([logms1, logms2])
    cov = np.array([[sigms1 ** 2, sigms1 * sigms2],
                    [sigms2 * sigms1, sigms2 ** 2]])

    p, _ = mvn.mvnun(low, upp, mu, cov)

    return p


def mtot_minn_weight(logm_tot, logm_inn, sig,
                     mtot_0, mtot_1, minn_0, minn_1):
    """Two-dimensional weight of galaxy in Mtot-Minn box."""
    return np.array([mass_gaussian_weight_2d(
        m1, m2, ss, ss, mtot_0, mtot_1, minn_0, minn_1)
                     for m1, m2, ss in zip(logm_tot, logm_inn, sig)])


def mass_gaussian_weight(logms, sigms, left, right):
    """Weights of stellar in bin."""
    return (norm.sf(left, loc=logms, scale=sigms) -
            norm.sf(right, loc=logms, scale=sigms))
