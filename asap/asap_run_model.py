#!/usr/bin/env python2
"""Model using the in-situ and ex-situ mass."""

import os
import pickle
import argparse

import emcee

import numpy as np

from astropy.table import Table

from um_model_plot import plot_mtot_minn_smf, plot_dsigma_profiles
from asap_data_io import parse_config, load_observed_data, \
    config_observed_data, config_um_data, load_um_data
from asap_utils import mcmc_save_chains, mcmc_save_results, \
    mcmc_initial_guess, mcmc_samples_stats
from asap_model_setup import setup_model
from asap_likelihood import asap_ln_prob
# from convergence import convergence_check


def initial_model(cfg, verbose=True):
    """Initialize the A.S.A.P model."""
    # Configuration for HSC data
    cfg = config_observed_data(cfg, verbose=verbose)
    obs_data, cfg = load_observed_data(cfg, verbose=verbose)

    # Configuration for UniverseMachine data.
    cfg = config_um_data(cfg, verbose=verbose)
    um_data = load_um_data(cfg, verbose=verbose)

    cfg = setup_model(cfg, verbose=verbose)

    return cfg, obs_data, um_data


def run_asap_model(args, verbose=True):
    """Run A.S.A.P model."""
    # Parse the configuration file.
    cfg = parse_config(args.config)

    # Load the data
    cfg, obs_data, um_data = initial_model(cfg, verbose=verbose)

    # Initialize the model
    mcmc_position = mcmc_initial_guess(
        cfg['param_ini'], cfg['param_sig'], cfg['mcmc_nwalkers'],
        cfg['mcmc_ndims'])

    if cfg['mcmc_nthreads'] > 1:
        from multiprocessing import Pool
        from contextlib import closing

        with closing(Pool(processes=cfg['mcmc_nthreads'])) as pool:
            mcmc_sampler = emcee.EnsembleSampler(
                cfg['mcmc_nwalkers'],
                cfg['mcmc_ndims'],
                asap_ln_prob,
                move=emcee.moves.StretchMove(a=3),
                pool=pool)
    else:
        mcmc_sampler = emcee.EnsembleSampler(
            cfg['mcmc_nwalkers'],
            cfg['mcmc_ndims'],
            asap_ln_prob)

    # Burn-in
    if verbose:
        print("# Phase: Burn-in ...")
    mcmc_burnin_result = mcmc_sampler.run_mcmc(
         mcmc_position, cfg['mcmc_nburnin'],
         progress=True)

    mcmc_burnin_position, _, mcmc_burnin_state = mcmc_burnin_result

    #  Pickle the results
    mcmc_save_results(cfg['mcmc_burnin_file'], mcmc_burnin_result)
    mcmc_burnin_chain = mcmc_sampler.chain
    mcmc_save_chains(cfg['mcmc_burnin_chain_file'], mcmc_burnin_chain)

    # Rest the chains
    mcmc_sampler.reset()

    # conv_crit = 3

    # MCMC run
    if verbose:
        print("# Phase: MCMC run ...")
    mcmc_run_result = mcmc_sampler.run_mcmc(
        mcmc_burnin_position, cfg['mcmc_nsamples'],
        rstate0=mcmc_burnin_state,
        progress=True)

    #  Pickle the result
    mcmc_save_results(cfg['mcmc_run_file'], mcmc_run_result)
    mcmc_run_chain = mcmc_sampler.chain
    mcmc_save_chains(cfg['mcmc_run_chain_file'], mcmc_run_chain)

    if verbose:
        print("# Get MCMC samples and best-fit parameters ...")
    # Get the MCMC samples
    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
        (-1, cfg['mcmc_ndims']))
    #  Save the samples
    mcmc_lnprob = mcmc_sampler.lnprobability.reshape(-1, 1)
    mcmc_best = mcmc_samples[np.argmax(mcmc_lnprob)]
    np.savez(cfg['mcmc_run_samples_file'],
             samples=mcmc_samples, lnprob=mcmc_lnprob,
             best=mcmc_best, acceptance=mcmc_sampler.acceptance_fraction)

    # Get the best-fit parameters and the 1-sigma error
    mcmc_params_stats = mcmc_samples_stats(mcmc_samples)
    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(mcmc_sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" %
              np.nanmax(mcmc_lnprob))
        print(mcmc_best)
        print("#------------------------------------------------------")
        for param_stats in mcmc_params_stats:
            print(param_stats)
        print("#------------------------------------------------------")

    return mcmc_best, mcmc_params_stats, mcmc_samples


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config',
        help="Configuration file",
        default='asap_default_config.yaml')

    args = parser.parse_args()

    run_asap_model(args)
