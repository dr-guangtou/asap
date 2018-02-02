#!/usr/bin/env python2
"""Model using the in-situ and ex-situ mass."""

from __future__ import division, print_function, unicode_literals

import sys
import time
import argparse

try:
    import emcee
    use_emcee = True
except ImportError:
    use_emcee = False

try:
    import dynesty
    use_dynesty = True
except ImportError:
    use_dynesty = False

import numpy as np

from asap_data_io import parse_config, load_observed_data, \
    config_observed_data, config_um_data, load_um_data
from asap_utils import mcmc_save_results, mcmc_initial_guess, \
    mcmc_save_pickle, mcmc_setup_moves
from asap_model_setup import setup_model
from asap_likelihood import asap_flat_prior, asap_ln_like, \
    asap_smf_lnlike, asap_dsigma_lnlike, asap_flat_prior_transform
from asap_model_prediction import asap_predict_model
# from convergence import convergence_check

__all__ = ['initial_model', 'asap_ln_prob_global', 'asap_ln_like_global',
           'asap_emcee_burnin', 'asap_emcee_run', 'asap_emcee_fit',
           'asap_dynesty_run', 'asap_dynesty_fit']


def initial_model(config, verbose=True):
    """Initialize the A.S.A.P model."""
    # Configuration for HSC data
    config_obs = config_observed_data(config, verbose=verbose)
    obs_data_use, config_obs = load_observed_data(config_obs, verbose=verbose)

    # Configuration for UniverseMachine data.
    config_obs_um = config_um_data(config_obs, verbose=verbose)
    um_data_use = load_um_data(config_obs_um, verbose=verbose)

    config_all = setup_model(config_obs_um, verbose=verbose)

    return config_all, obs_data_use, um_data_use


def asap_ln_prob_global(param_tuple, chi2=False):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    param_tuple: tuple of model parameters.

    """
    lp = asap_flat_prior(param_tuple, cfg['param_low'], cfg['param_upp'])

    if not np.isfinite(lp):
        return -np.inf

    return lp + asap_ln_like(param_tuple, cfg, obs_data, um_data, chi2=chi2)


def asap_ln_like_global(param_tuple):
    """Calculate the lnLikelihood of the model.

    Using the global properties, mainly for using the nested and
    dynesty sampling method.
    """
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
            obs_smf_inn=obs_data['obs_smf_inn'],
            um_smf_inn=um_smf_inn, chi2=False)
    else:
        smf_lnlike = 0.0

    # Check WL profiles
    msg = '# UM and observed WL profiles should have the same size!'
    assert len(um_dsigma_profs) == len(obs_data['obs_wl_dsigma'])
    assert len(um_dsigma_profs[0]) == len(obs_data['obs_wl_dsigma'][0].r)

    if not cfg['mcmc_smf_only']:
        dsigma_lnlike = np.nansum([
            asap_dsigma_lnlike(obs_dsigma_prof, um_dsigma_prof,
                               chi2=False)
            for (obs_dsigma_prof, um_dsigma_prof) in
            zip(obs_data['obs_wl_dsigma'], um_dsigma_profs)])
    else:
        dsigma_lnlike = 0.0

    return smf_lnlike + cfg['mcmc_wl_weight'] * dsigma_lnlike


def asap_emcee_burnin(mcmc_sampler, mcmc_position, config, verbose=True):
    """Run the MCMC chain."""
    # Burn-in
    if verbose:
        print("# Phase: Burn-in ...")
    mcmc_burnin_result = mcmc_sampler.run_mcmc(
        mcmc_position, config['mcmc_nburnin'],
        progress=True)

    mcmc_save_results(mcmc_burnin_result, mcmc_sampler,
                      config['mcmc_burnin_file'], config['mcmc_ndims'],
                      verbose=True)

    # Rest the chains
    mcmc_sampler.reset()

    return mcmc_burnin_result


def asap_emcee_run(mcmc_sampler, mcmc_burnin_result, config, verbose=True):
    """Run the MCMC chain."""
    mcmc_burnin_position, _, mcmc_burnin_state = mcmc_burnin_result

    if verbose:
        print("# Phase: MCMC run ...")
    mcmc_run_result = mcmc_sampler.run_mcmc(
        mcmc_burnin_position, config['mcmc_nsamples'],
        rstate0=mcmc_burnin_state,
        progress=True)

    mcmc_save_results(mcmc_run_result, mcmc_sampler,
                      cfg['mcmc_run_file'], cfg['mcmc_ndims'],
                      verbose=True)

    return mcmc_run_result


def asap_dynesty_run(dsampler, config, verbose=True):
    """Run Dynesty sampler for A.S.A.P model."""
    # generator for initial nested sampling
    ncall = dsampler.ncall
    niter = dsampler.it - 1
    tstart = time.time()
    for results in dsampler.sample_initial(nlive=cfg['dynesty_nlive_ini'],
                                           dlogz=cfg['dynesty_dlogz_ini'],
                                           maxcall=cfg['dynesty_maxcall_ini'],
                                           maxiter=cfg['dynesty_maxiter_ini']):

        (worst, ustar, vstar, loglstar, logvol,
         logwt, logz, logzvar, h, nc, worst_it,
         propidx, propiter, eff, delta_logz) = results

    ndur = time.time() - tstart
    if verbose:
        print('\ndone dynesty (initial) in {0}s'.format(ndur))


def asap_dynesty_fit(args, verbose=True):
    """Run A.S.A.P model using dynesty."""
    assert use_dynesty, "# Can not import dynesty !"

    global cfg, obs_data, um_data
    # Parse the configuration file  .
    config_initial = parse_config(args.config)

    # Load the data
    cfg, obs_data, um_data = initial_model(config_initial, verbose=verbose)

    if cfg['mcmc_nthreads'] > 1:
        from multiprocessing import Pool
        from contextlib import closing

        with closing(Pool(processes=cfg['mcmc_nthreads'])) as pool:
            if args.sampler == 'nested':
                dsampler = dynesty.NestedSampler(
                    asap_ln_like_global,
                    asap_flat_prior_transform,
                    cfg['mcmc_ndims'],
                    ptform_args=[cfg['param_low'], cfg['param_upp']],
                    bound=cfg['dynesty_bound'],
                    sample=cfg['dynesty_sample'],
                    nlive=cfg['dynesty_nlive_ini'],
                    bootstrap=cfg['dynesty_bootstrap'],
                    enlarge=cfg['dynesty_enlarge'],
                    walks=cfg['dynesty_walks'],
                    update_interval=cfg['dynesty_update_interval'],
                    pool=pool)

                dsampler.run_nested(
                    dlogz=cfg['dynesty_dlogz_run'],
                    maxiter=cfg['dynesty_maxiter_run'],
                    maxcall=cfg['dynesty_maxcall_run'])
            else:
                # logl_args=[cfg, obs_data, um_data],
                dsampler = dynesty.DynamicNestedSampler(
                    asap_ln_like_global,
                    asap_flat_prior_transform,
                    cfg['mcmc_ndims'],
                    ptform_args=[cfg['param_low'], cfg['param_upp']],
                    bound=cfg['dynesty_bound'],
                    sample=cfg['dynesty_sample'],
                    bootstrap=cfg['dynesty_bootstrap'],
                    enlarge=cfg['dynesty_enlarge'],
                    walks=cfg['dynesty_walks'],
                    update_interval=cfg['dynesty_update_interval'],
                    pool=pool)

                dsampler.run_nested(
                    dlogz_init=cfg['dynesty_dlogz_ini'],
                    nlive_init=cfg['dynesty_nlive_ini'],
                    maxiter_init=cfg['dynesty_maxiter_ini'],
                    maxcall_init=cfg['dynesty_maxcall_ini'],
                    nlive_batch=cfg['dynesty_nlive_run'],
                    maxiter_batch=cfg['dynesty_maxiter_run'],
                    maxcall_batch=cfg['dynesty_maxcall_run'])

    else:
        if args.sampler == 'nested':
            dsampler = dynesty.NestedSampler(
                asap_ln_like_global,
                asap_flat_prior_transform,
                cfg['mcmc_ndims'],
                ptform_args=[cfg['param_low'], cfg['param_upp']],
                bound=cfg['dynesty_bound'],
                sample=cfg['dynesty_sample'],
                nlive=cfg['dynesty_nlive_ini'],
                bootstrap=cfg['dynesty_bootstrap'],
                enlarge=cfg['dynesty_enlarge'],
                walks=cfg['dynesty_walks'],
                update_interval=cfg['dynesty_update_interval'])

            dsampler.run_nested(
                dlogz=cfg['dynesty_dlogz_run'],
                maxiter=cfg['dynesty_maxiter_run'],
                maxcall=cfg['dynesty_maxcall_run'])
        else:
            dsampler = dynesty.DynamicNestedSampler(
                asap_ln_like_global,
                asap_flat_prior_transform,
                cfg['mcmc_ndims'],
                ptform_args=[cfg['param_low'], cfg['param_upp']],
                bound=cfg['dynesty_bound'],
                sample=cfg['dynesty_sample'],
                bootstrap=cfg['dynesty_bootstrap'],
                enlarge=cfg['dynesty_enlarge'],
                walks=cfg['dynesty_walks'],
                update_interval=cfg['dynesty_update_interval'])

            dsampler.run_nested(
                dlogz_init=cfg['dynesty_dlogz_ini'],
                nlive_init=cfg['dynesty_nlive_ini'],
                maxiter_init=cfg['dynesty_maxiter_ini'],
                maxcall_init=cfg['dynesty_maxcall_ini'],
                nlive_batch=cfg['dynesty_nlive_run'],
                maxiter_batch=cfg['dynesty_maxiter_run'],
                maxcall_batch=cfg['dynesty_maxcall_run'])

    dynesty_results = dsampler.results

    # Show a summary
    print(dynesty_results.summary())

    # Pickle the results
    mcmc_save_pickle(cfg['dynesty_results_file'], dynesty_results)

    return dynesty_results


def asap_emcee_fit(args, verbose=True):
    """Run A.S.A.P model using emcee."""
    assert use_emcee, "# Can not import emcee!"

    global cfg, obs_data, um_data
    # Parse the configuration file  .
    config_initial = parse_config(args.config)

    # Load the data
    cfg, obs_data, um_data = initial_model(config_initial, verbose=verbose)

    # Decide the Ensemble moves for walkers
    emcee_move = mcmc_setup_moves(cfg)

    # Initialize the model
    mcmc_ini_position = mcmc_initial_guess(
        cfg['param_ini'], cfg['param_sig'], cfg['mcmc_nwalkers'],
        cfg['mcmc_ndims'])

    if cfg['mcmc_nthreads'] > 1:
        from multiprocessing import Pool
        from contextlib import closing

        with closing(Pool(processes=cfg['mcmc_nthreads'])) as pool:
            mcmc_sampler = emcee.EnsembleSampler(
                cfg['mcmc_nwalkers'],
                cfg['mcmc_ndims'],
                asap_ln_prob_global,
                moves=emcee_move,
                pool=pool)

            # Burn-in
            mcmc_burnin_result = asap_emcee_burnin(
                mcmc_sampler, mcmc_ini_position, cfg, verbose=True)

            mcmc_sampler.reset()

            # MCMC run
            mcmc_run_result = asap_emcee_run(
                mcmc_sampler, mcmc_burnin_result, cfg, verbose=True)
    else:
        mcmc_sampler = emcee.EnsembleSampler(
            cfg['mcmc_nwalkers'],
            cfg['mcmc_ndims'],
            asap_ln_prob_global,
            moves=emcee_move)

        # Burn-in
        mcmc_burnin_result = asap_emcee_burnin(
            mcmc_sampler, mcmc_ini_position, cfg, verbose=True)

        mcmc_sampler.reset()

        # MCMC run
        mcmc_run_result = asap_emcee_run(
            mcmc_sampler, mcmc_burnin_result, cfg, verbose=True)

    return mcmc_run_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config',
        help="Configuration file",
        default='asap_default_config.yaml')
    parser.add_argument(
        '-s', '--sampler', dest='sampler',
        help="Sampling method",
        default='emcee')

    args = parser.parse_args()

    if args.sampler == 'emcee':
        asap_emcee_fit(args)
    elif args.sampler == 'nested' or args.sampler == 'dynesty':
        asap_dynesty_fit(args)
    else:
        raise Exception("# Wrong sampler option! [emcee/nested/dynesty]")
