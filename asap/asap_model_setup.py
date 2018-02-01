"""Setup the model parameters."""

import os

__all__ = ['setup_model']


def setup_model(cfg, verbose=True):
    """Configure MCMC run and plots."""
    if 'model_type' not in cfg.keys():
        cfg['model_type'] = 'frac1'

    if cfg['model_type'] == 'simple':
        # Number of parameters
        cfg['mcmc_ndims'] = 4
        cfg['mcmc_labels'] = [r'$a_{\mathrm{SMHR}}$', r'$b_{\mathrm{SMHR}}$',
                              r'$a_{\sigma \log M_{\star}}$',
                              r'$b_{\sigma \log M_{\star}}$']
        # Initial values
        if 'param_ini' not in cfg.keys():
            cfg['param_ini'] = [0.59901, 3.69888, -0.0824, 1.2737]
        # Lower bounds
        if 'param_low' not in cfg.keys():
            cfg['param_low'] = [0.2, -1.5, -0.2, 0.0]
        # Upper bounds
        if 'param_upp' not in cfg.keys():
            cfg['param_upp'] = [1.0, 8.0, 0.0, 1.6]
        # Step to randomize the initial guesses
        if 'param_sig' not in cfg.keys():
            cfg['param_sig'] = [0.1, 0.3, 0.05, 0.2]

    elif cfg['model_type'] == 'frac1' or cfg['model_type'] == 'frac2':
        # Number of parameters
        cfg['mcmc_ndims'] = 6
        cfg['mcmc_labels'] = [r'$a_{\mathrm{SMHR}}$', r'$b_{\mathrm{SMHR}}$',
                              r'$a_{\sigma \log M_{\star}}$',
                              r'$b_{\sigma \log M_{\star}}$',
                              r'$\mathrm{f}_{\mathrm{ins}}$',
                              r'$\mathrm{f}_{\mathrm{exs}}$']
        # Initial values
        if 'param_ini' not in cfg.keys():
            cfg['param_ini'] = [0.599, 3.669, -0.048, 0.020, 0.80, 0.11]
        # Lower bounds
        if 'param_low' not in cfg.keys():
            cfg['param_low'] = [0.2, 0.0, -0.2, 0.0, 0.3, 0.0]
        # Upper bounds
        if 'param_upp' not in cfg.keys():
            cfg['param_upp'] = [1.0, 8.0, 0.0, 0.2, 1.0, 0.3]
        # Step to randomize the initial guesses
        if 'param_sig' not in cfg.keys():
            cfg['param_sig'] = [0.05, 0.1, 0.02, 0.005, 0.05, 0.05]

    elif cfg['model_type'] == 'frac3':
        # Number of parameters
        cfg['mcmc_ndims'] = 7
        cfg['mcmc_labels'] = [r'$a_{\mathrm{SMHR}}$',
                              r'$b_{\mathrm{SMHR}}$',
                              r'$a_{\sigma \log M_{\star}}$',
                              r'$b_{\sigma \log M_{\star}}$',
                              r'$\mathrm{f}_{\mathrm{ins}}$',
                              r'$\mathrm{f}_{\mathrm{exs}}$',
                              r'$\mathrm{f}_{\mathrm{tot}}$']
        # Initial values
        if 'param_ini' not in cfg.keys():
            cfg['param_ini'] = [0.599, 3.669, -0.048, 0.020,
                                0.80, 0.11, 0.80]
        # Lower bounds
        if 'param_low' not in cfg.keys():
            cfg['param_low'] = [0.2, 0.0, -0.2, 0.0,
                                0.3, 0.0, 0.5]
        # Upper bounds
        if 'param_upp' not in cfg.keys():
            cfg['param_upp'] = [1.0, 8.0, 0.0, 0.2,
                                1.0, 0.3, 1.0]
        # Step to randomize the initial guesses
        if 'param_sig' not in cfg.keys():
            cfg['param_sig'] = [0.05, 0.1, 0.02, 0.005, 0.05, 0.05, 0.1]

    elif cfg['model_type'] == 'frac4':
        # Number of parameters
        cfg['mcmc_ndims'] = 7
        cfg['mcmc_labels'] = [r'$a_{\mathrm{SMHR}}$',
                              r'$b_{\mathrm{SMHR}}$',
                              r'$a_{\sigma \log M_{\star}}$',
                              r'$b_{\sigma \log M_{\star}}$',
                              r'$\mathrm{f}_{\mathrm{ins}}$',
                              r'$\mathrm{f}_{\mathrm{exs},a}$',
                              r'$\mathrm{f}_{\mathrm{exs},b}$']
        # Initial values
        if 'param_ini' not in cfg.keys():
            cfg['param_ini'] = [0.599, 3.669, -0.048, 0.020,
                                0.80, 0.00, 0.10]
        # Lower bounds
        if 'param_low' not in cfg.keys():
            cfg['param_low'] = [0.2, 0.0, -0.2, 0.0,
                                0.3, -0.2, 0.0]
        # Upper bounds
        if 'param_upp' not in cfg.keys():
            cfg['param_upp'] = [1.0, 8.0, 0.0, 0.2,
                                1.0, 0.3, 0.5]
        # Step to randomize the initial guesses
        if 'param_sig' not in cfg.keys():
            cfg['param_sig'] = [0.05, 0.1, 0.02, 0.005,
                                0.05, 0.05, 0.1]

    elif cfg['model_type'] == 'frac5':
        # Number of parameters
        cfg['mcmc_ndims'] = 8
        cfg['mcmc_labels'] = [r'$a_{\mathrm{SMHR}}$',
                              r'$b_{\mathrm{SMHR}}$',
                              r'$a_{\sigma \log M_{\star}}$',
                              r'$b_{\sigma \log M_{\star}}$',
                              r'$\mathrm{f}_{\mathrm{ins},a}$',
                              r'$\mathrm{f}_{\mathrm{ins},a}$',
                              r'$\mathrm{f}_{\mathrm{exs},a}$',
                              r'$\mathrm{f}_{\mathrm{exs},b}$']
        # Initial values
        if 'param_ini' not in cfg.keys():
            cfg['param_ini'] = [0.599, 3.669, -0.048, 0.020,
                                0.00, 0.90, 0.00, 0.10]
        # Lower bounds
        if 'param_low' not in cfg.keys():
            cfg['param_low'] = [0.2, 0.0, -0.2, 0.0,
                                -0.2, 0.4, -0.2, 0.0]
        # Upper bounds
        if 'param_upp' not in cfg.keys():
            cfg['param_upp'] = [1.0, 8.0, 0.0, 0.2,
                                0.2, 1.0, 0.3, 0.5]
        # Step to randomize the initial guesses
        if 'param_sig' not in cfg.keys():
            cfg['param_sig'] = [0.05, 0.1, 0.02, 0.005,
                                0.05, 0.1, 0.05, 0.1]
    else:
        raise Exception("# Wrong model! Has to be 'simple' or `frac1-5`")

    assert len(cfg['param_ini']) == cfg['mcmc_ndims']
    assert len(cfg['param_low']) == cfg['mcmc_ndims']
    assert len(cfg['param_upp']) == cfg['mcmc_ndims']
    assert len(cfg['param_sig']) == cfg['mcmc_ndims']

    # Degree of freedom for the model: This is actually WRONG
    cfg['model_dof'] = (cfg['obs_smf_n_data'] + cfg['obs_dsigma_n_data'] -
                        cfg['mcmc_ndims'])
    # --------------------------------------------------- #

    # ------------------- Emcee Related ------------------ #
    if 'mcmc_nsamples' not in cfg.keys():
        cfg['mcmc_nsamples'] = 200

    if 'mcmc_nthreads' not in cfg.keys():
        cfg['mcmc_nthreads'] = 2

    if 'mcmc_nburnin' not in cfg.keys():
        cfg['mcmc_nburnin'] = 200

    if 'mcmc_nwalkers' not in cfg.keys():
        cfg['mcmc_nwalkers'] = 128

    if 'mcmc_smf_only' not in cfg.keys():
        cfg['mcmc_smf_only'] = False

    if 'mcmc_wl_only' not in cfg.keys():
        cfg['mcmc_wl_only'] = False

    if 'mcmc_wl_weight' not in cfg.keys():
        cfg['mcmc_wl_weight'] = 1.0

    if 'mcmc_prefix' not in cfg.keys():
        cfg['mcmc_prefix'] = 'asap_smdpl'

    if 'mcmc_out_dir' not in cfg.keys():
        cfg['mcmc_out_dir'] = '.'

    if 'mcmc_moves' not in cfg.keys():
        cfg['mcmc_moves'] = 'redblue'

    if 'mcmc_stretch_a' not in cfg.keys():
        cfg['mcmc_stretch_a'] = 4

    if 'mcmc_walk_s' not in cfg.keys():
        cfg['mcmc_walk_s'] = 3

    cfg['mcmc_burnin_file'] = os.path.join(
        cfg['mcmc_out_dir'], cfg['mcmc_prefix'] + '_burnin.npz')
    cfg['mcmc_run_file'] = os.path.join(
        cfg['mcmc_out_dir'], cfg['mcmc_prefix'] + '_run.npz')

    # ------------------ Dynesty Related ------------------ #
    cfg['dynesty_results_file'] = os.path.join(
        cfg['mcmc_out_dir'], cfg['mcmc_prefix'] + '_results.pkl')

    if 'dynesty_bound' not in cfg.keys():
        cfg['dynesty_bound'] = 'multi'

    if 'dynesty_sample' not in cfg.keys():
        cfg['dynesty_sample'] = 'rwalk'

    if 'dynesty_nlive_ini' not in cfg.keys():
        cfg['dynesty_nlive_ini'] = 200

    if 'dynesty_nlive_run' not in cfg.keys():
        cfg['dynesty_nlive_run'] = 200

    if 'dynesty_bootstrap' not in cfg.keys():
        cfg['dynesty_bootstrap'] = 40

    if 'dynesty_update_interval' not in cfg.keys():
        cfg['dynesty_update_interval'] = 0.8

    if 'dynesty_enlarge' not in cfg.keys():
        cfg['dynesty_enlarge'] = 1.0

    if 'dynesty_walks' not in cfg.keys():
        cfg['dynesty_walks'] = 25

    if 'dynesty_dlogz_ini' not in cfg.keys():
        cfg['dynesty_dlogz_ini'] = 5.0

    if 'dynesty_maxcall_ini' not in cfg.keys():
        cfg['dynesty_maxcall_ini'] = 10000

    if 'dynesty_maxiter_ini' not in cfg.keys():
        cfg['dynesty_maxiter_ini'] = 1000

    if 'dynesty_dlogz_run' not in cfg.keys():
        cfg['dynesty_dlogz_run'] = 0.1

    if 'dynesty_maxcall_run' not in cfg.keys():
        cfg['dynesty_maxcall_run'] = 50000

    if 'dynesty_maxiter_run' not in cfg.keys():
        cfg['dynesty_maxiter_run'] = 1000
    # --------------------------------------------------- #

    return cfg
