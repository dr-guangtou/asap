"""Setup the model parameters."""

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

    elif cfg['model_type'] == 'frac1':
        # Number of parameters
        cfg['mcmc_ndims'] = 6
        cfg['mcmc_labels'] = [r'$a_{\mathrm{SMHR}}$', r'$b_{\mathrm{SMHR}}$',
                              r'$a_{\sigma \log M_{\star}}$',
                              r'$b_{\sigma \log M_{\star}}$',
                              r'$\mathrm{f}_{\mathrm{in-situ}}$',
                              r'$\mathrm{f}_{\mathrm{ex-situ}}$']
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
    else:
        raise Exception("# Wrong model! Has to be 'simple' or `frac1`")

    assert len(cfg['param_ini']) == cfg['mcmc_ndims']
    assert len(cfg['param_low']) == cfg['mcmc_ndims']
    assert len(cfg['param_upp']) == cfg['mcmc_ndims']
    assert len(cfg['param_sig']) == cfg['mcmc_ndims']
    # --------------------------------------------------- #

    # ------------------- MCMC Related ------------------ #
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
    cfg['mcmc_burnin_file'] = cfg['mcmc_prefix'] + '_burnin.pkl'
    cfg['mcmc_run_file'] = cfg['mcmc_prefix'] + '_run.pkl'
    cfg['mcmc_burnin_chain_file'] = (cfg['mcmc_prefix'] +
                                     '_burnin_chain.pkl')
    cfg['mcmc_run_chain_file'] = (cfg['mcmc_prefix'] +
                                  '_run_chain.pkl')
    cfg['mcmc_run_samples_file'] = cfg['mcmc_prefix'] + '_samples.npz'
    # --------------------------------------------------- #

    return cfg
