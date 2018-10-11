"""Module to deal with configuration file."""
from __future__ import print_function, division, unicode_literals

import os
import yaml

import numpy as np

from . import parameters as pa

__all__ = ["parse_config", "config_obs", "config_um", "config_emcee"]


def parse_config(config_file):
    """Parse the `yaml` format configuration file.

    Parameters
    ----------
    config_file : string
        Location and name of the configuration file in `yaml` format.

    Return
    ------
        Configuration parameters in dictionary format.

    """
    return yaml.load(open(config_file))


def config_obs(cfg_obs, verbose=False):
    """Config parameters for observed data.

    Parameters
    ----------
    cfg_obs : dict
        The `obs` section from the configuration file.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Updated configuration parameters for observations.

    """
    if verbose:
        print("\n# Observations:")
    # ----------------- Cosmology ----------------------- #
    cfg_obs['h0'] = 0.7 if 'h0' not in cfg_obs else cfg_obs['h0']
    cfg_obs['omega_m'] = 0.7 if 'omega_m' not in cfg_obs else cfg_obs['omega_m']
    # --------------------------------------------------- #

    # -------------- Observed Data Related -------------- #
    # Catalog for observed galaxies
    cfg_obs['dir'] = '' if 'dir' not in cfg_obs else cfg_obs['dir']

    if 'galaxy' not in cfg_obs:
        # 's16a_wide2_massive_fsps1_imgsub_use_short.fits'
        raise Exception("# We need an input galaxy catalog !")
    else:
        if verbose:
            print("# Galaxy catalog: %s" % cfg_obs['galaxy'])
        cfg_obs['galaxy'] = os.path.join(cfg_obs['dir'], cfg_obs['galaxy'])
        if not os.path.isfile(cfg_obs['galaxy']):
            raise Exception("# Can not find the galaxy catalog: %s" % cfg_obs['galaxy'])

    # --------------------------------------------------- #
    # Observed weak lensing delta sigma profiles
    if 'dsigma' not in cfg_obs:
        # 's16a_wide2_massive_boxbin5_default'
        raise Exception("# We need the input weak lensing results !")
    else:
        if verbose:
            print("# DSigma results: %s" % cfg_obs['dsigma'])
        cfg_obs['dsigma'] = os.path.join(
            cfg_obs['dir'], 'dsigma', cfg_obs['dsigma'])
        if not os.path.isfile(cfg_obs['dsigma']):
            raise Exception("# Can not find the weak lensing results: %s" % cfg_obs['dsigma'])

    # --------------------------------------------------- #
    # Observed stellar mass functions
    if 'smf_inn' not in cfg_obs:
        # TODO: we can estimate SMF
        raise Exception("# We need the SMF of inner aperture mass !")
    else:
        if verbose:
            print("# SMF of inner Mstar: %s" % cfg_obs['smf_inn'])
        cfg_obs['smf_inn'] = os.path.join(cfg_obs['dir'], 'smf', cfg_obs['smf_inn'])
        if not os.path.isfile(cfg_obs['smf_inn']):
            raise Exception("# Can not find SMF_inn: %s" % cfg_obs['smf_inn'])

    if 'smf_tot' not in cfg_obs:
        # TODO: we can estimate SMF
        raise Exception("# We need the SMF of outer aperture mass !")
    else:
        if verbose:
            print("# SMF of total Mstar: %s" % cfg_obs['smf_tot'])
        cfg_obs['smf_tot'] = os.path.join(cfg_obs['dir'], 'smf', cfg_obs['smf_tot'])
        if not os.path.isfile(cfg_obs['smf_tot']):
            raise Exception("# Can not find SMF_tot: %s" % cfg_obs['smf_tot'])

    if 'smf_cov' not in cfg_obs:
        cfg_obs['smf_cov'] = None
    else:
        if verbose:
            print("# Covariances for SMFs: %s" % cfg_obs['smf_cov'])
        cfg_obs['smf_cov'] = os.path.join(cfg_obs['dir'], 'smf', cfg_obs['smf_cov'])
        if not os.path.isfile(cfg_obs['smf_cov']):
            raise Exception("# Can not find the covariance for SMF: %s" % cfg_obs['smf_cov'])

    # Total stellar mass function for comparison (optional)
    if 'smf_full' not in cfg_obs:
        cfg_obs['smf_full'] = None
    else:
        if verbose:
            print("# Reference SMF: %s" % cfg_obs['smf_full'])
        cfg_obs['smf_full'] = os.path.join(cfg_obs['dir'], cfg_obs['smf_full'])
        if not os.path.isfile(cfg_obs['smf_full']):
            raise Exception("# Can not find the reference SMF: %s" % cfg_obs['smf_full'])

    # --------------------------------------------------- #
    # Volume of the data
    cfg_obs['area'] = 145.0 if 'area' not in cfg_obs else cfg_obs['area']

    # --------------------------------------------------- #
    # Observed inner and outer mass
    cfg_obs['z_col'] = 'z_best' if 'z_col' not in cfg_obs else cfg_obs['z_col']
    cfg_obs['minn_col'] = 'logm_10' if 'minn_col' not in cfg_obs else cfg_obs['minn_col']
    cfg_obs['mtot_col'] = 'logm_max' if 'mtot_col' not in cfg_obs else cfg_obs['mtot_col']

    if verbose:
        print('# Column of inner Mstar: %s' % cfg_obs['minn_col'])
        print('# Column of total Mstar: %s' % cfg_obs['mtot_col'])

    return cfg_obs


def config_um(cfg_um, verbose=False):
    """Config the UniverseMachine data.

    Parameters
    ----------
    cfg_um : dict
        The `obs` section from the configuration file.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Updated configuration parameters for the UniverseMachine model.

    """
    if verbose:
        print("\n# UniverseMachine:")
    # ----------------- Cosmology ----------------------- #
    cfg_um['h0'] = 0.678 if 'h0' not in cfg_um else cfg_um['h0']
    cfg_um['omega_m'] = 0.307 if 'omega_m' not in cfg_um else cfg_um['omega_m']

    # ---------- UniverseMachine Mock Related ----------- #
    cfg_um['dir'] = '' if 'dir' not in cfg_um else cfg_um['dir']

    # Value added galaxy catalog
    if 'galaxy' not in cfg_um:
        raise Exception("# Need to have a UniverseMachine galaxy catalog !")
    else:
        if verbose:
            print("# Galaxy catalog : %s" % cfg_um['galaxy'])
        cfg_um['galaxy'] = os.path.join(cfg_um['dir'], cfg_um['galaxy'])
        if not os.path.isfile(cfg_um['galaxy']):
            raise Exception("# Can not find galaxy catalog : %s" % cfg_um['galaxy'])

    # Precomputed weak lensing paris
    if 'dsigma' not in cfg_um:
        raise Exception("# Need to have a UniverseMachine WL pre-compute results !")
    else:
        if verbose:
            print("# DSigma results : %s" % cfg_um['dsigma'])
        cfg_um['dsigma'] = os.path.join(cfg_um['dir'], cfg_um['dsigma'])
        if not os.path.isfile(cfg_um['dsigma']):
            raise Exception("# Can not find DSigma results : %s" % cfg_um['dsigma'])

    # --------------- Simulation Related ---------------- #
    # Default simulation is SMDPL
    cfg_um['lbox'] = 400.0 if 'lbox' not in cfg_um else cfg_um['lbox'] # Mpc/h

    cfg_um['volume'] = np.power(cfg_um['lbox'] / cfg_um['h0'], 3)
    if verbose:
        print("# Volumn of the simulation: %15.2f Mpc^3" % cfg_um['volume'])

    # Minimum Virial mass used in the modeling
    cfg_um['min_mvir'] = 11.5 if 'min_mvir' not in cfg_um else cfg_um['min_mvir']

    # Redshift of the simulatin snapshot
    cfg_um['redshift'] = 0.3637 if 'redshift' not in cfg_um else cfg_um['redshift']

    # Minimum and maximum radius of the DSigma profile, number of radius bins
    cfg_um['wl_minr'] = 0.08 if 'wl_minr' not in cfg_um else cfg_um['wl_minr']
    cfg_um['wl_maxr'] = 50.0 if 'wl_maxr' not in cfg_um else cfg_um['wl_maxr']
    cfg_um['wl_nbin'] = 22 if 'wl_nbin' not in cfg_um else cfg_um['wl_nbin']

    # Default: Do not add the contribution of stellar mass
    cfg_um['wl_add_stellar'] = False if 'wl_add_stellar' not in cfg_um else cfg_um['wl_add_stellar']

    cfg_um['mtot_nbin'] = 80 if 'mtot_nbin' not in cfg_um else cfg_um['mtot_nbin']
    cfg_um['mtot_nbin_min'] = 7 if 'mtot_nbin_min' not in cfg_um else cfg_um['mtot_nbin_min']
    cfg_um['min_nobj_per_bin'] = 30 if 'ngal_bin_min' not in cfg_um else cfg_um['min_nobj_per_bin']

    cfg_um['min_scatter'] = 0.01 if 'min_scatter' not in cfg_um else cfg_um['min_scatter']

    cfg_um['halo_col'] = 'logmh_host' if 'halo_col' not in cfg_um else cfg_um['halo_col']
    cfg_um['star_col'] = 'logms_tot' if 'star_col' not in cfg_um else cfg_um['star_col']
    if verbose:
        print("# Halo mass : %s" % cfg_um['halo_col'])
        print("# Stellar mass : %s" % cfg_um['star_col'])

    return cfg_um


def config_emcee(cfg_emcee, verbose=False):
    """Configuration parameters for using emcee sampler.

    Parameters
    ----------
    cfg_emcee : dict
        Configuration parameters for emcee sampler.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Updated configuration parameters about emcee sampler.

    """
    # Number of emcee walkers
    cfg_emcee['nwalkers'] = 128 if 'nwalkers' not in cfg_emcee else cfg_emcee['nwalkers']

    # Number of emcee walkers during burn-in
    if 'nwalkers_burnin' not in cfg_emcee:
        cfg_emcee['nwalkers_burnin'] = cfg_emcee['nwalkers']

    # Number of emcee runs
    cfg_emcee['nsamples'] = 200 if 'nsamples' not in cfg_emcee else cfg_emcee['nsamples']

    # Number of burn-in runs
    cfg_emcee['nburnin'] = 200 if 'nburnin' not in cfg_emcee else cfg_emcee['nburnin']

    # Number of processors to run on
    cfg_emcee['nthreads'] = 1 if 'nthreads' not in cfg_emcee else cfg_emcee['nthreads']

    # Choice of emcee move
    cfg_emcee['moves'] = 'stretch' if 'moves' not in cfg_emcee else cfg_emcee['moves']

    # Whether to use different move during burn-in
    if 'moves_burnin' not in cfg_emcee:
        cfg_emcee['moves_burnin'] = cfg_emcee['moves']

    if verbose:
        print("#    Use %5d walkers with %10s moves for %5d steps of burn-in" % (
            cfg_emcee['nwalkers_burnin'], cfg_emcee['moves_burnin'], cfg_emcee['nburnin']))
        print("#    Use %5d walkers with %10s moves for %5d steps of sampling" % (
            cfg_emcee['nwalkers'], cfg_emcee['moves'], cfg_emcee['nsamples']))

    # The a parameter for stretch move
    cfg_emcee['stretch_a'] = 4 if 'stretch_a' not in cfg_emcee else cfg_emcee['stretch_a']

    cfg_emcee['walk_s'] = None if 'walk_s' not in cfg_emcee else cfg_emcee['walk_s']

    cfg_emcee['de_sigma'] = 0.2 if 'de_sigma' not in cfg_emcee else cfg_emcee['de_sigma']

    return cfg_emcee


def config_model(cfg_model, verbose=False):
    """Basic configuration of the model.

    Parameters
    ----------
    cfg : dict
        Configuration parameters of the model.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Updated configuration parameters of the model.

    """
    # Type of the model
    cfg_model['type'] = 'basic' if 'type' not in cfg_model else cfg_model['type']

    # Choice of the sampler
    cfg_model['sampler'] = 'emcee' if 'sampler' not in cfg_model else cfg_model['sampler']

    # Do not change anything below unless you know what you are doing.
    # Only fit SMF
    cfg_model['smf_only'] = False if 'smf_only' not in cfg_model else cfg_model['smf_only']

    # Only fit weak lensing
    cfg_model['wl_only'] = False if 'wl_only' not in cfg_model else cfg_model['wl_only']

    # Weight for the likelihoods of the weak lensing data
    cfg_model['wl_weight'] = 1.0 if 'wl_weight' not in cfg_model else cfg_model['wl_weight']

    # About the output file
    cfg_model['out_dir'] = '' if 'out_dir' not in cfg_model else cfg_model['out_dir']

    cfg_model['prefix'] = 'asap_model' if 'prefix' not in cfg_model else cfg_model['prefix']
    if verbose:
        print("# Running model: %s" % cfg_model['prefix'])

    # Config the sampler
    if cfg_model['sampler'] == 'emcee':
        if verbose:
            print("#    Will use emcee as sampler ...")
        if 'emcee' not in cfg_model:
            cfg_model['emcee'] = {}
        cfg_model['emcee'] = config_emcee(cfg_model['emcee'], verbose=verbose)
    else:
        # Just use emcee for now
        raise Exception("# Wrong choice of sampler: [emcee]")

    return cfg_model


def update_configuration(cfg, verbose=False):
    """Basic configuration of the model.

    Parameters
    ----------
    cfg : dict
        Configuration parameters of the model.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Updated configuration parameters of the model.

    """
    # Basic model configuration
    cfg['model'] = config_model(cfg['model'], verbose=verbose)

    # Config the observation data
    cfg['obs'] = config_obs(cfg['obs'], verbose=verbose)

    # Config the UniverseMachine data
    cfg['um'] = config_um(cfg['um'], verbose=verbose)

    # Config the parameters
    params = pa.AsapParams(cfg['parameters'])

    return cfg, params
