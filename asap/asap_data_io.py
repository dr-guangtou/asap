"""Data input for A.S.A.P model."""

import os
import pickle

import yaml

import numpy as np

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

__all__ = ["parse_config",
           "load_observed_data", "config_observed_data",
           "load_um_data", "config_um_data"]


def parse_config(config_file):
    """Prepare configurations.

    Read configuration parameters from an input .yaml file.
    """
    cfg = yaml.load(open(config_file))

    return cfg


def load_observed_data(cfg, verbose=True):
    """Load the observed data."""
    # Galaxy catalog.
    obs_mass = Table.read(os.path.join(cfg['obs_dir'],
                                       cfg['obs_cat']))

    obs_minn = obs_mass[cfg['obs_minn_col']]
    obs_mtot = obs_mass[cfg['obs_mtot_col']]

    with open(cfg['obs_wl_out'], 'rb') as f:
        obs_wl_bin, obs_wl_dsigma = pickle.load(f)

    cfg['obs_wl_n_bin'] = len(obs_wl_bin)
    if verbose:
        if cfg['obs_wl_n_bin'] > 1:
            print("# There are %d weak lensing profiles in this sample" %
                  cfg['obs_wl_n_bin'])
        else:
            print("# There is 1 weak lensing profile in this sample")

    cfg['obs_dsigma_n_data'] = len(obs_wl_dsigma[0].r) * cfg['obs_wl_n_bin']

    if os.path.splitext(cfg['smf_inn_file'])[-1] == '.npy':
        obs_smf_inn = np.load(cfg['smf_inn_file'])
    else:
        obs_smf_inn = Table.read(cfg['smf_inn_file'])

    if os.path.splitext(cfg['smf_tot_file'])[-1] == '.npy':
        obs_smf_tot = np.load(cfg['smf_tot_file'])
    else:
        obs_smf_tot = Table.read(cfg['smf_tot_file'])

    cfg['obs_smf_inn_min'] = np.min(obs_smf_inn['logm_0'])
    cfg['obs_smf_inn_max'] = np.max(obs_smf_inn['logm_1'])
    cfg['obs_smf_inn_nbin'] = len(obs_smf_inn)

    cfg['obs_smf_tot_min'] = np.min(obs_smf_tot['logm_0'])
    cfg['obs_smf_tot_max'] = np.max(obs_smf_tot['logm_1'])
    cfg['obs_smf_tot_nbin'] = len(obs_smf_tot)

    cfg['obs_smf_n_data'] = cfg['obs_smf_tot_nbin'] + cfg['obs_smf_inn_nbin']

    if cfg['smf_cov_file'] is not None:
        obs_smf_cov = np.load(cfg['smf_cov_file'])
        assert cfg['obs_smf_n_data'] == len(obs_smf_cov)
    else:
        obs_smf_cov = None

    if verbose:
        print("# SMF for total stellar mass: ")
        print("  %7.4f -- %7.4f in %d bins" % (cfg['obs_smf_tot_min'],
                                               cfg['obs_smf_tot_max'],
                                               cfg['obs_smf_tot_nbin']))
        print("# SMF for inner stellar mass: ")
        print("  %7.4f -- %7.4f in %d bins" % (cfg['obs_smf_inn_min'],
                                               cfg['obs_smf_inn_max'],
                                               cfg['obs_smf_inn_nbin']))

    obs_logms_inn = obs_minn[obs_mtot >= cfg['obs_smf_tot_min']]
    obs_logms_tot = obs_mtot[obs_mtot >= cfg['obs_smf_tot_min']]

    if os.path.isfile(cfg['obs_smf_full_file']):
        smf_full = Table.read(cfg['obs_smf_full_file'])
        smf_full[smf_full['smf'] <= 0]['smf'] = 1E-8
        smf_full[smf_full['smf_low'] <= 0]['smf_low'] = 1E-9
        smf_full[smf_full['smf_upp'] <= 0]['smf_upp'] = 1E-7
        obs_smf_full = smf_full
        if verbose:
            print("# Pre-computed full SMF: %s" % cfg['obs_smf_full_fits'])
    else:
        obs_smf_full = None

    if verbose:
        print("# For inner stellar mass: ")
        print("    %d bins at %5.2f < logMinn < %5.2f" %
              (cfg['obs_smf_inn_nbin'], cfg['obs_smf_inn_min'],
               cfg['obs_smf_inn_max']))
        print("# For total stellar mass: ")
        print("    %d bins at %5.2f < logMtot < %5.2f" %
              (cfg['obs_smf_tot_nbin'], cfg['obs_smf_tot_min'],
               cfg['obs_smf_tot_max']))

    obs_zmin = np.nanmin(obs_mass[cfg['obs_z_col']])
    obs_zmax = np.nanmax(obs_mass[cfg['obs_z_col']])

    obs_volume = ((cfg['obs_cosmo'].comoving_volume(obs_zmax) -
                   cfg['obs_cosmo'].comoving_volume(obs_zmin)) *
                  (cfg['obs_area'] / 41254.0)).value
    cfg['obs_volume'] = obs_volume

    if verbose:
        print("# The volume of the HSC data is %15.2f Mpc^3" % obs_volume)

    return {'obs_mass': obs_mass,
            'obs_minn': obs_minn, 'obs_mtot': obs_mtot,
            'obs_logms_inn': obs_logms_inn, 'obs_logms_tot': obs_logms_tot,
            'obs_wl_bin': obs_wl_bin, 'obs_wl_dsigma': obs_wl_dsigma,
            'obs_smf_inn': obs_smf_inn, 'obs_smf_tot': obs_smf_tot,
            'obs_smf_full': obs_smf_full, 'obs_smf_cov': obs_smf_cov,
            'obs_volume': obs_volume}, cfg


def config_observed_data(cfg, verbose=True):
    """Config parameters for observed data."""
    # This is for HSC observation
    if 'obs_h0' not in cfg.keys():
        cfg['obs_h0'] = 0.7

    if 'obs_omega_m' in cfg.keys():
        cfg['obs_omega_m'] = 0.307

    cfg['obs_cosmo'] = FlatLambdaCDM(H0=cfg['obs_h0'] * 100,
                                     Om0=cfg['obs_omega_m'])
    # --------------------------------------------------- #

    # -------------- Observed Data Related -------------- #
    # Catalog for observed galaxies
    if 'obs_dir' not in cfg.keys():
        cfg['obs_dir'] = '../data/s16a_massive_wide2'

    if 'obs_cat' not in cfg.keys():
        cfg['obs_cat'] = 's16a_wide2_massive_fsps1_imgsub_use_short.fits'
    if verbose:
        print("# Stellar mass catalog: %s" % cfg['obs_cat'])

    # --------------------------------------------------- #
    # Observed weak lensing delta sigma profiles
    if 'obs_wl_sample' not in cfg.keys():
        cfg['obs_wl_sample'] = 's16a_wide2_massive_boxbin5_default'
    if verbose:
        print("# Weak lensing profile sample: %s" % cfg['obs_wl_sample'])

    obs_wl_dir = os.path.join(cfg['obs_dir'], 'dsigma')
    cfg['obs_wl_out'] = os.path.join(
        obs_wl_dir, cfg['obs_wl_sample'] + '_dsigma_results.pkl')

    # --------------------------------------------------- #
    # Observed stellar mass functions
    if 'obs_smf_inn' in cfg.keys():
        cfg['smf_inn_file'] = os.path.join(cfg['obs_dir'], 'smf',
                                           cfg['obs_smf_inn'])
    else:
        cfg['smf_inn_file'] = os.path.join(
            cfg['obs_dir'], 'smf', 's16a_wide2_massive_smf_m10_new.npy')
    if verbose:
        print("# Pre-computed SMF for inner logMs: %s" % cfg['smf_inn_file'])

    if 'obs_smf_tot' in cfg.keys():
        cfg['smf_tot_file'] = os.path.join(cfg['obs_dir'], 'smf',
                                           cfg['obs_smf_tot'])
    else:
        cfg['smf_tot_file'] = os.path.join(
            cfg['obs_dir'], 'smf', 's16a_wide2_massive_smf_mmax_new.npy')
    if verbose:
        print("# Pre-computed SMF for total logMs: %s" % cfg['smf_tot_file'])

    if 'obs_smf_cov' in cfg.keys():
        cfg['smf_cov_file'] = os.path.join(cfg['obs_dir'], 'smf',
                                           cfg['obs_smf_cov'])
    else:
        cfg['smf_cov_file'] = None
    if verbose:
        print("# Covariances for SMFs: %s" % cfg['smf_cov_file'])

    # Total stellar mass function for comparison (optional)
    if 'obs_smf_full_fits' not in cfg.keys():
        cfg['obs_smf_full_fits'] = 'primus_smf_z0.3_0.4.fits'

    cfg['obs_smf_full_file'] = os.path.join(
        cfg['obs_dir'], cfg['obs_smf_full_fits'])

    # --------------------------------------------------- #
    # Volume of the data
    if 'obs_area' not in cfg.keys():
        cfg['obs_area'] = 145.0

    if 'obs_z_col' not in cfg.keys():
        cfg['obs_z_col'] = 'z_best'

    # --------------------------------------------------- #
    # Observed inner and outer mass
    if 'obs_minn_col' not in cfg.keys():
        cfg['obs_minn_col'] = 'logm_10'

    if 'obs_mtot_col' not in cfg.keys():
        cfg['obs_mtot_col'] = 'logm_max'

    if verbose:
        print('# Using %s as inner stellar mass.' %
              cfg['obs_minn_col'])
        print('# Using %s as total stellar mass.' %
              cfg['obs_mtot_col'])

    return cfg


def load_um_data(cfg, verbose=True):
    """Load the UniverseMachine data."""
    um_mock = Table(np.load(os.path.join(cfg['um_dir'],
                                         cfg['um_model'])))
    um_mass_encl = np.load(os.path.join(cfg['um_dir'],
                                        cfg['um_wl_cat']))
    assert len(um_mock) == len(um_mass_encl)

    # Mask for central galaxies
    mask_central = um_mock['mask_central']

    return {'um_mock': um_mock, 'um_mass_encl': um_mass_encl,
            'mask_central': mask_central}


def config_um_data(cfg, verbose=False, **kwargs):
    """Config the UniverseMachine data."""
    # ---------- UniverseMachine Mock Related ----------- #
    if 'um_dir' not in cfg.keys():
        cfg['um_dir'] = '../data/s16a_massive_wide2/um2'

    # Default model is SMDPL
    if 'um_lbox' not in cfg.keys():
        cfg['um_lbox'] = 400.0  # Mpc/h

    if 'um_h0' not in cfg.keys():
        cfg['um_h0'] = 0.678

    if 'um_omega_m' not in cfg.keys():
        cfg['um_omega_m'] = 0.307

    cfg['um_cosmo'] = FlatLambdaCDM(H0=cfg['um_h0'] * 100.0,
                                    Om0=cfg['um_omega_m'])

    cfg['um_volume'] = np.power(cfg['um_lbox'] / cfg['um_h0'], 3)
    if verbose:
        print("# The volume of the UniverseMachine mock is %15.2f Mpc^3" %
              cfg['um_volume'])

    # Value added catalog
    if 'um_model' not in cfg.keys():
        cfg['um_model'] = 'um_smdpl_0.7124_new_vagc_mpeak_11.5.npy'

    # Precomputed weak lensing paris
    if 'um_wl_cat' not in cfg.keys():
        cfg['um_wl_cat'] = ('um_smdpl_0.7124_new_vagc_mpeak_11.5' +
                            '_10m_r_0.08_50_22bins.npy')

    if 'um_min_mvir' not in cfg.keys():
        cfg['um_min_mvir'] = 11.5

    if 'um_redshift' not in cfg.keys():
        cfg['um_redshift'] = 0.3637

    if 'um_wl_minr' not in cfg.keys():
        cfg['um_wl_minr'] = 0.08

    if 'um_wl_maxr' not in cfg.keys():
        cfg['um_wl_maxr'] = 50.0

    if 'um_wl_nbin' not in cfg.keys():
        cfg['um_wl_nbin'] = 22

    if 'um_wl_add_stellar' not in cfg.keys():
        cfg['um_wl_add_stellar'] = False

    if 'um_mtot_nbin' not in cfg.keys():
        cfg['um_mtot_nbin'] = 80

    if 'um_mtot_nbin_min' not in cfg.keys():
        cfg['um_mtot_nbin_min'] = 7

    if 'um_ngal_bin_min' not in cfg.keys():
        cfg['um_min_nobj_per_bin'] = 30

    if 'um_min_scatter' not in cfg.keys():
        cfg['um_min_scatter'] = 0.01

    if 'um_halo_col' not in cfg.keys():
        cfg['um_halo_col'] = 'logmh_vir'

    if 'um_star_col' not in cfg.keys():
        cfg['um_star_col'] = 'logms_tot'

    return cfg
