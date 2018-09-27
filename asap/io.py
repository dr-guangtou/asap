"""Data input for A.S.A.P model."""
from __future__ import print_function, division, unicode_literals

import os
import pickle

import yaml

import numpy as np

from astropy.table import Table, Column
from astropy.cosmology import FlatLambdaCDM

__all__ = ["load_obs", "load_um"]


def load_obs(cfg, verbose=True):
    """Load the observed data."""
    # Galaxy catalog.
    obs_mass = Table.read(os.path.join(cfg['obs_dir'],
                                       cfg['obs_cat']))

    obs_minn = np.array(obs_mass[cfg['obs_minn_col']])
    obs_mtot = np.array(obs_mass[cfg['obs_mtot_col']])

    with open(cfg['obs_wl_out'], 'rb') as f:
        # BUG: Tricky work around for pickling Python 2 array in Python 3
        # https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        obs_wl_bin, obs_wl_dsigma = u.load()

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

    cfg['obs_ngal_use'] = ((obs_mtot >= cfg['obs_smf_tot_min']) &
                           (obs_minn >= cfg['obs_smf_inn_min'])).sum()

    # TODO : test this margin
    cfg['obs_min_mtot'] = cfg['obs_smf_tot_min'] - 0.1

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


def load_um(cfg):
    """Load the UniverseMachine data."""
    um_mock = Table(np.load(os.path.join(cfg['um_dir'],
                                         cfg['um_model'])))

    # Only select the useful columns
    cols_use = ['halo_id', 'upid', 'sm', 'icl', 'x', 'y', 'z',
                'mtot_galaxy', 'mstar_mhalo', 'logms_gal',
                'logms_icl', 'logms_tot', 'logms_halo',
                'logmh_vir', 'logmh_peak', 'logmh_host']
    um_mock_use = um_mock[cols_use]

    # Value added a few useful columns
    um_mock_use.add_column(Column(data=(um_mock_use['mtot_galaxy'] /
                                        um_mock_use['mstar_mhalo']),
                                  name='frac_cen_tot'))
    um_mock_use.add_column(Column(data=(um_mock_use['sm'] /
                                        um_mock_use['mtot_galaxy']),
                                  name='frac_ins_cen'))
    um_mock_use.add_column(Column(data=(um_mock_use['icl'] /
                                        um_mock_use['mtot_galaxy']),
                                  name='frac_exs_cen'))
    um_mock_use = um_mock_use.as_array()

    # Load the pre-compute lensing pairs
    um_mass_encl = np.load(os.path.join(cfg['um_dir'],
                                        cfg['um_wl_cat']))
    assert len(um_mock_use) == len(um_mass_encl)

    # Mask for central galaxies
    mask_central = (um_mock_use['upid'] == -1)

    # Mask for massive enough halo
    mask_mass = (um_mock_use[cfg['um_halo_col']] >= cfg['um_min_mvir'])

    return {'um_mock': um_mock_use[mask_mass],
            'um_mass_encl': um_mass_encl[mask_mass, :],
            'mask_central': mask_central[mask_mass]}
