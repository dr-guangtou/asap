"""Utility functions for A.S.A.P model."""

import yaml

__all__ = ["parse_config"]


def parse_config(config_file):
    """Prepare configurations.

    Read configuration parameters from an input .yaml file.
    """
    cfg = yaml.load(open(config_file))

    return cfg


def setup_observed_data(cfg):
    """Config parameters for observed data"""
    # This is for HSC observation
    if 'obs_h0' not in cfg.keys():
        cfg['obs_h0'] = 0.7

    if 'obs_omega_m' in cfg.keys():
        cfg['obs_omega_m'] = 0.307

    cfg['cosmo'] = FlatLambdaCDM(H0=cfg['obs_h0'] * 100,
                                 Om0=cfg['obs_omega_m'])
    # --------------------------------------------------- #

    # -------------- Observed Data Related -------------- #
    # Catalog for observed galaxies
    if 'obs_dir' in cfg.keys():
        cfg['obs_dir'] = '../data/s16a_massive_wide2'

    if 'obs_cat' in cfg.keys():
        cfg['obs_cat'] = 's16a_wide2_massive_fsps1_imgsub_use_short.fits'
    if verbose:
        print("# Stellar mass catalog: %s" % cfg['obs_cat'])

    obs_mass = Table.read(os.path.join(obs_dir, obs_cat))

    # --------------------------------------------------- #
    # Observed weak lensing delta sigma profiles
    if 'obs_wl_sample' in cfg.keys():
        cfg['obs_wl_sample'] = 's16a_wide2_massive_boxbin3_default'
    if verbose:
        print("# Weak lensing profile sample: %s" % cfg['obs_wl_sample'])

    obs_wl_dir = os.path.join(obs_dir, 'dsigma')
    obs_wl_out = os.path.join(obs_wl_dir,
                              (obs_wl_sample + '_dsigma_results.pkl'))

    with open(obs_wl_out, 'rb') as f:
        self.obs_wl_bin, self.obs_wl_dsigma = pickle.load(f)

    self.obs_wl_n_bin = len(self.obs_wl_bin)
    if verbose:
        if self.obs_wl_n_bin > 1:
            print("# There are %d weak lensing profiles in this sample" %
                  self.obs_wl_n_bin)
        else:
            print("# There is 1 weak lensing profile in this sample")

    # --------------------------------------------------- #
    # Observed stellar mass functions
    if 'obs_smf_inn' in cfg.keys():
        smf_inn_file = os.path.join(obs_dir, cfg['obs_smf_inn'])
    else:
        smf_inn_file = os.path.join(
            obs_dir, 'smf', 's16a_wide2_massive_smf_m10_11.5.fits')
    if verbose:
        print("# Pre-computed SMF for inner logMs: %s" % smf_inn_file)

    if 'obs_smf_tot' in cfg.keys():
        smf_tot_file = os.path.join(obs_dir, cfg['obs_smf_tot'])
    else:
        smf_tot_file = os.path.join(
            obs_dir, 'smf', 's16a_wide2_massive_smf_m100_11.5.fits')
    if verbose:
        print("# Pre-computed SMF for total logMs: %s" % smf_tot_file)

    self.obs_smf_inn = Table.read(smf_inn_file)
    self.obs_smf_tot = Table.read(smf_tot_file)
    self.obs_smf_inn_min = np.nanmin(self.obs_smf_inn['logm_0'])
    self.obs_smf_inn_max = np.nanmax(self.obs_smf_inn['logm_1'])
    self.obs_smf_inn_nbin = len(self.obs_smf_inn)
    self.obs_smf_tot_min = np.nanmin(self.obs_smf_tot['logm_0'])
    self.obs_smf_tot_max = np.nanmax(self.obs_smf_tot['logm_1'])
    self.obs_smf_tot_nbin = len(self.obs_smf_tot)

    # Total stellar mass function for comparison (optional)
    obs_smf_full_file = os.path.join(obs_dir,
                                     's82_total_smf_z0.15_0.43.fits')
    if os.path.isfile(obs_smf_full_file):
        smf_full = Table.read(obs_smf_full_file)
        smf_full[smf_full['smf'] <= 0]['smf'] = 1E-8
        smf_full[smf_full['smf_low'] <= 0]['smf_low'] = 1E-9
        smf_full[smf_full['smf_upp'] <= 0]['smf_upp'] = 1E-7
        self.obs_smf_full = smf_full
        if verbose:
            print("# Pre-computed full SMF: %s" %
                  's82_total_smf_z0.15_0.43.fits')
    else:
        self.obs_smf_full = None

    # --------------------------------------------------- #
    # Volume of the data
    if 'obs_area' in cfg.keys():
        obs_area = cfg['obs_area']
    else:
        obs_area = 145.0

    if 'obs_z_col' in cfg.keys():
        obs_z_col = cfg['obs_z_col']
    else:
        obs_z_col = 'z_best'

    obs_zmin = np.nanmin(obs_mass[obs_z_col])
    obs_zmax = np.nanmax(obs_mass[obs_z_col])

    self.obs_volume = ((self.cosmo.comoving_volume(obs_zmax) -
                        self.cosmo.comoving_volume(obs_zmin)) *
                       (obs_area / 41254.0)).value
    if verbose:
        print("# The volume of the HSC data is %15.2f Mpc^3" %
              self.obs_volume)

    # --------------------------------------------------- #
    # Observed inner and outer mass
    if 'obs_minn_col' in cfg.keys():
        obs_minn_col = cfg['obs_minn_col']
    else:
        obs_minn_col = 'logm_10'

    if 'obs_mtot_col' in cfg.keys():
        obs_mtot_col = cfg['obs_mtot_col']
    else:
        obs_mtot_col = 'logm_100'

    self.obs_minn = obs_mass[obs_minn_col]
    self.obs_mtot = obs_mass[obs_mtot_col]

    self.obs_logms_inn = self.obs_minn[self.obs_mtot >=
                                       self.obs_smf_tot_min]
    self.obs_logms_tot = self.obs_mtot[self.obs_mtot >=
                                       self.obs_smf_tot_min]

    if verbose:
        print('# Using %s as inner stellar mass.' %
              obs_minn_col)
        print('# Using %s as total stellar mass.' %
              obs_mtot_col)
        print("# For inner stellar mass: ")
        print("    %d bins at %5.2f < logMinn < %5.2f" %
              (self.obs_smf_inn_nbin, self.obs_smf_inn_min,
               self.obs_smf_inn_max))
        print("# For total stellar mass: ")
        print("    %d bins at %5.2f < logMtot < %5.2f" %
              (self.obs_smf_tot_nbin, self.obs_smf_tot_min,
               self.obs_smf_tot_max))

def setupUniverseMachine(self, verbose=False, **cfg):
