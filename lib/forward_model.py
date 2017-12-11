"""
Forward modeling the M100 and M10 SMFs along with HSC WL signals.
"""

import os
import copy
import pickle
from time import time

import emcee
# from emcee.utils import MPIPool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter

from scipy import interpolate

from astropy.table import Table, Column
from astropy.cosmology import FlatLambdaCDM

from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import delta_sigma_from_precomputed_pairs
from halotools.utils import crossmatch

# UniverseMachine Python IO
from umachine_pyio.load_mock import load_mock_from_binaries, value_added_mock
from stellar_mass_function import compute_smf, \
    bootstrap_smf
from model_predictions import total_stellar_mass_including_satellites, \
    precompute_lensing_pairs
from full_mass_profile_model import sm_profile_from_mhalo
from swot_weak_lensing import SwotWL
from load_mdpl2_mock import value_added_mdpl2_mock
# from convergence import convergence_check


class UMMassProfModel():
    """
    UniverseMachine Mhalo-M100-M10 models.

    Based on the Mh_vir, Ms_GAL, Ms_ICL, and Ms_Halo predicted by the
    MhaloUniverseMachine model.

    At UniverseMachine side:
        Mh_vir  : Virial mass of the host halo
        Ms_GAL  : Stellar mass of the "main-body" of the galaxy
                  (so-called BCG mass)
        Ms_ICL  : Stellar mass of the ICL component (accreted stars)
        Ms_Tot  : M_GAL + M_ICL, total stellar mass of the galaxy
        Ms_Halo : Total stellar mass of all galaxies in a halo
                  (central + satellite)

    At HSC Observation side:
        M_Inn   : Mass within a smaller aperture
        M_Tot   : Mass within a larger aperture, as proxy of the observed
                  "total" stellar mass
        M_10    : Stellar mass within 10 kpc elliptical aperture
        M_100   : Stellar mass within 100 kpc elliptical aperture
        M_max   : Maximum stellar mass that can be recovered by HSC 1-D
                  profiles

    We assume:
        1) Mh_vir and Ms_halo has a log-log linear relation with small scatter
           **for central galaxy, at high-Mh_vir end**

           log(Mh_vir) = shmr_a x log(Ms_halo) + shmr_b

           shmr_a, shmr_b are free parameters of this model.
    """

    def __init__(self, **kwargs):
        """
        Initiate the model.

        Parameters:
        -----------
        """

        # Setup the configuration
        self.setupConfig(**kwargs)

        # Prepare the UniverseMachine mock catalog
        if self.um_cat is None:
            print('#-----------------------------------------------')
            print('#   Start to prepare the mock catalog ')
            self.um_mock = self.umPrepCatalog(self.um_dir,
                                              self.um_model,
                                              self.um_subvolumes,
                                              self.um_lbox,
                                              self.um_min_mvir)
        else:
            print('#-----------------------------------------------')
            print('#   Read in prepared mock catalog ')
            cat_prepared = os.path.join(self.um_dir, self.um_cat)
            msg = ('!! Cat not find the prepared '
                   'value-added UM mock catalog: %s' % cat_prepared)
            assert os.path.isfile(cat_prepared), msg
            self.um_mock = Table.read(cat_prepared, format='fits')

        # Prepare the pre-computed weak lensing pairs from simulation
        if self.um_wlcat is None:
            print('#-----------------------------------------------')
            print('#   Start to pre-compute the lensing pairs ')
            self.um_mass_encl = self.umPreComputeWL(
                self.sim_halocat,
                um_min_mvir=self.um_min_mvir,
                wl_min_r=self.um_wl_minr,
                wl_max_r=self.um_wl_maxr,
                wl_n_bins=self.um_wl_nbin,
                particle_data=self.sim_particle_data
                )
        else:
            print('#-----------------------------------------------')
            print('#   Read in the pre-compute lensing pairs ')
            um_wlcat = os.path.join(self.um_dir, self.um_wlcat)
            msg = ('!! Cat not find the pre-computed '
                   'weak lensing pairs data: %s' % um_wlcat)
            assert os.path.isfile(um_wlcat), msg
            self.um_mass_encl = np.load(um_wlcat)['data']

        # The halo catalog and the precomputed lensing pairs should have
        # the same size
        msg = ('!! The mock catalog and the pre-computed lensing ',
               'pairs should have the same size! %d v.s %d' % (
                   len(self.um_mock), len(self.um_mass_encl)
               ))

        # Mask for central galaxies
        self.mask_central = self.um_mock['mask_central']

        # Load stellar masses
        self.obs_logms_inn, self.obs_logms_tot = self.loadObsMass()

        # Load the stellar mass functions
        #  Inner mass
        if self.obs_smf_inn is None:
            self.obs_smf_inn = self.computeSMF(self.obs_logms_inn,
                                               self.obs_volume,
                                               self.obs_smf_inn_nbin,
                                               self.obs_smf_inn_min,
                                               self.obs_smf_inn_max,
                                               add_err=self.obs_smf_adderr,
                                               n_boots=10000)

        #  Total mass
        if self.obs_smf_tot is None:
            self.obs_smf_tot = self.computeSMF(self.obs_logms_tot,
                                               self.obs_volume,
                                               self.obs_smf_tot_nbin,
                                               self.obs_smf_tot_min,
                                               self.obs_smf_tot_max,
                                               add_err=self.obs_smf_adderr,
                                               n_boots=10000)

        # Full SMF as reference
        if self.obs_smf_full is None:
            # TODO: place holder
            self.obs_smf_full = None

        # Load observed weak lensing profiles
        self.obs_wl_profs = self.loadObsWL()

    def setupConfig(self, verbose=False, **kwargs):
        """
        Setup the configuration of the model.

        Kwargs:
        -----------
        Default configuration of the model.

        cosmo_h0 : float
            Reduced Hubble constant, to setup cosmology.

        cosmo_omega_m : float
            Omega_m, to setup cosmology.

        um_dir : string
            Location of the umdir model.

        um_model : string
            Model name for UM mock catalog.

        um_lbox : int or float
            Size of the simulation box, in unit of Mpc

        um_subvolumes : int
            Number of subvolumes in UniverseMachine mock.

        um_min_mvir : float
            Minimum halo mass used in the model.

        um_cat : string
            Prepared value-added catalog for UM mocks.

        um_wlcat : string
            Pre-computed weak lensing paris using simulation.

        sim_name : string
            Name of the simulation.

        sim_redshift : float or int
            Redshift of the simulation.

        wl_min_r : float
            Minimum radius for weak lensing analysis.

        wl_max_r : float
            Maximum radius for weak lensing analysis.

        wl_n_bins : int
            Number of bins in log(R) space for weak lensing analysis.

        obs_dir : string
            Directory for the observed data.

        obs_cat : string
            File of the observed stellar mass catalog.

        obs_area : float
            Effective area of the data in unit of square degree.

        obs_zmin : float
            Minimum redshift of the obseved sample.

        obs_zmax : float
            Maximum redshift of the observed sample.

        obs_mass : astropy.table
            Astropy table for the observed stellar masses.

        obs_wl_sample : string
            Sample name for the obseved weak lensing profiles.

        obs_smf_inn : string
            File for pre-computed stellar mass function of the inner mass.

        obs_smf_tot : string
            File for pre-computed stellar mass function of the total mass.

        obs_smf_full : string
            File for pre-computed stellar mass function extended to low
            stellar mass.

        obs_minn_col : string
            Column name for the inner stellar mass.

        obs_mtot_col : string
            Column name for the total stellar mass.

        obs_smf_inn_min : float
            Minimum inner stellar mass for SMF.

        obs_smf_inn_max : float
            Maximum inner stellar mass for SMF.

        obs_smf_inn_nbin : float
            Number of mass bins for SMF of inner stellar mass.

        obs_smf_tot_min : float
            Minimum total stellar mass for SMF.

        obs_smf_tot_max : float
            Maximum toter stellar mass for SMF.

        obs_smf_tot_nbin : float
            Number of mass bins for SMF of toter stellar mass.

        obs_smf_adderr : float
            Additional error for SMF.
        """

        # ---------------- Cosmology Related ----------------- #
        if ('cosmos_h0' in kwargs.keys()):
            self.h0 = kwargs['cosmo_h0']
        else:
            self.h0 = 0.70
        self.H = (self.h0 * 100)

        if ('cosmo_omega_m' in kwargs.keys()):
            self.omega_m = kwargs['cosmo_omega_m']
        else:
            self.omega_m = 0.307

        self.cosmo = FlatLambdaCDM(H0=self.H,
                                   Om0=self.omega_m)
        # --------------------------------------------------- #

        # -------------- Observed Data Related -------------- #
        if ('obs_dir' in kwargs.keys()):
            self.obs_dir = kwargs['obs_dir']
        else:
            self.obs_dir = 'Data'

        if ('obs_cat' in kwargs.keys()):
            self.obs_cat = kwargs['obs_cat']
        else:
            self.obs_cat = 's16a_wide_massive_fastlane_short_170610.fits'
        if verbose:
            print("# Input stellar mass catalog: %s" %
                  self.obs_cat)

        self.obs_mass = Table.read(
            os.path.join(self.obs_dir, self.obs_cat)
            )

        if ('obs_wl_sample' in kwargs.keys()):
            self.obs_wl_sample = kwargs['obs_wl_sample']
        else:
            self.obs_wl_sample = 's16a_massive_m100_11.6'
        if verbose:
            print("# Input weak lensing profile sample: %s" %
                  self.obs_wl_sample)

        self.obs_wl_dir = os.path.join(self.obs_dir,
                                       self.obs_wl_sample)

        self.obs_wl_bin = Table.read(
            os.path.join(self.obs_wl_dir, (self.obs_wl_sample + '.txt')),
            format='ascii'
            )

        self.obs_wl_n_bin = len(self.obs_wl_bin)
        if verbose:
            if self.obs_wl_n_bin > 1:
                print("# There are %d weak lensing profiles in this sample" %
                      self.obs_wl_n_bin)
            else:
                print("# There is 1 weak lensing profile in this sample")

        if ('obs_wl_calib' in kwargs.keys()):
            self.obs_wl_calib = kwargs['obs_wl_calib']
        else:
            self.obs_wl_calib = (2.0 * 0.87)    # S16A; from Alexie

        if ('obs_smf_inn' in kwargs.keys()):
            self.obs_smf_inn = Table.read(
                os.path.join(self.obs_dir, kwargs['obs_smf_inn']),
                format='fits'
            )
            if verbose:
                print("# Pre-computed SMF for inner logMs: %s" %
                      kwargs['obs_smf_inn'])
        else:
            self.obs_smf_inn = None

        if ('obs_smf_tot' in kwargs.keys()):
            self.obs_smf_tot = Table.read(
                os.path.join(self.obs_dir, kwargs['obs_smf_tot']),
                format='fits'
            )
            if verbose:
                print("# Pre-computed SMF for total logMs: %s" %
                      kwargs['obs_smf_tot'])
        else:
            self.obs_smf_tot = None

        if ('obs_smf_full' in kwargs.keys()):
            self.obs_smf_full = Table.read(
                os.path.join(self.obs_dir, kwargs['obs_smf_full']),
                format='fits'
            )
            if verbose:
                print("# Pre-computed full SMF: %s" %
                      kwargs['obs_smf_full'])
        else:
            self.obs_smf_full = None

        if ('obs_area' in kwargs.keys()):
            self.obs_area = kwargs['obs_area']
        else:
            self.obs_area = 144.6

        if ('obs_zmin' in kwargs.keys()):
            self.obs_zmin = kwargs['zmin']
        else:
            self.obs_zmin = 0.29

        if ('obs_zmax' in kwargs.keys()):
            self.obs_zmax = kwargs['zmax']
        else:
            self.obs_zmax = 0.51

        self.obs_volume = ((self.cosmo.comoving_volume(self.obs_zmax) -
                            self.cosmo.comoving_volume(self.obs_zmin)) *
                           (self.obs_area / 41254.0)).value
        if verbose:
            print("# The volume of the HSC data is %15.2f Mpc^3" %
                  self.obs_volume)

        if ('obs_smf_adderr' in kwargs.keys()):
            self.obs_smf_adderr = kwargs['obs_smf_adderr']
        else:
            self.obs_smf_adderr = 0.1

        if self.obs_smf_inn is not None:
            self.obs_smf_inn_min = np.nanmin(self.obs_smf_inn['logm_0'])
        else:
            if ('obs_smf_inn_min' in kwargs.keys()):
                self.obs_smf_inn_min = kwargs['obs_smf_inn_min']
            else:
                self.obs_smf_inn_min = 10.8

        if self.obs_smf_inn is not None:
            self.obs_smf_inn_max = np.nanmax(self.obs_smf_inn['logm_1'])
        else:
            if ('obs_smf_inn_max' in kwargs.keys()):
                self.obs_smf_inn_max = kwargs['obs_smf_inn_max']
            else:
                self.obs_smf_inn_max = 11.9

        if self.obs_smf_inn is not None:
            self.obs_smf_inn_nbin = len(self.obs_smf_inn)
        else:
            if ('obs_smf_inn_nbin' in kwargs.keys()):
                self.obs_smf_inn_nbin = kwargs['obs_smf_inn_nbin']
            else:
                self.obs_smf_inn_nbin = 11

        if self.obs_smf_tot is not None:
            self.obs_smf_tot_min = np.nanmin(self.obs_smf_tot['logm_0'])
        else:
            if ('obs_smf_tot_min' in kwargs.keys()):
                self.obs_smf_tot_min = kwargs['obs_smf_tot_min']
            else:
                self.obs_smf_tot_min = 11.6

        if self.obs_smf_tot is not None:
            self.obs_smf_tot_max = np.nanmax(self.obs_smf_tot['logm_1'])
        else:
            if ('obs_smf_tot_max' in kwargs.keys()):
                self.obs_smf_tot_max = kwargs['obs_smf_tot_max']
            else:
                self.obs_smf_tot_max = 12.2

        if self.obs_smf_tot is not None:
            self.obs_smf_tot_nbin = len(self.obs_smf_tot)
        else:
            if ('obs_smf_tot_nbin' in kwargs.keys()):
                self.obs_smf_tot_nbin = kwargs['obs_smf_tot_nbin']
            else:
                self.obs_smf_tot_nbin = 8

        if ('obs_minn_col' in kwargs.keys()):
            self.obs_minn_col = kwargs['obs_minn_col']
        else:
            self.obs_minn_col = 'logm_10'

        if ('obs_mtot_col' in kwargs.keys()):
            self.obs_mtot_col = kwargs['obs_mtot_col']
        else:
            self.obs_mtot_col = 'logm_100'

        if verbose:
            print('# Using %s as proxy of inner stellar mass.' %
                  self.obs_minn_col)
            print('# Using %s as proxy of total stellar mass.' %
                  self.obs_mtot_col)
            print("# For inner stellar mass: ")
            print("    %d bins at %5.2f < logMinn < %5.2f" %
                  (self.obs_smf_inn_nbin, self.obs_smf_inn_min,
                   self.obs_smf_inn_max))
            print("# For total stellar mass: ")
            print("    %d bins at %5.2f < logMtot < %5.2f" %
                  (self.obs_smf_tot_nbin, self.obs_smf_tot_min,
                   self.obs_smf_tot_max))
        # --------------------------------------------------- #

        # ---------- UniverseMachine Mock Related ----------- #
        if ('um_dir' in kwargs.keys()):
            self.um_dir = kwargs['um_dir']
        else:
            self.um_dir = 'Data'

        if ('um_model' in kwargs.keys()):
            self.um_model = kwargs['um_model']
        else:
            self.um_model = 'um2_mdpl2/obs_sm9p75_to_um_mock_z0p3.npy'
            # UM1 self.um_model = 'a_1.002310'

        if ('um_lbox' in kwargs.keys()):
            self.um_lbox = kwargs['um_lbox']
        else:
            self.um_lbox = 1000.0  # Mpc/h
            # UM1 : self.um_lbox = 250.0  # Mpc/h

        self.um_volume = np.power(self.um_lbox / self.h0, 3)
        if verbose:
            print("# The volume of the UniverseMachine mock is %15.2f Mpc^3" %
                  self.um_volume)

        if ('um_subvolumes' in kwargs.keys()):
            self.um_subvolumes = kwargs['um_subvolumes']
        else:
            # Only useful for UM1 catalog
            self.um_subvolumes = 144

        if ('um_min_mvir' in kwargs.keys()):
            self.um_min_mvir = kwargs['um_min_mvir']
        else:
            self.um_min_mvir = 12.5

        if ('um_cat' in kwargs.keys()):
            self.um_cat = kwargs['um_cat']
        else:
            self.um_cat = None

        if ('um_wlcat' in kwargs.keys()):
            self.um_wlcat = kwargs['um_wlcat']
        else:
            self.um_wlcat = None

        if ('um_redshift' in kwargs.keys()):
            self.um_redshift = kwargs['um_redshift']
        else:
            self.um_redshift = 0.3637

        if ('um_wl_minr' in kwargs.keys()):
            self.um_wl_minr = kwargs['um_wl_minr']
        else:
            self.um_wl_minr = 0.02

        if ('um_wl_maxr' in kwargs.keys()):
            self.um_wl_maxr = kwargs['um_wl_maxr']
        else:
            self.um_wl_maxr = 50.0

        if ('um_wl_nbin' in kwargs.keys()):
            self.um_wl_nbin = kwargs['um_wl_nbin']
        else:
            self.um_wl_nbin = 22

        if ('um_mtot_nbin' in kwargs.keys()):
            self.um_mtot_nbin = kwargs['um_mtot_nbin']
        else:
            self.um_mtot_nbin = 40

        if ('um_ngal_bin_min' in kwargs.keys()):
            self.um_ngal_bin_min = kwargs['um_ngal_bin_min']
        else:
            self.um_ngal_bin_min = 25

        if ('um_min_scatter' in kwargs.keys()):
            self.um_min_scatter = kwargs['um_min_scatter']
        else:
            self.um_min_scatter = 0.05
        # --------------------------------------------------- #

        # ----------------- Simulation Related -------------- #
        if ('sim_name' in kwargs.keys()):
            self.sim_name = kwargs['sim_name']
        else:
            self.sim_name = 'multidark'
            # UM1 self.sim_name = 'bolplanck'

        if ('sim_redshift' in kwargs.keys()):
            self.sim_redshift = kwargs['sim_redshift']
        else:
            self.sim_redshift = 0.0

        self.sim_halocat = CachedHaloCatalog(simname=self.sim_name,
                                             redshift=self.sim_redshift)

        if ('sim_particle_catalog' in kwargs.keys()):
            self.sim_particle_catalog = kwargs['sim_particle_catalog']
            self.sim_particle_data = Table.read(self.sim_particle_catalog)
        else:
            self.sim_particle_catalog = None
            self.sim_particle_data = None

        if ('sim_particle_mass' in kwargs.keys()):
            self.sim_particle_mass = kwargs['sim_particle_mass']
        else:
            self.sim_particle_mass = None

        if ('sim_num_ptcl_per_dim' in kwargs.keys()):
            self.sim_num_ptcl_per_dim = kwargs['sim_num_ptcl_per_dim']
        else:
            self.sim_num_ptcl_per_dim = None

        if ('sim_cosmo' in kwargs.keys()):
            self.sim_cosmo = kwargs['sim_cosmo']
        else:
            self.sim_cosmo = None
        # --------------------------------------------------- #

        # --------------- Weak Lensing Related -------------- #
        if ('wl_min_r' in kwargs.keys()):
            self.wl_min_r = kwargs['wl_min_r']
        else:
            self.wl_min_r = 0.05

        if ('wl_max_r' in kwargs.keys()):
            self.wl_max_r = kwargs['wl_max_r']
        else:
            self.wl_max_r = 50.0

        if ('wl_n_bins' in kwargs.keys()):
            self.wl_n_bins = kwargs['wl_n_bins']
        else:
            self.wl_n_bins = 15

        self.wl_rp_bins = np.logspace(np.log10(self.wl_min_r),
                                      np.log10(self.wl_max_r),
                                      self.wl_n_bins)
        # --------------------------------------------------- #

        # --------------- Plotting Related ------------------ #
        if ('obs_mtot_color' in kwargs.keys()):
            self.obs_mtot_color = kwargs['obs_mtot_color']
        else:
            self.obs_mtot_color = 'steelblue'

        if ('obs_minn_color' in kwargs.keys()):
            self.obs_minn_color = kwargs['obs_minn_color']
        else:
            self.obs_minn_color = 'salmon'

        if ('um_mtot_color' in kwargs.keys()):
            self.um_mtot_color = kwargs['um_mtot_color']
        else:
            self.um_mtot_color = 'royalblue'

        if ('um_minn_color' in kwargs.keys()):
            self.um_minn_color = kwargs['um_minn_color']
        else:
            self.um_minn_color = 'maroon'

        if ('obs_mark' in kwargs.keys()):
            self.obs_mark = kwargs['obs_mark']
        else:
            self.obs_mark = 'o'

        if ('um_mark' in kwargs.keys()):
            self.um_mark = kwargs['um_mark']
        else:
            self.um_mark = 'h'
        # --------------------------------------------------- #

        # ------------------- MCMC Related ------------------ #
        if ('mcmc_nsamples' in kwargs.keys()):
            self.mcmc_nsamples = kwargs['mcmc_nsamples']
        else:
            self.mcmc_nsamples = 200

        if ('mcmc_nthreads' in kwargs.keys()):
            self.mcmc_nthreads = kwargs['mcmc_nthreads']
        else:
            self.mcmc_nthreads = 2

        if ('mcmc_nburnin' in kwargs.keys()):
            self.mcmc_nburnin = kwargs['mcmc_nburnin']
        else:
            self.mcmc_nburnin = 100

        if ('mcmc_nwalkers' in kwargs.keys()):
            self.mcmc_nwalkers = kwargs['mcmc_nwalkers']
        else:
            self.mcmc_nwalkers = 100

        if ('mcmc_smf_only' in kwargs.keys()):
            self.mcmc_smf_only = kwargs['mcmc_smf_only']
        else:
            self.mcmc_smf_only = False

        if ('mcmc_wl_only' in kwargs.keys()):
            self.mcmc_wl_only = kwargs['mcmc_wl_only']
        else:
            self.mcmc_wl_only = False

        if ('mcmc_wl_weight' in kwargs.keys()):
            self.mcmc_wl_weight = kwargs['mcmc_wl_weight']
        else:
            self.mcmc_wl_weight = 1.0

        if ('mcmc_labels' in kwargs.keys()):
            self.mcmc_labels = kwargs['mcmc_labels']
        else:
            self.mcmc_labels = [r'$a$', r'$b$', r'$c$', r'$d$']

        # Right now, the model only has four parameters
        self.mcmc_ndims = 4

        if ('mcmc_burnin_file' in kwargs.keys()):
            self.mcmc_burnin_file = kwargs['mcmc_burnin_file']
        else:
            self.mcmc_burnin_file = 'um2_m100_m10_burnin_result.pkl'

        if ('mcmc_run_file' in kwargs.keys()):
            self.mcmc_run_file = kwargs['mcmc_run_file']
        else:
            self.mcmc_run_file = 'um2_m100_m10_run_result.pkl'

        if ('mcmc_burnin_chain_file' in kwargs.keys()):
            self.mcmc_burnin_chain_file = kwargs['mcmc_burnin_chain_file']
        else:
            self.mcmc_burnin_chain_file = 'um2_m100_m10_burnin_chain.pkl'

        if ('mcmc_run_chain_file' in kwargs.keys()):
            self.mcmc_run_chain_file = kwargs['mcmc_run_chain_file']
        else:
            self.mcmc_run_chain_file = 'um2_m100_m10_run_chain.pkl'

        if ('mcmc_run_samples_file' in kwargs.keys()):
            self.mcmc_run_samples_file = kwargs['mcmc_run_samples_file']
        else:
            self.mcmc_run_samples_file = 'um2_m100_m10_run_samples.npz'

        # Priors
        if ('shmr_a_ini' in kwargs.keys()):
            self.shmr_a_ini = kwargs['shmr_a_ini']
        else:
            self.shmr_a_ini = 0.600

        if ('shmr_a_low' in kwargs.keys()):
            self.shmr_a_low = kwargs['shmr_a_low']
        else:
            self.shmr_a_low = 0.250

        if ('shmr_a_upp' in kwargs.keys()):
            self.shmr_a_upp = kwargs['shmr_a_upp']
        else:
            self.shmr_a_upp = 1.000

        if ('shmr_b_ini' in kwargs.keys()):
            self.shmr_b_ini = kwargs['shmr_b_ini']
        else:
            self.shmr_b_ini = 3.60

        if ('shmr_b_low' in kwargs.keys()):
            self.shmr_b_low = kwargs['shmr_b_low']
        else:
            self.shmr_b_low = -1.50

        if ('shmr_b_upp' in kwargs.keys()):
            self.shmr_b_upp = kwargs['shmr_b_upp']
        else:
            self.shmr_b_upp = 8.00

        if ('sigms_a_ini' in kwargs.keys()):
            self.sigms_a_ini = kwargs['sigms_a_ini']
        else:
            self.sigms_a_ini = -0.06

        if ('sigms_a_low' in kwargs.keys()):
            self.sigms_a_low = kwargs['sigms_a_low']
        else:
            self.sigms_a_low = -0.20

        if ('sigms_a_upp' in kwargs.keys()):
            self.sigms_a_upp = kwargs['sigms_a_upp']
        else:
            self.sigms_a_upp = 0.00

        if ('sigms_b_ini' in kwargs.keys()):
            self.sigms_b_ini = kwargs['sigms_b_ini']
        else:
            self.sigms_b_ini = 1.00

        if ('sigms_b_low' in kwargs.keys()):
            self.sigms_b_low = kwargs['sigms_b_low']
        else:
            self.sigms_b_low = 0.00

        if ('sigms_b_upp' in kwargs.keys()):
            self.sigms_b_upp = kwargs['sigms_b_upp']
        else:
            self.sigms_b_upp = 1.60

        self.theta_start = (self.shmr_a_ini, self.shmr_b_ini,
                            self.sigms_a_ini, self.sigms_b_ini)
        # --------------------------------------------------- #

        return None

    def loadObsMass(self):
        """
        Load the observed stellar mass data.

        Return the arrays of observed inner and total masses.
        """
        logms_tot_obs = self.obs_mass[self.obs_mtot_col]
        logms_inn_obs = self.obs_mass[self.obs_minn_col]

        if self.obs_smf_tot_min is not None:
            mask_mtot_obs = (logms_tot_obs >= self.obs_smf_tot_min)
            logms_tot_obs = logms_tot_obs[mask_mtot_obs]
            logms_inn_obs = logms_inn_obs[mask_mtot_obs]

        return (np.asarray(logms_inn_obs),
                np.asarray(logms_tot_obs))

    def computeSMF(self, logms, volume, nbin, min_logms, max_logms,
                   add_err=None, bootstrap=True, n_boots=5000):
        """
        Estimate the observed SMF and bootstrap errors.

        Parameters:
        -----------

        logms : ndarray
            Log10 stellar mass.

        volume : float
            The volume of the data, in unit of Mpc^3.

        nbin : int
            Number of bins in log10 stellar mass.

        min_logms : float
            Minimum stellar mass.

        max_logms : float
            Maximum stellar mass.

        add_err : float, optional
            Additional error to be added to the SMF.
            e.g. 0.1 == 10%
            Default: None

        bootstrap : bool, optional
            Use bootstrap resampling to measure the error of SMF.
            Default: True

        n_boots : int, optional
            Number of bootstrap resamplings.
            Default: 5000
        """
        if not bootstrap:
            smf_single = compute_smf(logms, volume, nbin,
                                     min_logms, max_logms,
                                     return_bins=True)
            mass_cen, smf, smf_err, mass_bins = smf_single
            smf_low = (smf - smf_err)
            smf_upp = (smf + smf_err)
        else:
            smf_boot = bootstrap_smf(logms, volume, nbin,
                                     min_logms, max_logms,
                                     n_boots=n_boots)
            mass_cen, smf_s, smf_err, smf_b, mass_bins = smf_boot

            smf = np.nanmedian(smf_b, axis=0)
            smf_low = np.nanpercentile(smf_b, 16, axis=0,
                                       interpolation='midpoint')
            smf_upp = np.nanpercentile(smf_b, 84, axis=0,
                                       interpolation='midpoint')

        if add_err is not None:
            smf_err += (smf * add_err)
            smf_low -= (smf * add_err)
            smf_upp += (smf * add_err)

        # Left and right edges of the mass bins
        bins_0 = mass_bins[0:-1]
        bins_1 = mass_bins[1:]

        smf_table = Table()
        smf_table['logm_mean'] = mass_cen
        smf_table['logm_0'] = bins_0
        smf_table['logm_1'] = bins_1
        smf_table['smf'] = smf
        smf_table['smf_err'] = smf_err
        smf_table['smf_low'] = smf_low
        smf_table['smf_upp'] = smf_upp

        return smf_table

    def loadObsWL(self):
        """
        Load the observed WL profiles.
        """
        wl_profiles = []
        for wl_bin in self.obs_wl_bin:
            idx = str(wl_bin['BIN_ID']).strip()
            wl_file = os.path.join(
                self.obs_wl_dir, self.obs_wl_sample + '_' + idx + '.asc'
            )
            msg = ("# Cannot find the WL profile "
                   "for bin %s : %s" % (idx, wl_file))
            assert os.path.isfile(wl_file), msg
            wl_box = boxBinWL(wl_file)
            # Including the WL calibration error
            wl_box.sig /= self.obs_wl_calib

            wl_box.setBinId(wl_bin['BIN_ID'])
            wl_box.setMassLimits(wl_bin['LOW_M100'], wl_bin['UPP_M100'],
                                 wl_bin['LOW_M10'], wl_bin['UPP_M10'])
            wl_profiles.append(wl_box)

        return wl_profiles

    def umGetWLProf(self, mask, mock_use=None, mass_encl_use=None,
                    verbose=False, r_interp=None, mstar_lin=None):
        """
        Predict weak lensing mass profiles using pre-computed pairs.

        Parameters:
        -----------

        mask : ndarray
            Mask array that defines the subsample.

        mock_use : astropy.table, optional
            The mock catalog used for computation, in case it is differnt
            from the default one.
            Default: None

        mass_encl_use : ndarray, optional
            The pre-computed enclosed mass array used for computation,
            in case it is different from the default one.
            Default: None

        r_interp : array, optional
            Radius array to interpolate to.
            Default: None

        mstar_lin : float, optional
            Linear mass for "point source".
            Default: None
        """
        if mock_use is None:
            mock_use = self.um_mock

        if mass_encl_use is None:
            mass_encl_use = self.um_mass_encl

        # Radius bins
        rp_bins = np.logspace(np.log10(self.um_wl_minr),
                              np.log10(self.um_wl_maxr),
                              self.um_wl_nbin)
        # Box size
        period = self.um_lbox
        subsample = mock_use[mask]

        if verbose:
            print("# Deal with %d galaxies in the subsample" % np.sum(mask))

        #  Use the mask to get subsample positions and pre-computed pairs
        subsample_positions = np.vstack([subsample['x'],
                                         subsample['y'],
                                         subsample['z']]).T

        subsample_mass_encl_precompute = mass_encl_use[mask, :]

        if self.sim_cosmo is None:
            sim_cosmo = self.sim_halocat.cosmology
        else:
            sim_cosmo = self.sim_cosmo

        rp_ht_units, ds_ht_units = delta_sigma_from_precomputed_pairs(
            subsample_positions,
            subsample_mass_encl_precompute,
            rp_bins, period,
            cosmology=self.sim_cosmo
        )

        # Unit conversion
        ds_phys_msun_pc2 = ((1. + self.um_redshift) ** 2 *
                            (ds_ht_units * sim_cosmo.h) /
                            (1e12))

        rp_phys = ((rp_ht_units) /
                   (abs(1. + self.um_redshift) *
                    sim_cosmo.h))

        # Add the point source term
        if mstar_lin is not None:
            ds_phys_msun_pc2 += (
                mstar_lin / 1e12 / (np.pi * (rp_phys ** 2.0))
                )

        if r_interp is not None:
            intrp = interpolate.interp1d(rp_phys, ds_phys_msun_pc2,
                                         kind='cubic', bounds_error=False)
            dsigma = intrp(r_interp)

            return (r_interp, dsigma)
        else:
            return (rp_phys, ds_phys_msun_pc2)

    def umPredictMass(self, shmr_a, shmr_b, sigms_a, sigms_b,
                      star_col='logms_tot',
                      halo_col='logmh_host',
                      constant_bin=False):
        """
        Predict M100, M10, Mtot using Mvir, M_gal, M_ICL.

        Parameters:
        -----------

        logms_obs_inn : ndarray
            Observed stellar mass within a smaller aperture.

        logms_obs_tot : ndarray
            Observed stellar mass witin a larger aperture

        shmr_a : float
            Slope of the logMs = a x logMh + b relation.

        shmr_b : float
            Normalization of the logMs = a x logMh + b relation.

        sigms_a : float
            Slope of the SigMs = a x logMh + b relation.

        sigms_b : float
            Normalization of the SigMs = a x logMh + b relation.

        halo_col : string
            Column for halo mass.

        star_col : string
            Column for stellar mass.

        constant_bin : boolen
            Whether to use constant bin size for logMs_tot or not.
        """
        # Make the model
        # ! Required by sm_profile_from_mhalo, this is not logMhalo
        mhalo = self.um_mock[halo_col]
        random_scatter_in_dex = self.mockGetSimpleMhScatter(
            mhalo, sigms_a, sigms_b
        )

        # Log M100 bin size
        # ! Binning will create artifical step-like feature on M100-M10 plane
        # ! Bin size should not be too large

        if constant_bin:
            # Constant log-linear bin size
            logms_tot_bins = np.linspace(self.obs_smf_tot_min,
                                         self.obs_smf_tot_max,
                                         self.um_mtot_nbin)

        else:
            # Try equal number object bin
            nobj_bin = np.ceil(len(self.obs_logms_tot) / self.um_mtot_nbin)
            nobj_bin = (nobj_bin if nobj_bin <= self.um_ngal_bin_min else
                        self.um_gal_bin_min)

            logms_tot_sort = np.sort(self.obs_logms_tot)
            logms_tot_bins = logms_tot_sort[
                np.where(np.arange(len(logms_tot_sort)) % nobj_bin == 0)]
            logms_tot_bins[-1] = mtot_sort[-1]

        # Fraction of 'sm' + 'icl' to the total stellar mass of the halo
        # (including satellites)
        self.frac_tot_by_halo = ((10.0 ** self.um_mock['logms_tot']) /
                                 (10.0 ** self.um_mock['logms_halo']))
        self.frac_inn_by_tot = ((10.0 ** self.um_mock['logms_gal']) /
                                (10.0 ** self.um_mock['logms_tot']))

        # This returns UM predicted m10, m100, smtot
        um_mass_model = sm_profile_from_mhalo(
            mhalo, shmr_a, shmr_b,
            random_scatter_in_dex,
            self.frac_tot_by_halo,
            self.frac_inn_by_tot,
            self.obs_logms_tot,
            self.obs_logms_inn
            logms_tot_bins,
            ngal_min=self.um_ngal_bin_min
        )

        return um_mass_model, mhalo

    def umPredictSMF(self, logms_mod_tot, logms_mod_inn):
        """
        Estimate SMFs of Minn and Mtot predicted by UM.
        """
        # SMF of the predicted Mtot (M1100)
        um_smf_tot = self.computeSMF(logms_mod_tot,
                                     self.um_volume,
                                     self.obs_smf_tot_nbin,
                                     self.obs_smf_tot_min,
                                     self.obs_smf_tot_max,
                                     bootstrap=False)

        # SMF of the predicted Minn (M10)
        um_smf_inn = self.computeSMF(logms_mod_inn,
                                     self.um_volume,
                                     self.obs_smf_inn_nbin,
                                     self.obs_smf_inn_min,
                                     self.obs_smf_inn_max,
                                     bootstrap=False)

        return (um_smf_tot, um_smf_inn)

    def umPredictWL(self, logms_mod_tot, logms_mod_inn, mask_mtot,
                    um_wl_min_ngal=2, return_mhalo=False,
                    verbose=False):
        """
        Predict the WL profiles to compare with observations.

        Parameters:
        -----------

        logms_mod_tot : ndarray
            Total stellar mass (e.g. M100) predicted by UM.

        logms_mod_inn : ndarray
            Inner stellar mass (e.g. M10) predicted by UM.

        mask_tot : bool array
            Mask for the input mock catalog and precomputed WL pairs.

        um_wl_min_ngal : int, optional
            Minimum requred galaxies in each bin to estimate WL profile.
        """
        # The mock catalog and precomputed mass files for subsamples
        mock_use = self.um_mock[mask_mtot]
        mass_encl_use = self.um_mass_encl[mask_mtot, :]

        # Radius bins
        rp_center = self.obs_wl_profs[0].r

        # Zeros arrary for empty sample
        sig_zeros = np.zeros(self.wl_n_bins)

        um_wl_profs = []
        um_wl_mhalo = []
        um_wl_mtot = []
        um_wl_minn = []
        for obs_prof in self.obs_wl_profs:
            bin_mask = ((logms_mod_tot >= obs_prof.low_mtot) &
                        (logms_mod_tot <= obs_prof.upp_mtot) &
                        (logms_mod_inn >= obs_prof.low_minn) &
                        (logms_mod_inn <= obs_prof.upp_mtot))

            # "Point source" term for the central galaxy
            mstar_lin = np.nanmedian(10.0 * logms_mod_tot[bin_mask])
            um_wl_mtot.append(np.nanmedian(logms_mod_tot[bin_mask]))
            um_wl_minn.append(np.nanmedian(logms_mod_inn[bin_mask]))
            um_wl_mhalo.append(
                np.nanmedian(mock_use[bin_mask]['logmh_vir'])
                )

            if (np.sum(bin_mask) <= um_wl_min_ngal):
                wl_prof = sig_zeros
                if verbose:
                    print("# Not enough UM galaxy "
                          "in bin %d !" % obs_prof.bin_id)
            else:
                wl_rp, wl_prof = self.umGetWLProf(bin_mask,
                                                  mock_use=mock_use,
                                                  mass_encl_use=mass_encl_use,
                                                  r_interp=rp_center,
                                                  mstar_lin=mstar_lin)

            um_wl_profs.append(wl_prof)

        if return_mhalo:
            return (um_wl_profs, um_wl_mhalo, um_wl_mtot, um_wl_minn)
        else:
            return um_wl_profs

    def umPredictModel(self, shmr_a, shmr_b, sigms_a, sigms_b,
                       plotSMF=False, plotWL=False,
                       return_wl_mhalo=False):
        """
        Given model parameters, returns the SMFs and WL profiles.

        Parameters:
        -----------

        shmr_a : float
            Slope of the logMs = a x logMh + b relation.

        shmr_b : float
            Normalization of the logMs = a x logMh + b relation.

        sigms_a : float
            Slope of the SigMs = a x logMh + b relation.

        sigms_b : float
            Normalization of the SigMs = a x logMh + b relation.

        plotSMF : bool, optional
            Show the comparison of SMF.
            Default: False

        plotWL : bool, optional
            Show the comparisons of WL.
            Default: False

        return_wl_mhalo : bool, optional
            Return the median halo mass in each bin.
            Default: False
        """
        # Predict stellar mass
        (um_mass_model, mhalo) = self.umPredictMass(shmr_a, shmr_b,
                                                    sigms_a, sigms_b)

        (logms_mod_inn, logms_mod_tot_all,
         logms_mod_halo_all, mask_mtot) = um_mass_model

        logms_mod_tot = logms_mod_tot_all[mask_mtot]

        # Predict the SMFs
        um_smf_tot, um_smf_inn = self.umPredictSMF(logms_mod_tot,
                                                   logms_mod_inn)
        # TODO: If one mass bin is empty, set the error to a higer value
        # mask_zero = um_smf_tot['smf'] <= 1.0E-10
        # um_smf_tot['smf'][mask_zero] = np.nan
        # um_smf_tot['smf_err'][mask_zero] = np.nan

        # Predict the WL signal
        if return_wl_mhalo:
            um_wl_predict = self.umPredictWL(logms_mod_tot,
                                             logms_mod_inn,
                                             mask_mtot,
                                             return_mhalo=True)
            (um_wl_profs, um_wl_mhalo,
             um_wl_mtot, um_wl_minn) = um_wl_predict
        else:
            um_wl_profs = self.umPredictWL(logms_mod_tot,
                                           logms_mod_inn,
                                           mask_mtot)

        if plotSMF:
            self.umPlotSMF(um_smf_tot, um_smf_inn,
                           logms_mod_tot, logms_mod_inn)

        if return_wl_mhalo:
            return (um_smf_tot, um_smf_inn, um_wl_profs,
                    logms_mod_inn, logms_mod_tot_all,
                    logms_mod_halo_all, mask_mtot,
                    um_wl_mhalo, mhalo)
        else:
            return (um_smf_tot, um_smf_inn, um_wl_profs,
                    logms_mod_inn, logms_mod_tot_all,
                    logms_mod_halo_all, mask_mtot)

    def showObsSMF(self, **kwargs):
        """
        Show a summary plot of the input data.
        """
        fig = plt.figure(figsize=(9, 6))

        if self.obs_smf_inn is not None:
            plt.fill_between(self.obs_smf_inn['logm_mean'],
                             np.log10(self.obs_smf_inn['smf_low']),
                             np.log10(self.obs_smf_inn['smf_upp']),
                             facecolor='lightsalmon',
                             edgecolor='none',
                             interpolate=True)
            plt.plot(self.obs_smf_inn['logm_mean'],
                     np.log10(self.obs_smf_inn['smf']),
                     label=r'$\mathrm{M_inn}$',
                     linewidth=2, c='orangered')

        if self.obs_smf_tot is not None:
            plt.fill_between(self.obs_smf_tot['logm_mean'],
                             np.log10(self.obs_smf_tot['smf_low']),
                             np.log10(self.obs_smf_tot['smf_upp']),
                             facecolor='grey',
                             edgecolor='none',
                             interpolate=True)
            plt.plot(self.obs_smf_tot['logm_mean'],
                     np.log10(self.obs_smf_tot['smf']),
                     label=r'$\mathrm{M_tot}$',
                     linewidth=2, c='black')

        plt.xlabel(r'$\log (M_{\star}/M_{\odot})$', fontsize=20)
        plt.ylabel((r'$\mathrm{d}N/\mathrm{d}\log M_{\star}\ $'
                    r'$[{\mathrm{Mpc}^{-3}}{\mathrm{dex}^{-1}}]$'),
                   size=20)

        return fig

    def umPreComputeWL(self, sim_halocat, um_min_mvir=None,
                       wl_min_r=0.1, wl_max_r=40.0, wl_n_bins=11,
                       verbose=True, particle_data=None):
        """
        Precompute lensing pairs using UM mock catalog.

        Parameters:
        -----------

        sim_name : string
            Name of the simulation used in this model.

        sim_redshift : float
            Redshift of the simulation.

        um_min_mvir : float, optional
            Minimum halo mass used in computation.
            Default: None

        wl_min_r : float, optional
            Minimum radius for WL measurement.
            Default: 0.1

        wl_max_r : float, optional
            Maximum radius for WL measurement.
            Default: 40.0

        wl_n_bins : int, optional
            Number of bins in log(R) space.
            Default: 11

        particle_data : astropy.table, optional
            External particle data catalog.
        """
        sim_halocat = self.sim_halocat

        if particle_data is not None:
            print("# Using external partical table")
            sim_particles = particle_data
        else:
            sim_particles = sim_halocat.ptcl_table

        if self.sim_particle_mass is not None:
            sim_particle_masses = self.sim_particle_mass
        else:
            sim_particle_masses = sim_halocat.particle_mass
        print("#   The simulation particle mass is %f" %
              sim_particle_masses)

        if self.sim_num_ptcl_per_dim is not None:
            sim_total_num_ptcl_in_snapshot = (
                self.sim_num_ptcl_per_dim ** 3
                )
        else:
            sim_total_num_ptcl_in_snapshot = (
                sim_halocat.num_ptcl_per_dim ** 3
                )
        print("#   The number of particles is %d" %
              sim_total_num_ptcl_in_snapshot)

        sim_downsampling_factor = (sim_total_num_ptcl_in_snapshot /
                                   float(len(sim_particles)))

        if um_min_mvir is not None:
            sample = self.um_mock[self.um_mock['logmh_peak'] >= um_min_mvir]
        else:
            sample = self.um_mock

        # Radius bins
        rp_bins = np.logspace(np.log10(wl_min_r),
                              np.log10(wl_max_r),
                              wl_n_bins)
        # Box size
        sim_period = self.um_lbox

        start = time()
        sim_mass_encl = precompute_lensing_pairs(
            sample['x'], sample['y'], sample['z'],
            sim_particles['x'], sim_particles['y'], sim_particles['z'],
            sim_particle_masses, sim_downsampling_factor,
            rp_bins, sim_period)
        end = time()
        runtime = (end - start)

        msg = ("Total runtime for {0} galaxies and {1:.1e} particles "
               "={2:.2f} seconds")
        print(msg.format(len(sample), len(sim_particles), runtime))

        return sim_mass_encl

    def umPrepCatalog(self, um_dir, um_model, um_subvolumes,
                      um_lbox, um_min_mvir):
        """
        Prepare the UniverseMachine mock catalog.

        The goal is to prepare a FITS catalog that include all
        necessary information.
        During the modeling part, we just need to load this catalog once.

        Parameters:
        -----------

        um_dir : string
            Directory of the UM model.

        um_model : string
            Name of the UM model catalog.

        um_subvolumnes : int
            Number of subvolumes of the UM model.

        um_lbox : float
            Size of the UM simulation box size.

        um_min_mvir : float
            Minimum halo mass used here.
        """
        # properties used in the model
        MOCK_PROPERTIES = list(('sm', 'sfr', 'obs_sm', 'obs_sfr', 'icl',
                                'mvir', 'halo_id', 'upid', 'mpeak',
                                'x', 'y', 'z'))
        # Load the UM mock catalog
        cat_dir = os.path.join(um_dir, um_model)

        um_model_pre, um_model_ext = os.path.splitext(cat_dir)
        if um_model_ext == '.npy':
            um_mock = value_added_mdpl2_mock(cat_dir)
        else:
            um_mock = load_mock_from_binaries(
                galprops=MOCK_PROPERTIES,
                root_dirname=cat_dir,
                subvolumes=np.arange(um_subvolumes)
            )
            # Value-added the mock catalog
            um_mock = value_added_mock(um_mock, um_lbox)

        # Sort the catalog based on the host halo ID
        um_mock.sort('halo_hostid')

        # Make a mask for central galaxies
        mask_central = um_mock['upid'] == -1
        um_mock.add_column(Column(data=mask_central,
                                  name='mask_central'))

        # Add a column as the BCG+ICL mass
        um_mock.add_column(Column(data=(um_mock['sm'] + um_mock['icl']),
                                  name='mtot_galaxy'))

        # Total stellar masses within a halo, including the satellites
        mstar_mhalo = total_stellar_mass_including_satellites(um_mock,
                                                              'mtot_galaxy')
        um_mock.add_column(Column(data=mstar_mhalo,
                                  name='mstar_mhalo'))

        # Add log10(Mass)
        # Stellar mass
        um_mock.add_column(Column(data=np.log10(um_mock['sm']),
                                  name='logms_gal'))
        um_mock.add_column(Column(data=np.log10(um_mock['icl']),
                                  name='logms_icl'))
        um_mock.add_column(Column(data=np.log10(um_mock['mtot_galaxy']),
                                  name='logms_tot'))
        um_mock.add_column(Column(data=np.log10(um_mock['mstar_mhalo']),
                                  name='logms_halo'))
        # Halo mass
        um_mock.add_column(Column(data=np.log10(um_mock['mvir']),
                                  name='logmh_vir'))
        um_mock.add_column(Column(data=np.log10(um_mock['mpeak']),
                                  name='logmh_peak'))

        # The 'mvir' in the mock catalog is not the halo mass of parent halo
        # Get the host halo mass of each galaxy
        idxA, idxB = crossmatch(um_mock['halo_hostid'],
                                um_mock['halo_id'])

        mhalo_host = copy.deepcopy(um_mock['mvir'].data)
        mhalo_host[idxA] = um_mock[idxB]['mvir']

        logmh_host = copy.deepcopy(um_mock['logmh_vir'].data)
        logmh_host[idxA] = um_mock[idxB]['logmh_vir']

        # Notice: There are ~500 satellites without host halo
        #         And there are ~4000 satellits without matched centrals
        #         Right now, we just use their 'mvir' as host halo mass
        um_mock.add_column(Column(data=mhalo_host,
                                  name='mhalo_host'))
        um_mock.add_column(Column(data=logmh_host,
                                  name='logmh_host'))

        if um_min_mvir is not None:
            um_mock = um_mock[um_mock['logmh_peak'] >= um_min_mvir]

        return um_mock

    def mockFitSimpleSHMR(self, um_min_mvir=12.0, verbose=True,
                          halo_col='logmh_vir', star_col='logms_halo'):
        """
        Fit a log-log linear relation between halo and stellar mass:
            logMh_vir = a x logMs + b to the UM

        Parameters:
        -----------

        um_min_mvir : float, optional
            Minimum halo mass used in the fitting.
            Default : None

        halo_col : string
            Column for halo mass.

        star_col : string
            Column for stellar mass.
        """

        # Fit Mhalo-Mstar log-log linear relation
        mock_central = self.um_mock[self.um_mock['mask_central']]
        log_mh = mock_central[halo_col]
        log_ms = mock_central[star_col]

        # Only fit for halos more massive than certain value
        mask_massive = (mock_central['logmh_peak'] > um_min_mvir)
        if verbose:
            print("# Use %d galaxies for the fitting..." %
                  np.sum(mask_massive))

        # Using Numpy Polyfit is much faster
        # Log-Log Linear relation
        shmr_a, shmr_b = np.polyfit(
            log_mh[mask_massive],
            log_ms[mask_massive],
            1)
        if verbose:
            print("# SHMR : %s = %6.4f x %s + %6.4f" % (star_col, shmr_a,
                                                        halo_col, shmr_b))

        return (shmr_a, shmr_b)

    def mockFitSimpleMhScatter(self, um_min_mvir=12.0, verbose=True,
                               n_bins=10, halo_col='logmh_vir',
                               star_col='logms_halo'):
        """
        Fit a log-log linear relation between halo mass and the scatter
        of the stellar mass using the running median:
            Sigma_Ms = c x logMh + d

        Parameters:
        -----------

        um_min_mvir : float, optional
            Minimum halo mass used in the fitting.
            Default : None

        n_bins : int, optional
            Number of bins in logM_Halo.
            Default: 10

        halo_col : string, optional
            Column for halo mass.
            Default: 'logmh_vir'

        star_col : string, optional
            Column for stellar mass.
            Default: 'logms_halo'
        """
        # First, estimate the running scatters of logMs_tot
        mhalo_bin = np.linspace(um_min_mvir, 15.0, n_bins)

        mock_central = self.um_mock[self.um_mock['mask_central']]
        log_mh = mock_central[halo_col]
        log_ms = mock_central[star_col]

        idx_mhalo = np.digitize(log_mh, mhalo_bin)
        mhalo_center = [np.nanmean(log_mh[idx_mhalo == k])
                        for k in range(len(mhalo_bin))]
        sigma_mstar = [np.nanstd(log_ms[idx_mhalo == k])
                       for k in range(len(mhalo_bin))]

        # Fit logMhalo - sigma_Ms_tot relation
        # Log-Log Linear relation
        sigms_a, sigms_b = np.polyfit(mhalo_center,
                                      sigma_mstar,
                                      1)
        if verbose:
            print("# Mh-SigMs : Sig(%s) = %6.4f x %s + %6.4f" % (
                star_col, sigms_a, halo_col, sigms_b))

        return (sigms_a, sigms_b, mhalo_center, sigma_mstar)

    def mockGetSimpleMhScatter(self, logMhalo, sigms_a, sigms_b,
                               min_scatter=None,
                               max_scatter=None):
        """
        Get the predicted scatter of stellar mass at fixed halo mass.
        Assuming a simple log-log linear relation.

        Parameters:
        -----------

        logMhalo : ndarray
            log10(Mhalo), for UM, use the true host halo mass.

        sigms_a : float
            Slope of the SigMs = a x logMh + b relation.

        sigms_b : float
            Normalization of the SigMs = a x logMh + b relation.

        min_scatter : float, optional
            Minimum allowed scatter.
            Sometimes the relation could lead to negative or super tiny
            scatter at high-mass end.  Use min_scatter to replace the
            unrealistic values.
            Default: 0.05
        """
        logSigMs = (sigms_a * logMhalo + sigms_b)

        if min_scatter is None:
            min_scatter = self.um_min_scatter
        logSigMs[logSigMs <= min_scatter] = min_scatter

        if max_scatter is not None:
            logSigMs[logSigMs >= max_scatter] = max_scatter

        return np.asarray(logSigMs)

    def lnPrior(self, theta):
        """
        Set the priors, right now it is sadly flat....
        """
        shmr_a, shmr_b, sigms_a, sigms_b = theta

        if shmr_a < self.shmr_a_low:
            return -np.inf
        if shmr_a > self.shmr_a_upp:
            return -np.inf

        if shmr_b < self.shmr_b_low:
            return -np.inf
        if shmr_b > self.shmr_b_upp:
            return -np.inf

        if sigms_a < self.sigms_a_low:
            return -np.inf
        if sigms_a > self.sigms_a_upp:
            return -np.inf

        if sigms_b < self.sigms_b_low:
            return -np.inf
        if sigms_b > self.sigms_b_upp:
            return -np.inf

        return 0.0

    def lnProb(self, theta):
        """
        Probability function to sample in an MCMC.

        Parameters:
        -----------

        theta : tuple
            Input parameters = (shmr_a, shmr_b, sigms_a, sigms_b)
        """
        lp = self.lnPrior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnLike(theta)

    def lnLike(self, theta):
        """
        Log likelihood of a UM model.

        Parameters:
        -----------

        theta : tuple
            Input parameters = (shmr_a, shmr_b, sigms_a, sigms_b)

        wl_weight : float, optional
            Noive weighting for WL profiles when adding the likelihood.
            Default: 1.0

        smf_only : boolen, optional
            Only fit the SMF.
            Default: False

        wl_only : boolen, optional
            Only fit the WL profiles.
            Default: False
        """
        # Unpack the input parameters
        shmr_a, shmr_b, sigms_a, sigms_b = theta

        # Generate the model predictions
        um_model = self.umPredictModel(shmr_a, shmr_b, sigms_a, sigms_b)
        (um_smf_tot, um_smf_inn, um_wl_profs,
         logms_mod_inn, logms_mod_tot_all,
         logms_mod_halo_all, mask_mtot) = um_model

        # Check SMF
        # msg = '# UM and observed SMFs should have the same size!'
        # assert len(um_smf_inn) == len(self.obs_smf_inn), msg
        # assert len(um_smf_tot) == len(self.obs_smf_tot), msg

        if self.mcmc_wl_only is False:
            #  SMF for Mto t
            smf_mtot_invsigma2 = (
                1.0 / (
                    (self.obs_smf_tot['smf_upp'][:-2] -
                     self.obs_smf_tot['smf'][:-2]) ** 2 +
                    (um_smf_tot['smf_err'][:-2] ** 2)
                )
            )
            """
            smf_mtot_invsigma2 = (
                1.0 / (
                    (self.obs_smf_tot['smf_upp'][:-2] -
                     self.obs_smf_tot['smf'][:-2]) ** 2
                )
            )
            """

            lnlike_smf = (
                -0.5 * (
                    np.nansum((self.obs_smf_tot['smf'][:-2] -
                               um_smf_tot['smf'][:-2]) ** 2 *
                              smf_mtot_invsigma2 -
                              np.log(smf_mtot_invsigma2))
                )
            )
            """
            lnlike_mtot = (
                -0.5 * (
                    np.nansum((self.obs_smf_tot['smf'][:-2] -
                               um_smf_tot['smf'][:-2]) ** 2 *
                              smf_mtot_invsigma2)
                )
            )
            """
        else:
            lnlike_smf = np.nan

        # Check WL profiles
        # msg = '# UM and observed WL profiles should have the same size!'
        # assert len(um_wl_profs) == len(self.obs_wl_profs)
        # assert len(um_wl_profs[0] == len(self.obs_wl_profs[0].r))

        if self.mcmc_smf_only is False:
            lnlike_wl = 0.0
            for ii in range(self.obs_wl_n_bin):
                wl_invsigma2 = (
                    1.0 / (
                        (self.obs_wl_profs[ii].err_w ** 2) +
                        (um_wl_profs[ii] ** 2)
                    )
                )

                lnlike_wl += (
                    -0.5 * (
                        np.nansum((self.obs_wl_profs[ii].sig -
                                   um_wl_profs[ii]) ** 2 * wl_invsigma2 -
                                  np.log(wl_invsigma2))
                    )
                )
        else:
            lnlike_wl = np.nan

        if self.mcmc_smf_only:
            return lnlike_smf
        elif self.mcmc_wl_only:
            return lnlike_wl
        else:
            return (lnlike_smf + self.mcmc_wl_weight * lnlike_wl)

    def umPlotSMF(self, um_smf_tot, um_smf_inn,
                  logms_mod_tot, logms_mod_inn,
                  shmr_a=None, shmr_b=None,
                  sigms_a=None, sigms_b=None,
                  um_smf_tot_all=None, **kwargs):
        """
        Plot the UM predicted M100-M10 plane and their SMFs.

        Parameters:
        -----------
        """
        fig, axes = plt.subplots(2, figsize=(7, 9))
        ax1 = axes[0]
        ax2 = axes[1]

        # Scatter plot
        if len(logms_mod_tot) > len(self.obs_logms_tot):
            ax1.scatter(logms_mod_tot, logms_mod_inn,
                        label=r'$\mathrm{Model}$',
                        s=12, alpha=0.7, marker=self.um_mark,
                        c='royalblue')

            ax1.scatter(self.obs_logms_tot, self.obs_logms_inn,
                        label=r'$\mathrm{Data}$',
                        s=15, alpha=0.5, marker=self.obs_mark,
                        c='lightsalmon')
        else:
            ax1.scatter(self.obs_logms_tot, self.obs_logms_inn,
                        label=r'$\mathrm{Data}$',
                        s=12, alpha=0.7, marker=self.obs_mark,
                        c='lightsalmon')

            ax1.scatter(logms_mod_tot, logms_mod_inn,
                        label=r'$\mathrm{Model}$',
                        s=15, alpha=0.5, marker=self.um_mark,
                        c='royalblue')

        ax1.legend(fontsize=15, loc='lower right')

        ax1.set_xlabel(r'$\log M_{\star,\ \mathrm{100,\ UM}}$', fontsize=20)
        ax1.set_ylabel(r'$\log M_{\star,\ \mathrm{10,\ UM}}$', fontsize=20)

        ax1.set_xlim(np.nanmin(self.obs_logms_tot) - 0.02,
                     np.nanmax(self.obs_logms_tot) + 0.09)
        ax1.set_ylim(np.nanmin(self.obs_logms_inn) - 0.02,
                     np.nanmax(self.obs_logms_inn) + 0.09)

        if shmr_a is not None and shmr_b is not None:
            seg1 = (r'$\log M_{\star} = %6.3f \times$' % shmr_a)
            seg2 = (r'$\log M_{\rm halo} + %6.3f$' % shmr_b)
            ax1.text(0.26, 0.91, (seg1 + seg2),
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     fontsize=12,
                     transform=ax1.transAxes)

        if sigms_a is not None and sigms_b is not None:
            seg1 = (r'$\sigma(\log M_{\star}) = %6.3f \times$' % sigms_a)
            seg2 = (r'$\log M_{\rm halo} + %6.3f$' % sigms_b)
            ax1.text(0.26, 0.83, (seg1 + seg2),
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     fontsize=12,
                     transform=ax1.transAxes)

        # Full SMF in the background if available
        #  +0.1 dex is a magic number to convert S82 SMF from BC03 to
        #  FSPS model
        if self.obs_smf_full is not None:
            ax2.fill_between(self.obs_smf_full['logm_mean'],
                             np.log10(self.obs_smf_full['smf_low']) + 0.1,
                             np.log10(self.obs_smf_full['smf_upp']) + 0.1,
                             facecolor='mediumseagreen',
                             edgecolor='none',
                             interpolate=True,
                             alpha=0.55, zorder=0,
                             label=r'$\mathrm{Data:\ S82}$')

            ax2.scatter(self.obs_smf_full['logm_mean'],
                        np.log10(self.obs_smf_full['smf']) + 0.1,
                        c='seagreen', marker='s',
                        s=20, label='__no_label__',
                        alpha=1.0, zorder=0)

        if um_smf_tot_all is not None:
            print("test")
            ax2.plot(um_smf_tot_all['logm_mean'],
                     np.log10(um_smf_tot_all['smf']),
                     linewidth=1.5, linestyle='--',
                     c='royalblue',
                     label='__no_label__')

        # SMF plot
        ax2.fill_between(self.obs_smf_tot['logm_mean'],
                         np.log10(self.obs_smf_tot['smf_low']),
                         np.log10(self.obs_smf_tot['smf_upp']),
                         facecolor='steelblue',
                         edgecolor='none',
                         interpolate=True,
                         alpha=0.4,
                         label=r'$\mathrm{Data:\ Mtot}$')

        ax2.fill_between(self.obs_smf_inn['logm_mean'],
                         np.log10(self.obs_smf_inn['smf_low']),
                         np.log10(self.obs_smf_inn['smf_upp']),
                         facecolor='lightsalmon',
                         edgecolor='none',
                         interpolate=True,
                         alpha=0.4,
                         label=r'$\mathrm{Data:\ Minn}$')

        ax2.scatter(self.obs_smf_inn['logm_mean'],
                    np.log10(self.obs_smf_inn['smf']),
                    marker=self.obs_mark,
                    c='lightsalmon',
                    s=20, label='__no_label__',
                    alpha=1.0)

        ax2.scatter(self.obs_smf_tot['logm_mean'],
                    np.log10(self.obs_smf_tot['smf']),
                    marker=self.obs_mark,
                    c='steelblue',
                    s=20, label='__no_label__',
                    alpha=1.0)

        ax2.plot(um_smf_tot['logm_mean'],
                 np.log10(um_smf_tot['smf']),
                 linewidth=4, linestyle='--',
                 c='royalblue',
                 label=r'$\mathrm{UM:\ Mtot}$')

        ax2.plot(um_smf_inn['logm_mean'],
                 np.log10(um_smf_inn['smf']),
                 linewidth=4, linestyle='--',
                 c='salmon',
                 label=r'$\mathrm{UM:\ Minn}$')

        ax2.legend(fontsize=12, loc='upper right')

        ax2.set_xlabel(r'$\log (M_{\star}/M_{\odot})$',
                       fontsize=20)
        ax2.set_ylabel((r'$\mathrm{d}N/\mathrm{d}\log M_{\star}\ $'
                        r'$[{\mathrm{Mpc}^{-3}}{\mathrm{dex}^{-1}}]$'),
                       size=20)

        mask_inn = np.log10(self.obs_smf_inn['smf']) > -7.5
        mask_tot = np.log10(self.obs_smf_tot['smf']) > -7.5

        ax2.set_xlim(np.nanmin(self.obs_smf_inn[mask_inn]['logm_mean']) - 0.15,
                     np.nanmax(self.obs_smf_tot[mask_tot]['logm_mean']) + 0.55)

        if self.obs_smf_full is not None:
            ax2.set_ylim(np.nanmin(np.log10(self.obs_smf_inn[mask_inn]['smf']))
                         - 0.2,
                         np.nanmax(np.log10(self.obs_smf_full['smf']))
                         )
        else:
            ax2.set_ylim(np.nanmin(np.log10(self.obs_smf_inn[mask_inn]['smf']))
                         - 0.2,
                         np.nanmax(np.log10(self.obs_smf_tot[mask_tot]['smf']))
                         + 0.8)

        return

    def umPlotWL(self, um_wl_profs,
                 obs_mhalo=None,
                 um_wl_mhalo=None,
                 **kwargs):
        """
        Plot the UM predicted weak lensing profiles.

        Parameters:
        -----------
        """
        if self.obs_wl_n_bin <= 4:
            n_col = self.obs_wl_n_bin
            n_row = 1
        else:
            n_col = 4
            n_row = int(np.ceil(self.obs_wl_n_bin / 4.0))

        fig = plt.figure(figsize=(3 * n_row, 3.8 * n_col))
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.0, hspace=0.00)

        min_wl = np.nanmin(np.asarray(um_wl_profs)) * 0.9
        max_wl = np.nanmax(np.asarray(um_wl_profs)) * 1.4

        for ii in range(self.obs_wl_n_bin):

            ax = plt.subplot(gs[ii])
            ax.loglog()

            if ii % n_col != 0:
                ax.yaxis.set_major_formatter(NullFormatter())
            else:
                ax.set_ylabel(r'$\Delta\Sigma$ $[M_{\odot}/{\rm pc}^2]$',
                              size=20)
            if ii % n_row != 0:
                ax.xaxis.set_major_formatter(NullFormatter())
            else:
                ax.set_xlabel(r'$r_{\rm p}$ ${\rm [Mpc]}$',
                              size=20)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)

            # Observed WL profile
            obs_prof = self.obs_wl_profs[ii]
            ax.errorbar(obs_prof.r, obs_prof.sig,
                        obs_prof.err_w, fmt='o',
                        color='lightsalmon',
                        ecolor='salmon',
                        alpha=0.8)
            ax.plot(obs_prof.r, obs_prof.sig,
                    linewidth=0.5, color='salmon',
                    alpha=0.5)

            # Label the mass range
            ax.text(0.04, 0.28,
                    r'${\rm Bin: %d}$' % obs_prof.bin_id,
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=12.0,
                    transform=ax.transAxes,
                    color='k', alpha=0.9)

            ax.text(0.04, 0.18,
                    r"$\log M_{\rm tot}:[%5.2f,%5.2f]$" % (obs_prof.low_mtot,
                                                           obs_prof.upp_mtot),
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=12.0,
                    transform=ax.transAxes,
                    color='k', alpha=0.9)

            ax.text(0.04, 0.08,
                    r"$\log M_{\rm inn}:[%5.2f,%5.2f]$" % (obs_prof.low_minn,
                                                           obs_prof.upp_minn),
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=12.0,
                    transform=ax.transAxes,
                    color='k', alpha=0.9)

            # Predicted WL profile
            ax.plot(obs_prof.r, um_wl_profs[ii],
                    linewidth=3.0, color='royalblue')

            if um_wl_mhalo is not None:
                ax.text(0.35, 0.90,
                        r"$[\log M_{\rm Vir, UM}]=%5.2f$" % (um_wl_mhalo[ii]),
                        verticalalignment='center',
                        horizontalalignment='left',
                        fontsize=15.0,
                        transform=ax.transAxes,
                        color='royalblue')

            # X, Y Limits
            ax.set_xlim(0.05, 61.0)
            ax.set_ylim(0.1005, 299.0)
            # ax.set_ylim(min_wl, max_wl)

        return

    def umObsCompare(self, um_model):
        """
        Compare the predicted SMF and WL profiles to observations.

        Parameters:
        -----------
        """

        return

    def mcmcInitialGuess(self):
        """
        Set the starting position for the MCMC.

        Parameters:
        -----------
        """
        self.mcmc_position = np.zeros([self.mcmc_nwalkers,
                                       self.mcmc_ndims])

        # SHMR a
        self.mcmc_position[:, 0] = (
            self.theta_start[0] + 5e-2 *
            np.random.randn(self.mcmc_nwalkers)
            )

        # SHMR b
        self.mcmc_position[:, 1] = (
            self.theta_start[1] + 5e-2 *
            np.random.randn(self.mcmc_nwalkers)
            )

        # Mh-Sigma(Ms) a
        self.mcmc_position[:, 2] = (
            self.theta_start[2] + 5e-2 *
            np.random.randn(self.mcmc_nwalkers)
            )

        # Mh-Sigma(Ms) b
        self.mcmc_position[:, 3] = (
            self.theta_start[3] + 5e-2 *
            np.random.randn(self.mcmc_nwalkers)
            )

        return None

    def mcmcGetParameters(self):
        """
        Computes the 1D marginalized parameter constraints from
        self.mcmcsamples.
        """
        (self.shmr_a_mcmc, self.shmr_b_mcmc,
         self.sigms_a_mcmc, self.sigms_b_mcmc) = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(self.mcmc_samples, [16, 50, 84], axis=0))
            )

        return None

    def mcmcFit(self, verbose=True, **kwargs):
        """
        Peform an MCMC fit to the wp data using the power-law model.

        Parameters:

        -----------
        """
        # Setup the sampler
        if verbose:
            print("# Setup the sampler ...")
        self.mcmc_sampler = emcee.EnsembleSampler(
            self.mcmc_nwalkers, self.mcmc_ndims,
            self.lnProb
            )

        if verbose:
            print("# Setup the initial guesses ...")
        # Setup the initial condition
        self.mcmcInitialGuess()

        # Burn-in
        if verbose:
            print("# Phase: Burn-in ...")
        self.mcmc_burnin_result = self.mcmc_sampler.run_mcmc(
             self.mcmc_position, self.mcmc_nburnin
             )
        mcmc_burnin_position, _, mcmc_burnin_state = self.mcmc_burnin_result
        #  Pickle the results
        self.mcmcSaveResults(self.mcmc_burnin_file,
                             self.mcmc_burnin_result)
        #  Pickle the chain
        self.mcmc_burnin_chain = self.mcmc_sampler.chain
        self.mcmcSaveChains(self.mcmc_burnin_chain_file,
                            self.mcmc_burnin_chain)

        # Rest the chains
        self.mcmc_sampler.reset()

        # MCMC run
        if verbose:
            print("# Phase: MCMC run ...")
        self.mcmc_run_result = self.mcmc_sampler.run_mcmc(
            mcmc_burnin_position, self.mcmc_nsamples,
            rstate0=mcmc_burnin_state)
        #  Pickle the result
        self.mcmcSaveResults(self.mcmc_run_file,
                             self.mcmc_run_result)
        self.mcmc_run_chain = self.mcmc_sampler.chain
        self.mcmcSaveChains(self.mcmc_run_chain_file,
                            self.mcmc_run_chain)

        if verbose:
            print("# Get MCMC samples and best-fit parameters ...")
        # Get the MCMC samples
        self.mcmc_samples = self.mcmc_sampler.chain[:, :, :].reshape(
            (-1, self.mcmc_ndims)
            )
        #  Save the samples
        np.savez(self.mcmc_run_samples_file, data=self.mcmc_samples)

        self.mcmc_lnprob = self.mcmc_sampler.lnprobability.reshape(-1, 1)

        # Get the best-fit parameters and the 1-sigma error
        self.mcmcGetParameters()
        if verbose:
            print("#------------------------------------------------------")
            print("#  Mean acceptance fraction",
                  np.mean(self.mcmc_sampler.acceptance_fraction))
            print("#------------------------------------------------------")
            print("#  Best ln(Probability): %11.5f" %
                  np.nanmax(self.mcmc_lnprob))
            print("#------------------------------------------------------")
            print("# logMs,tot = "
                  "%7.4f x logMvir + %7.4f" % (self.shmr_a_mcmc[0],
                                               self.shmr_b_mcmc[0])
                  )
            print("#  a Error:  +%7.4f/-%7.4f" % (self.shmr_a_mcmc[1],
                                                  self.shmr_a_mcmc[2]))
            print("#  b Error:  +%7.4f/-%7.4f" % (self.shmr_b_mcmc[1],
                                                  self.shmr_b_mcmc[2]))
            print("#------------------------------------------------------")
            print("# sigma(logMs,tot) = "
                  "%7.4f x logMvir + %7.4f" % (self.sigms_a_mcmc[0],
                                               self.sigms_b_mcmc[0])
                  )
            print("#  c Error:  +%7.4f/-%7.4f" % (self.sigms_a_mcmc[1],
                                                  self.sigms_a_mcmc[2]))
            print("#  d Error:  +%7.4f/-%7.4f" % (self.sigms_b_mcmc[1],
                                                  self.sigms_b_mcmc[2]))
            print("#------------------------------------------------------")

        return None

    def mcmcCornerPlot(self):
        """
        Show the corner plot of the MCMC samples.
        """
        import corner
        from palettable.colorbrewer.sequential import OrRd_9
        ORG = OrRd_9.mpl_colormap

        fig = corner.corner(
            self.mcmc_samples,
            bins=25, color=ORG(0.7),
            smooth=1, labels=self.mcmc_labels,
            label_kwargs={'fontsize': 40},
            quantiles=[0.16, 0.5, 0.84],
            plot_contours=True,
            fill_contours=True,
            show_titles=True,
            title_kwargs={"fontsize": 30},
            hist_kwargs={"histtype": 'stepfilled',
                         "alpha": 0.4,
                         "edgecolor": "none"},
            use_math_text=True
            )

        return fig

    def mcmcTracePlot(self):
        """
        Tacee plot of the MCMC sampling.
        """

        return

    def mcmcSaveChains(self, mcmc_chain_file, mcmc_chain, **kwargs):
        """
        Save the chain to an ascii file.
        """
        pickle_file = open(mcmc_chain_file, 'wb')
        pickle.dump(mcmc_chain, pickle_file)
        pickle_file.close()

        return None

    def mcmcSaveResults(self, pkl_name, mcmc_result, **kwargs):
        """
        Save the MCMC run results into a pickle file.
        """
        pkl_file = open(pkl_name, 'wb')
        mcmc_position, mcmc_prob, mcmc_state = mcmc_result
        pickle.dump(mcmc_position, pkl_file, -1)
        pickle.dump(mcmc_prob, pkl_file, -1)
        pickle.dump(mcmc_state, pkl_file, -1)
        pkl_file.close()

        return None

    def mcmcLoadChains(self, mcmc_chain_file, **kwargs):
        """
        Save the chain to an ascii file.
        """
        pickle_file = open(mcmc_chain_file, 'rb')
        self.mcmc_chain = pickle.load(pickle_file)
        pickle_file.close()

        return None


class boxBinWL(SwotWL):
    """
    Class for HSC weak lensing profile within a box defined by
    Mtot and Minn.
    """

    def setBinId(self, bin_id):
        """
        Set the bin id for the box.
        """
        self.bin_id = bin_id

    def setMassLimits(self, low_mtot, upp_mtot, low_minn, upp_minn):
        """
        Set the lower and upper mass limits for both Mtot and Minn.
        """
        self.low_mtot = low_mtot
        self.upp_mtot = upp_mtot

        self.low_minn = low_minn
        self.upp_minn = upp_minn
