"""Model using the in-situ and ex-situ mass."""

from __future__ import print_function, division

import os
import pickle

import emcee
# from emcee.utils import MPIPool

import numpy as np

from scipy import interpolate

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

from halotools.mock_observables import delta_sigma_from_precomputed_pairs

from stellar_mass_function import get_smf_bootstrap
from full_mass_profile_model import mass_prof_model_simple, \
    mass_prof_model_frac1
from um_model_plot import plot_mtot_minn_smf, plot_dsigma_profiles
from asap_utils import mcmc_save_results, mcmc_save_chains
# from convergence import convergence_check


class InsituExsituModel(object):
    """UniverseMachine Mhalo-M100-M10 models.

    Based on the Mh_vir, Ms_GAL, Ms_ICL, and Ms_Halo predicted by the
    MhaloUniverseMachine model.

    At UniverseMachine side:
        Mh_vir  : Virial mass of the host halo
        Ms_ins  : Stellar mass of the in-situ component
        Ms_exs  : Stellar mass of the ex-situ component (accreted stars)
        Ms_tot  : M_ins + M_exs, total stellar mass of the galaxy
        Ms_halo : Total stellar mass of all galaxies in a halo
                  (central + satellite)

    At HSC Observation side:
        M_inn   : Mass within a smaller aperture
        M_tot   : Mass within a larger aperture, as proxy of the observed
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
        """Initialize the model."""
        # Setup the HSC data
        self.setupHscData(verbose=True, **kwargs)

        # Setup data from UniverseMachine model
        self.setupUniverseMachine(verbose=True, **kwargs)

        # Setup configuration parameters
        self.setupConfig(verbose=True, **kwargs)

    def setupHscData(self, verbose=False, **kwargs):
        """HSC Data."""
        # ---------------- Cosmology Related ----------------- #
        # This is for HSC observation
        if 'cosmos_h0' in kwargs.keys():
            obs_h0 = kwargs['obs_h0']
        else:
            obs_h0 = 0.70

        if 'cosmo_omega_m' in kwargs.keys():
            obs_omega_m = kwargs['obs_omega_m']
        else:
            obs_omega_m = 0.307

        self.cosmo = FlatLambdaCDM(H0=obs_h0 * 100,
                                   Om0=obs_omega_m)
        # --------------------------------------------------- #

        # -------------- Observed Data Related -------------- #
        # Catalog for observed galaxies
        if 'obs_dir' in kwargs.keys():
            obs_dir = kwargs['obs_dir']
        else:
            obs_dir = '../data/s16a_massive_wide2'

        if 'obs_cat' in kwargs.keys():
            obs_cat = kwargs['obs_cat']
        else:
            obs_cat = 's16a_wide2_massive_fsps1_imgsub_use_short.fits'
        if verbose:
            print("# Input stellar mass catalog: %s" % obs_cat)

        obs_mass = Table.read(os.path.join(obs_dir, obs_cat))

        # --------------------------------------------------- #
        # Observed weak lensing delta sigma profiles
        if 'obs_wl_sample' in kwargs.keys():
            obs_wl_sample = kwargs['obs_wl_sample']
        else:
            obs_wl_sample = 's16a_wide2_massive_boxbin5_default'
        if verbose:
            print("# Input weak lensing profile sample: %s" % obs_wl_sample)

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
        if 'obs_smf_inn' in kwargs.keys():
            smf_inn_file = os.path.join(obs_dir, kwargs['obs_smf_inn'])
        else:
            smf_inn_file = os.path.join(
                obs_dir, 'smf', 's16a_wide2_massive_smf_m10_11.5.fits')
        if verbose:
            print("# Pre-computed SMF for inner logMs: %s" % smf_inn_file)

        if 'obs_smf_tot' in kwargs.keys():
            smf_tot_file = os.path.join(obs_dir, kwargs['obs_smf_tot'])
        else:
            smf_tot_file = os.path.join(
                obs_dir, 'smf', 's16a_wide2_massive_smf_mmax_11.5.fits')
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
        smf_tot_fits = 'primus_smf_z0.3_0.4.fits'
        obs_smf_full_file = os.path.join(obs_dir, smf_tot_fits)
        if os.path.isfile(obs_smf_full_file):
            smf_full = Table.read(obs_smf_full_file)
            smf_full[smf_full['smf'] <= 0]['smf'] = 1E-8
            smf_full[smf_full['smf_low'] <= 0]['smf_low'] = 1E-9
            smf_full[smf_full['smf_upp'] <= 0]['smf_upp'] = 1E-7
            self.obs_smf_full = smf_full
            if verbose:
                print("# Pre-computed full SMF: %s" % smf_tot_fits)
        else:
            self.obs_smf_full = None

        # --------------------------------------------------- #
        # Volume of the data
        if 'obs_area' in kwargs.keys():
            obs_area = kwargs['obs_area']
        else:
            obs_area = 145.0

        if 'obs_z_col' in kwargs.keys():
            obs_z_col = kwargs['obs_z_col']
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
        if 'obs_minn_col' in kwargs.keys():
            obs_minn_col = kwargs['obs_minn_col']
        else:
            obs_minn_col = 'logm_10'

        if 'obs_mtot_col' in kwargs.keys():
            obs_mtot_col = kwargs['obs_mtot_col']
        else:
            obs_mtot_col = 'logm_max'

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

    def setupUniverseMachine(self, verbose=False, **kwargs):
        """Load UniverseMachine data."""
        # ---------- UniverseMachine Mock Related ----------- #
        if 'um_dir' in kwargs.keys():
            um_dir = kwargs['um_dir']
        else:
            um_dir = '../data/s16a_massive_wide2/um2'

        if 'um_lbox' in kwargs.keys():
            self.um_lbox = kwargs['um_lbox']
        else:
            # For SMDPL
            self.um_lbox = 400.0  # Mpc/h

        # H0 for the model
        if 'um_h0' in kwargs.keys():
            um_h0 = kwargs['um_h0']
        else:
            um_h0 = 0.678

        if 'um_omega_m' in kwargs.keys():
            um_omega_m = kwargs['um_omega_m']
        else:
            um_omega_m = 0.307

        self.um_cosmo = FlatLambdaCDM(H0=um_h0 * 100.0,
                                      Om0=um_omega_m)

        self.um_volume = np.power(self.um_lbox / um_h0, 3)
        if verbose:
            print("# The volume of the UniverseMachine mock is %15.2f Mpc^3" %
                  self.um_volume)

        # Value added catalog
        if 'um_model' in kwargs.keys():
            um_model = kwargs['um_model']
        else:
            um_model = 'um_smdpl_0.7124_new_vagc_mpeak_11.5.npy'

        # Precomputed weak lensing paris
        if 'um_wl_cat' in kwargs.keys():
            um_wl_cat = kwargs['um_wl_cat']
        else:
            um_wl_cat = ('um_smdpl_0.7124_new_vagc_mpeak_11.5' +
                         '_10m_r_0.08_50_22bins.npy')

        self.um_mock = Table(np.load(os.path.join(um_dir, um_model)))
        self.um_mass_encl = np.load(os.path.join(um_dir, um_wl_cat))
        assert len(self.um_mock) == len(self.um_mass_encl)

        # Mask for central galaxies
        self.mask_central = self.um_mock['mask_central']

        if 'um_min_mvir' in kwargs.keys():
            self.um_min_mvir = kwargs['um_min_mvir']
        else:
            self.um_min_mvir = 11.5

        if 'um_redshift' in kwargs.keys():
            self.um_redshift = kwargs['um_redshift']
        else:
            self.um_redshift = 0.3637

        if 'um_wl_minr' in kwargs.keys():
            self.um_wl_minr = kwargs['um_wl_minr']
        else:
            self.um_wl_minr = 0.08

        if 'um_wl_maxr' in kwargs.keys():
            self.um_wl_maxr = kwargs['um_wl_maxr']
        else:
            self.um_wl_maxr = 50.0

        if 'um_wl_nbin' in kwargs.keys():
            self.um_wl_nbin = kwargs['um_wl_nbin']
        else:
            self.um_wl_nbin = 22

        if 'um_wl_add_stellar' in kwargs.keys():
            self.um_wl_add_stellar = kwargs['um_wl_add_stellar']
        else:
            self.um_wl_add_stellar = False

        if 'um_mtot_nbin' in kwargs.keys():
            self.um_mtot_nbin = kwargs['um_mtot_nbin']
        else:
            self.um_mtot_nbin = 80

        if 'um_mtot_nbin_min' in kwargs.keys():
            self.um_mtot_nbin_min = kwargs['um_mtot_nbin_min']
        else:
            self.um_mtot_nbin_min = 7

        if 'um_ngal_bin_min' in kwargs.keys():
            self.um_min_nobj_per_bin = kwargs['um_min_nobj_per_bin']
        else:
            self.um_min_nobj_per_bin = 30

        if 'um_min_scatter' in kwargs.keys():
            self.um_min_scatter = kwargs['um_min_scatter']
        else:
            self.um_min_scatter = 0.01
        # --------------------------------------------------- #

    def setupConfig(self, verbose=False, **kwargs):
        """Configure MCMC run and plots."""
        if 'model_type' in kwargs.keys():
            self.model_type = kwargs['model_type']
        else:
            self.model_type = 'frac1'

        if self.model_type == 'simple':

            # Number of parameters
            self.mcmc_ndims = 4
            self.mcmc_labels = [r'$a_{\mathrm{SMHR}}$',
                                r'$b_{\mathrm{SMHR}}$',
                                r'$a_{\sigma \log M_{\star}}$',
                                r'$b_{\sigma \log M_{\star}}$']

            # Initial values
            if 'param_ini' in kwargs.keys():
                self.param_ini = kwargs['param_ini']
                assert len(self.param_ini) == self.mcmc_ndims
            else:
                self.param_ini = [0.59901, 3.69888,
                                  -0.0824, 1.2737]
            # Lower bounds
            if 'param_low' in kwargs.keys():
                self.param_low = kwargs['param_low']
                assert len(self.param_low) == self.mcmc_ndims
            else:
                self.param_low = [0.2, -1.5, -0.2, 0.0]

            # Upper bounds
            if 'param_upp' in kwargs.keys():
                self.param_upp = kwargs['param_upp']
                assert len(self.param_upp) == self.mcmc_ndims
            else:
                self.param_upp = [1.0, 8.0, 0.0, 1.6]

            # Step to randomize the initial guesses
            if 'param_sig' in kwargs.keys():
                self.param_sig = kwargs['param_sig']
                assert len(self.param_sig) == self.mcmc_ndims
            else:
                self.param_sig = [0.1, 0.3, 0.05, 0.2]

        elif self.model_type == 'frac1':

            # Number of parameters
            self.mcmc_ndims = 6
            self.mcmc_labels = [r'$a_{\mathrm{SMHR}}$',
                                r'$b_{\mathrm{SMHR}}$',
                                r'$a_{\sigma \log M_{\star}}$',
                                r'$b_{\sigma \log M_{\star}}$',
                                r'$\mathrm{f}_{\mathrm{in-situ}}$',
                                r'$\mathrm{f}_{\mathrm{ex-situ}}$']

            # Initial values
            if 'param_ini' in kwargs.keys():
                self.param_ini = kwargs['param_ini']
                assert len(self.param_ini) == self.mcmc_ndims
            else:
                self.param_ini = [0.599017, 3.668879,
                                  -0.0476, 0.020,
                                  0.80, 0.10]
            # Lower bounds
            if 'param_low' in kwargs.keys():
                self.param_low = kwargs['param_low']
                assert len(self.param_low) == self.mcmc_ndims
            else:
                self.param_low = [0.2, 0.0, -0.2, 0.0, 0.3, 0.0]

            # Upper bounds
            if 'param_upp' in kwargs.keys():
                self.param_upp = kwargs['param_upp']
                assert len(self.param_upp) == self.mcmc_ndims
            else:
                self.param_upp = [1.0, 8.0, 0.0, 0.2, 1.0, 0.3]

            # Step to randomize the initial guesses
            if 'param_sig' in kwargs.keys():
                self.param_sig = kwargs['param_sig']
                assert len(self.param_sig) == self.mcmc_ndims
            else:
                self.param_sig = [0.05, 0.1, 0.02, 0.005, 0.05, 0.05]

        else:
            raise Exception("# Wrong model! Has to be 'simple' or `frac1`")
        # --------------------------------------------------- #

        # ------------------- MCMC Related ------------------ #
        if 'mcmc_nsamples' in kwargs.keys():
            self.mcmc_nsamples = kwargs['mcmc_nsamples']
        else:
            self.mcmc_nsamples = 200

        if 'mcmc_nthreads' in kwargs.keys():
            self.mcmc_nthreads = kwargs['mcmc_nthreads']
        else:
            self.mcmc_nthreads = 2

        if 'mcmc_nburnin' in kwargs.keys():
            self.mcmc_nburnin = kwargs['mcmc_nburnin']
        else:
            self.mcmc_nburnin = 100

        if 'mcmc_nwalkers' in kwargs.keys():
            self.mcmc_nwalkers = kwargs['mcmc_nwalkers']
        else:
            self.mcmc_nwalkers = 100

        if 'mcmc_smf_only' in kwargs.keys():
            self.mcmc_smf_only = kwargs['mcmc_smf_only']
        else:
            self.mcmc_smf_only = False

        if 'mcmc_wl_only' in kwargs.keys():
            self.mcmc_wl_only = kwargs['mcmc_wl_only']
        else:
            self.mcmc_wl_only = False

        if 'mcmc_wl_weight' in kwargs.keys():
            self.mcmc_wl_weight = kwargs['mcmc_wl_weight']
        else:
            self.mcmc_wl_weight = 1.0

        if 'mcmc_burnin_file' in kwargs.keys():
            self.mcmc_burnin_file = kwargs['mcmc_burnin_file']
        else:
            self.mcmc_burnin_file = 'um_smdpl_m100_m10_burnin_result.pkl'

        if 'mcmc_run_file' in kwargs.keys():
            self.mcmc_run_file = kwargs['mcmc_run_file']
        else:
            self.mcmc_run_file = 'um_smdpl_m100_m10_run_result.pkl'

        if 'mcmc_burnin_chain_file' in kwargs.keys():
            self.mcmc_burnin_chain_file = kwargs['mcmc_burnin_chain_file']
        else:
            self.mcmc_burnin_chain_file = 'um_smdpl_m100_m10_burnin_chain.pkl'

        if 'mcmc_run_chain_file' in kwargs.keys():
            self.mcmc_run_chain_file = kwargs['mcmc_run_chain_file']
        else:
            self.mcmc_run_chain_file = 'um_smdpl_m100_m10_run_chain.pkl'

        if 'mcmc_run_samples_file' in kwargs.keys():
            self.mcmc_run_samples_file = kwargs['mcmc_run_samples_file']
        else:
            self.mcmc_run_samples_file = 'um_smdpl_m100_m10_run_samples.npz'
        # --------------------------------------------------- #

        return None

    def umGetWLProf(self, mask, mock_use=None, mass_encl_use=None,
                    verbose=False, r_interp=None, mstar_lin=None):
        """Weak lensing dsigma profiles using pre-computed pairs.

        Parameters
        ----------
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

        if verbose:
            print("# Deal with %d galaxies in the subsample" % np.sum(mask))
        #  Use the mask to get subsample positions and pre-computed pairs
        subsample = mock_use[mask]
        subsample_positions = np.vstack([subsample['x'],
                                         subsample['y'],
                                         subsample['z']]).T

        subsample_mass_encl_precompute = mass_encl_use[mask, :]

        if self.um_cosmo is None:
            um_cosmo = self.sim_halocat.cosmology
        else:
            um_cosmo = self.um_cosmo

        rp_ht_units, ds_ht_units = delta_sigma_from_precomputed_pairs(
            subsample_positions,
            subsample_mass_encl_precompute,
            rp_bins, self.um_lbox,
            cosmology=self.um_cosmo
        )

        # Unit conversion
        ds_phys_msun_pc2 = ((1. + self.um_redshift) ** 2 *
                            (ds_ht_units * self.um_cosmo.h) /
                            (1e12))

        rp_phys = ((rp_ht_units) /
                   (abs(1. + self.um_redshift) *
                    um_cosmo.h))

        # Add the point source term
        if mstar_lin is not None:
            ds_phys_msun_pc2[0] += (
                mstar_lin / 1e12 / (np.pi * (rp_phys ** 2.0))
                )

        if r_interp is not None:
            intrp = interpolate.interp1d(rp_phys, ds_phys_msun_pc2,
                                         kind='cubic', bounds_error=False)
            dsigma = intrp(r_interp)

            return (r_interp, dsigma)

        return (rp_phys, ds_phys_msun_pc2)

    def umPredictMass(self, parameters,
                      star_col='logms_tot',
                      halo_col='logmh_vir',
                      constant_bin=False):
        """M100, M10, Mtot using Mvir, M_gal, M_ICL.

        Parameters
        ----------

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
        if self.model_type == 'simple':
            return mass_prof_model_simple(
                self.um_mock,
                self.obs_logms_tot,
                self.obs_logms_inn,
                parameters,
                min_logms=self.obs_smf_tot_min,
                max_logms=self.obs_smf_tot_max,
                n_bins=self.um_mtot_nbin,
                constant_bin=constant_bin,
                logmh_col=halo_col,
                logms_col=star_col,
                min_scatter=self.um_min_scatter,
                min_nobj_per_bin=self.um_min_nobj_per_bin
                )
        elif self.model_type == 'frac1':
            return mass_prof_model_frac1(
                self.um_mock,
                self.obs_logms_tot,
                self.obs_logms_inn,
                parameters,
                min_logms=self.obs_smf_tot_min,
                max_logms=self.obs_smf_tot_max,
                n_bins=self.um_mtot_nbin,
                constant_bin=constant_bin,
                logmh_col=halo_col,
                logms_col=star_col,
                min_scatter=self.um_min_scatter,
                min_nobj_per_bin=self.um_min_nobj_per_bin
                )
        else:
            raise Exception("# Wrong model choice! ")

    def umPredictSMF(self, logms_mod_tot, logms_mod_inn):
        """Stellar mass functions of Minn and Mtot predicted by UM."""
        # SMF of the predicted Mtot (M1100)
        um_smf_tot = get_smf_bootstrap(logms_mod_tot,
                                       self.um_volume,
                                       self.obs_smf_tot_nbin,
                                       self.obs_smf_tot_min,
                                       self.obs_smf_tot_max,
                                       n_boots=1)

        # SMF of the predicted Minn (M10)
        um_smf_inn = get_smf_bootstrap(logms_mod_inn,
                                       self.um_volume,
                                       self.obs_smf_inn_nbin,
                                       self.obs_smf_inn_min,
                                       self.obs_smf_inn_max,
                                       n_boots=1)

        return (um_smf_tot, um_smf_inn)

    def umSingleWL(self, mock_use, mass_encl_use, obs_prof,
                   logms_mod_tot, logms_mod_inn,
                   um_wl_min_ngal=15, verbose=False,
                   add_stellar=False):
        """Individual WL profile for UM galaxies."""
        bin_mask = ((logms_mod_tot >= obs_prof.low_mtot) &
                    (logms_mod_tot <= obs_prof.upp_mtot) &
                    (logms_mod_inn >= obs_prof.low_minn) &
                    (logms_mod_inn <= obs_prof.upp_mtot))

        # "Point source" term for the central galaxy
        # TODO: more appropriate way to add stellar mass component?
        if add_stellar:
            mstar_lin = np.nanmedian(10.0 * logms_mod_tot[bin_mask])
        else:
            mstar_lin = None

        if np.sum(bin_mask) <= um_wl_min_ngal:
            # TODO: using zero or NaN ?
            wl_prof = np.zeros(len(self.obs_wl_dsigma[0].r))
            if verbose:
                print("# Not enough UM galaxy "
                      "in bin %d !" % obs_prof.bin_id)
        else:
            wl_rp, wl_prof = self.umGetWLProf(
                bin_mask,
                mock_use=mock_use,
                mass_encl_use=mass_encl_use,
                r_interp=self.obs_wl_dsigma[0].r,
                mstar_lin=mstar_lin
                )

        return wl_prof

    def umPredictWL(self, logms_mod_tot, logms_mod_inn, mask_mtot,
                    um_wl_min_ngal=15, verbose=False,
                    add_stellar=False):
        """WL profiles to compare with observations.

        Parameters
        ----------

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

        return [self.umSingleWL(mock_use, mass_encl_use, obs_prof,
                                logms_mod_tot, logms_mod_inn,
                                um_wl_min_ngal=um_wl_min_ngal,
                                verbose=verbose, add_stellar=add_stellar)
                for obs_prof in self.obs_wl_dsigma]

    def umPredictModel(self, parameters,
                       constant_bin=False,
                       plotSMF=False, plotWL=False):
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

        """
        # Predict stellar mass
        um_mass_model = self.umPredictMass(parameters,
                                           constant_bin=constant_bin)

        (logms_mod_inn, logms_mod_tot_all,
         logms_mod_halo, mask_mtot, um_mock_use) = um_mass_model

        # Predict the SMFs
        um_smf_tot, um_smf_inn = self.umPredictSMF(
            logms_mod_tot_all[mask_mtot],
            logms_mod_inn)

        # TODO: If one mass bin is empty, set the error to a higer value
        # mask_zero = um_smf_tot['smf'] <= 1.0E-10
        # um_smf_tot['smf'][mask_zero] = np.nan
        # um_smf_tot['smf_err'][mask_zero] = np.nan

        um_wl_profs = self.umPredictWL(logms_mod_tot_all[mask_mtot],
                                       logms_mod_inn,
                                       mask_mtot,
                                       add_stellar=self.um_wl_add_stellar)

        if plotSMF:
            um_smf_tot_all = get_smf_bootstrap(logms_mod_tot_all,
                                               self.um_volume,
                                               20, 10.5, 12.5,
                                               n_boots=1)
            logms_mod_tot = logms_mod_tot_all[mask_mtot]
            plot_mtot_minn_smf(self.obs_smf_tot, self.obs_smf_inn,
                               self.obs_mtot, self.obs_minn,
                               um_smf_tot, um_smf_inn,
                               logms_mod_tot,
                               logms_mod_inn,
                               obs_smf_full=self.obs_smf_full,
                               um_smf_tot_all=um_smf_tot_all)

        if plotWL:
            # TODO: add halo mass information
            plot_dsigma_profiles(self.obs_wl_dsigma,
                                 um_wl_profs,
                                 obs_mhalo=None,
                                 um_wl_mhalo=None)

        return (um_smf_tot, um_smf_inn, um_wl_profs,
                logms_mod_inn, logms_mod_tot_all[mask_mtot],
                logms_mod_halo, mask_mtot,
                um_mock_use)

    def mockFitSimpleSHMR(self, um_min_mvir=12.0, verbose=True,
                          halo_col='logmh_vir',
                          star_col='logms_halo'):
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
        # Only use central for this fit
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

    def lnPrior(self, param_tuple):
        """Priors of parameters."""
        param_list = list(param_tuple)

        for param, low, upp in zip(param_list,
                                   self.param_low,
                                   self.param_upp):
            if param <= low or param >= upp:
                return -np.inf

        return 0.0

    def lnProb(self, param_tuple):
        """Probability function to sample in an MCMC.

        Parameters
        ----------
        param_tuple: tuple of model parameters.

        """
        lp = self.lnPrior(param_tuple)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnLike(param_tuple)

    def wlLikelihood(self, index, um_wl_profs, chi2=False):
        """Calculate the likelihood for WL profile."""
        wl_obs = self.obs_wl_dsigma[index].sig
        wl_obs_err = self.obs_wl_dsigma[index].err_s
        wl_um = um_wl_profs[index]

        wl_var = (wl_obs_err[:-2] ** 2)
        wl_dif = (wl_obs[:-2] - wl_um[:-2]) ** 2

        if chi2 is False:
            lnlike_wl = -0.5 * (
                (wl_dif / wl_var).sum() +
                np.log(2 * np.pi * wl_var).sum()
                )
            print("WL bin %d likelihood: %f" % ((index + 1), lnlike_wl))
            print("          chi2: %f" % (wl_dif / wl_var).sum())
        else:
            lnlike_wl = np.nansum(
                ((wl_obs[:-2] - wl_um[:-2]) ** 2 / wl_var)
                )

        return lnlike_wl

    def reducedChi2(self, param_tuple):
        """Reduced chi2 of a UM model.

        Parameters
        ----------
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
        parameters = list(param_tuple)
        n_param = len(parameters)

        # Generate the model predictions
        (um_smf_tot,
         um_smf_inn,
         um_wl_profs,
         logms_mod_inn,
         logms_mod_tot,
         logms_mod_halo,
         mask_mtot,
         um_mock_use) = self.umPredictModel(parameters,
                                            constant_bin=False)

        # Check SMF
        msg = '# UM and observed SMFs should have the same size!'
        assert len(um_smf_inn) == len(self.obs_smf_inn), msg
        assert len(um_smf_tot) == len(self.obs_smf_tot), msg

        if self.mcmc_wl_only is False:
            #  SMF for Mto t
            smf_mtot_var = (
                (self.obs_smf_tot['smf_upp'] -
                 self.obs_smf_tot['smf']) ** 2
            )

            chi2_smf = (
                np.nansum(
                    ((self.obs_smf_tot['smf'] - um_smf_tot['smf']) ** 2 /
                     smf_mtot_var)
                ))
        else:
            chi2_smf = 0.0

        # Check WL profiles
        msg = '# UM and observed WL profiles should have the same size!'
        assert len(um_wl_profs) == len(self.obs_wl_dsigma)
        assert len(um_wl_profs[0]) == len(self.obs_wl_dsigma[0].r)

        if self.mcmc_smf_only is False:
            chi2_wl = np.nansum([self.wlLikelihood(ii, um_wl_profs, chi2=True)
                                 for ii in range(self.obs_wl_n_bin)])
        else:
            chi2_wl = 0.0

        return (chi2_smf + self.mcmc_wl_weight * chi2_wl) / n_param

    def lnLike(self, param_tuple):
        """Log likelihood of a UM model.

        Parameters
        ----------
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
        parameters = list(param_tuple)

        # Generate the model predictions
        (um_smf_tot,
         um_smf_inn,
         um_wl_profs,
         logms_mod_inn,
         logms_mod_tot,
         logms_mod_halo,
         mask_mtot,
         um_mock_use) = self.umPredictModel(parameters,
                                            constant_bin=False)

        # Check SMF
        msg = '# UM and observed SMFs should have the same size!'
        assert len(um_smf_inn) == len(self.obs_smf_inn), msg
        assert len(um_smf_tot) == len(self.obs_smf_tot), msg

        if self.mcmc_wl_only is False:
            #  SMF for Mto t
            smf_mtot_var = (
                self.obs_smf_tot['smf_upp'] - self.obs_smf_tot['smf']) ** 2

            smf_mtot_dif = (self.obs_smf_tot['smf'] - um_smf_tot['smf']) ** 2

            print("Chi2 for SMF: %f" % (smf_mtot_dif / smf_mtot_var).sum())

            lnlike_smf = -0.5 * (
                (smf_mtot_dif / smf_mtot_var).sum() +
                np.log(2 * np.pi * smf_mtot_var).sum()
                )

            print("lnLikelihood for SMF: %f" % lnlike_smf)
        else:
            lnlike_smf = 0.0

        # Check WL profiles
        msg = '# UM and observed WL profiles should have the same size!'
        assert len(um_wl_profs) == len(self.obs_wl_dsigma)
        assert len(um_wl_profs[0]) == len(self.obs_wl_dsigma[0].r)

        if self.mcmc_smf_only is False:
            lnlike_wl = np.nansum([self.wlLikelihood(ii, um_wl_profs)
                                   for ii in range(self.obs_wl_n_bin)])
        else:
            lnlike_wl = 0.0

        return lnlike_smf + self.mcmc_wl_weight * lnlike_wl

    def mcmcInitialGuess(self):
        """Initialize guesses for the MCMC run."""
        self.mcmc_position = np.zeros([self.mcmc_nwalkers,
                                       self.mcmc_ndims])

        for ii, param_0 in enumerate(self.param_ini):
            self.mcmc_position[:, ii] = (
                param_0 + self.param_sig[ii] *
                np.random.randn(self.mcmc_nwalkers)
                )

        return

    def mcmcGetParameters(self, mcmc_samples):
        """
        Computes the 1D marginalized parameter constraints from
        self.mcmcsamples.
        """
        return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                   zip(*np.percentile(mcmc_samples,
                                      [16, 50, 84], axis=0)))

    def mcmcFit(self, verbose=True, nproc=1, **kwargs):
        """
        Peform an MCMC fit to the wp data using the power-law model.

        Parameters:

        -----------
        """
        # TODO Should use HDF5 file to save everything
        # import h5py
        # hfile_name = 'um_smdpl_m100_m10_mcmc.h5'
        # hfile = h5py.File(hfilename, "a")

        # Setup the initial condition
        self.mcmcInitialGuess()

        if nproc > 1:
            from multiprocessing import Pool
            from contextlib import closing

            with closing(Pool(processes=nproc)) as pool:
                mcmc_sampler = emcee.EnsembleSampler(
                    self.mcmc_nwalkers,
                    self.mcmc_ndims,
                    self.lnProb,
                    move=emcee.moves.StretchMove(a=4),
                    pool=pool
                    )

                # Burn-in
                if verbose:
                    print("# Phase: Burn-in ...")
                mcmc_burnin_result = mcmc_sampler.run_mcmc(
                     self.mcmc_position, self.mcmc_nburnin,
                     progress=True
                    )
        else:
            mcmc_sampler = emcee.EnsembleSampler(
                self.mcmc_nwalkers,
                self.mcmc_ndims,
                self.lnProb
                )

            # Burn-in
            if verbose:
                print("# Phase: Burn-in ...")
            mcmc_burnin_result = mcmc_sampler.run_mcmc(
                 self.mcmc_position, self.mcmc_nburnin,
                 progress=True
                 )

        mcmc_burnin_position, _, mcmc_burnin_state = mcmc_burnin_result

        #  Pickle the results
        mcmc_save_results(self.mcmc_burnin_file, mcmc_burnin_result)

        #  Pickle the chain
        mcmc_burnin_chain = mcmc_sampler.chain
        mcmc_save_chains(self.mcmc_burnin_chain_file, mcmc_burnin_chain)

        # Rest the chains
        mcmc_sampler.reset()

        # conv_crit = 3

        # MCMC run
        if verbose:
            print("# Phase: MCMC run ...")
        mcmc_run_result = mcmc_sampler.run_mcmc(
            mcmc_burnin_position,
            self.mcmc_nsamples,
            rstate0=mcmc_burnin_state,
            progress=True
            )

        #  Pickle the result
        self.mcmcSaveResults(self.mcmc_run_file,
                             mcmc_run_result)
        mcmc_run_chain = mcmc_sampler.chain
        self.mcmcSaveChains(self.mcmc_run_chain_file,
                            mcmc_run_chain)

        if verbose:
            print("# Get MCMC samples and best-fit parameters ...")
        # Get the MCMC samples
        mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
            (-1, self.mcmc_ndims)
            )
        #  Save the samples
        np.savez(self.mcmc_run_samples_file, data=mcmc_samples)

        mcmc_lnprob = mcmc_sampler.lnprobability.reshape(-1, 1)

        # Get the best-fit parameters and the 1-sigma error
        mcmc_params_stats = self.mcmcGetParameters(mcmc_samples)
        if verbose:
            print("#------------------------------------------------------")
            print("#  Mean acceptance fraction",
                  np.mean(mcmc_sampler.acceptance_fraction))
            print("#------------------------------------------------------")
            print("#  Best ln(Probability): %11.5f" %
                  np.nanmax(mcmc_lnprob))
            mcmc_best = mcmc_samples[np.argmax(mcmc_lnprob)]
            print(mcmc_best)
            print("#------------------------------------------------------")
            for param_stats in mcmc_params_stats:
                print(param_stats)
            print("#------------------------------------------------------")

        return mcmc_best, mcmc_params_stats, mcmc_samples
