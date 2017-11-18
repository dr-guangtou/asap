import os
import sys
import copy
import pickle
from time import time

import emcee
from emcee.utils import MPIPool

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

from forward_model import UMMassProfModel

from astropy.cosmology import Planck15


"""
def sort_on_runtime(pos):
    p = np.atleast_2d(pos)
    idx = np.argsort(p[:, 0])[::-1]
    return p[idx], idx


pool = MPIPool(loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)
"""

kwargs = {'obs_wl_sample': 's16a_m100_m10_boxbin',
          'obs_smf_inn': 's16a_massive_fastlane_pzbright_logm10_smf_default.fits',
          'obs_smf_tot': 's16a_massive_fastlane_pzbright_logm100_smf_default.fits',
          'obs_smf_full': 's82_total_smf_z0.15_0.43.fits',
          'mcmc_nsamples': 10,
          'mcmc_nburnin': 10,
          'mcmc_nwalkers': 10,
          'mcmc_smf_only': False,
          'mcmc_wl_only': False,
          'mcmc_wl_weight': 1.0,
          'sim_cosmo': Planck15,
          'um_cat': 'UM2_multidark_mock_value_added_mpeak_11.5.fits',
          'um_wlcat':'multidark_rockstar_halotools_v0p4_a0.7333_precompute_mpeak_11.5_20bins.npz'
          }

um2 = UMMassProfModel(verbose=True, **kwargs)


class UM2Model(object):

    def __init__(self):
        self.model = True

    def __call__(self, theta):
        lp = um2.lnPrior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + um2.lnLike(theta)


mm = UM2Model()

# Setup the sampler
print("# Setup the sampler ...")
mcmc_sampler = emcee.EnsembleSampler(
    um2.mcmc_nwalkers,
    um2.mcmc_ndims,
    mm, threads=12
    )

# threads=8
"""
    pool=pool,
    runtime_sortingfn=sort_on_runtime
"""

# Setup the initial condition
theta_start = [0.639, 3.18, -0.05, 0.88]
mcmc_position = np.zeros([um2.mcmc_nwalkers,
                          um2.mcmc_ndims])

mcmc_position[:, 0] = (
    theta_start[0] + 5e-2 *
    np.random.randn(um2.mcmc_nwalkers)
    )

mcmc_position[:, 1] = (
    theta_start[1] + 5e-1 *
    np.random.randn(um2.mcmc_nwalkers)
    )

mcmc_position[:, 2] = (
    theta_start[2] + 5e-2 *
    np.random.randn(um2.mcmc_nwalkers)
    )

mcmc_position[:, 3] = (
    theta_start[3] + 1e-2 *
    np.random.randn(um2.mcmc_nwalkers)
    )

# Burn-in
print("# Phase: Burn-in ...")
(mcmc_position, mcmc_prob, mcmc_state) = mcmc_sampler.run_mcmc(
     mcmc_position, um2.mcmc_nburnin
     )

pkl_file = open('um2_m100_m10_mcmc_burnin.pkl', 'wb')
pickle.dump(mcmc_position, pkl_file, -1)
pickle.dump(mcmc_prob, pkl_file, -1)
pickle.dump(mcmc_state, pkl_file, -1)
pkl_file.close()

pkl_file = open('um2_m100_m10_mcmc_burnin_chain.pkl', 'wb')
pickle.dump(mcmc_sampler.chain, pkl_file)
pkl_file.close()

mcmc_sampler.reset()

# MCMC run
print("# Phase: MCMC run ...")
(mcmc_position, mcmc_prob, mcmc_state) = mcmc_sampler.run_mcmc(
    mcmc_position, um2.mcmc_nsamples,
    rstate0=mcmc_state
    )
pkl_file = open('um2_m100_m10_mcmc_run.pkl', 'wb')
pickle.dump(mcmc_position, pkl_file, -1)
pickle.dump(mcmc_prob, pkl_file, -1)
pickle.dump(mcmc_state, pkl_file, -1)
pkl_file.close()

pkl_file = open('um2_m100_m10_mcmc_run_chain.pkl', 'wb')
pickle.dump(mcmc_sampler.chain, pkl_file)
pkl_file.close()

print("# Get MCMC samples and best-fit parameters ...")

# Get the MCMC samples
mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
    (-1, um2.mcmc_ndims)
    )
np.savez('um2_m100_m10_mcmc_samples.npz',
         data=mcmc_samples)

mcmc_lnprob = mcmc_sampler.lnprobability.reshape(-1, 1)

# Get the best-fit parameters and the 1-sigma error
(shmr_a_mcmc, shmr_b_mcmc,
 sigms_a_mcmc, sigms_b_mcmc) = map(
     lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
     zip(*np.percentile(mcmc_samples, [16, 50, 84], axis=0))
     )

print("# ------------------------------------------------------ #")
print("#  Mean acceptance fraction",
      np.mean(mcmc_sampler.acceptance_fraction))
print("# ------------------------------------------------------ #")
print("#  Best ln(Probability): %11.5f" %
      np.nanmax(mcmc_lnprob))
print("# ------------------------------------------------------ #")
print("# logMs,tot = "
      "%7.4f x logMvir + %7.4f" % (shmr_a_mcmc[0],
                                   shmr_b_mcmc[0])
      )
print("#  a Error:  +%7.4f/-%7.4f" % (shmr_a_mcmc[1],
                                      shmr_a_mcmc[2]))
print("#  b Error:  +%7.4f/-%7.4f" % (shmr_b_mcmc[1],
                                      shmr_b_mcmc[2]))
print("# ------------------------------------------------------ #")
print("# sigma(logMs,tot) = "
      "%7.4f x logMvir + %7.4f" % (sigms_a_mcmc[0],
                                   sigms_b_mcmc[0])
      )
print("#  c Error:  +%7.4f/-%7.4f" % (sigms_a_mcmc[1],
                                      sigms_a_mcmc[2]))
print("#  d Error:  +%7.4f/-%7.4f" % (sigms_b_mcmc[1],
                                      sigms_b_mcmc[2]))
print("# ------------------------------------------------------ #")

# pool.close()
