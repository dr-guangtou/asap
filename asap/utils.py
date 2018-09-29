"""Utilities for A.S.A.P model."""
from __future__ import print_function, division, unicode_literals

import copy
import pickle

from astropy.table import Column, vstack
from scipy.stats import mvn, norm

import numpy as np


__all__ = ["mcmc_save_results", "mcmc_samples_stats", "mcmc_load_results",
           "mass_gaussian_weight_2d", "mass_gaussian_weight",
           "rank_splitting_sample"]


def mcmc_samples_stats(mcmc_samples):
    """1D marginalized parameter constraints."""
    return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
               zip(*np.percentile(mcmc_samples,
                                  [16, 50, 84], axis=0)))


def mcmc_load_results(mcmc_file):
    """Retrieve the MCMC results from .npz file."""
    mcmc_data = np.load(mcmc_file)

    return (mcmc_data['samples'], mcmc_data['chains'],
            mcmc_data['lnprob'], mcmc_data['best'],
            mcmc_data['position'], mcmc_data['acceptance'])


def mcmc_save_results(mcmc_results, mcmc_sampler, mcmc_file,
                      mcmc_ndims, verbose=True):
    """Save the MCMC run results."""
    (mcmc_position, mcmc_lnprob, _) = mcmc_results

    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
        (-1, mcmc_ndims))
    mcmc_chains = mcmc_sampler.chain
    mcmc_lnprob = mcmc_sampler.lnprobability
    ind_1, ind_2 = np.unravel_index(np.argmax(mcmc_lnprob, axis=None),
                                    mcmc_lnprob.shape)
    mcmc_best = mcmc_chains[ind_2, ind_1, :]
    mcmc_params_stats = mcmc_samples_stats(mcmc_samples)

    np.savez(mcmc_file,
             samples=mcmc_samples, lnprob=np.array(mcmc_lnprob),
             best=np.array(mcmc_best), chains=mcmc_chains,
             position=np.asarray(mcmc_position),
             acceptance=np.array(mcmc_sampler.acceptance_fraction))

    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(mcmc_sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" % np.max(mcmc_lnprob))
        print(mcmc_best)
        print("#------------------------------------------------------")
        for param_stats in mcmc_params_stats:
            print(param_stats)
        print("#------------------------------------------------------")

    return


def mass_gaussian_weight_2d(logms1, logms2, sigms1, sigms2,
                            bin1_l, bin1_r, bin2_l, bin2_r):
    """Weight of galaaxy using two stellar masses."""
    p, _ = mvn.mvnun([bin1_l, bin2_l],
                     [bin1_r, bin2_r],
                     [logms1, logms2],
                     [[sigms1 ** 2, sigms1 * sigms2],
                      [sigms2 * sigms1, sigms2 ** 2]])

    return p


def mtot_minn_weight(logm_tot, logm_inn, sig,
                     mtot_0, mtot_1, minn_0, minn_1):
    """Two-dimensional weight of galaxy in Mtot-Minn box."""
    return [mass_gaussian_weight_2d(m1, m2, ss, ss,
                                    mtot_0, mtot_1, minn_0, minn_1)
            for m1, m2, ss in zip(logm_tot, logm_inn, sig)]


def mass_gaussian_weight(logms, sigms, left, right):
    """Weights of stellar in bin."""
    return (norm.sf(left, loc=logms, scale=sigms) -
            norm.sf(right, loc=logms, scale=sigms))


def rank_splitting_sample(cat, X_col, Y_col, n_bins=5, n_sample=2,
                          X_min=None, X_max=None, X_bins=None,
                          id_each_bin=True):
    """Split sample into N_sample with fixed distribution in X, but different
    rank orders in Y.

    Parameters:
    -----------
    cat : astropy.table
        Table for input catalog
    X_col : string
        Name of the column for parameter that should have fixed distribution
    Y_col : string
        Name of the column for parameter that need to be split
    n_bins : int
        Number of bins in X
    n_sample : int
        Number of bins in Y
    X_min : float
        Minimum value of X for the binning
    X_max: float
        Maximum value of X for the binning
    X_bins : array
        Edges of X bins, provided by the users
        Usefull for irregular binnings

    Return
    ------

    """
    data = copy.deepcopy(cat)

    X = data[X_col]
    X_len = len(X)
    if X_bins is None:
        if X_min is None:
            X_min = np.nanmin(X)
        if X_max is None:
            X_max = np.nanmax(X)

        msg = '# Sample size should be much larger than number of bins in X'
        assert X_len > (2 * n_bins), msg

        X_bins = np.linspace(X_min, X_max, (n_bins + 1))
    else:
        n_bins = (len(X_bins) - 1)

    # Place holder for sample ID
    data.add_column(Column(data=(np.arange(X_len) * 0),
                           name='sample_id'))
    data.add_column(Column(data=np.arange(X_len), name='index_ori'))

    # Create index array for object in each bin
    X_idxbins = np.digitize(X, X_bins, right=True)

    bin_list = []
    for ii in range(n_bins):
        subbin = data[X_idxbins == (ii + 1)]
        subbin.sort(Y_col)

        subbin_len = len(subbin)
        subbin_size = int(np.ceil(subbin_len / n_sample))

        idx_start, idx_end = 0, subbin_size
        for jj in range(n_sample):
            if idx_end > subbin_len:
                idx_end = subbin_len
            if id_each_bin:
                subbin['sample_id'][idx_start:idx_end] = ((jj + 1) +
                                                          (ii * n_sample))
            else:
                subbin['sample_id'][idx_start:idx_end] = (jj + 1)
            idx_start = idx_end
            idx_end += subbin_size

        bin_list.append(subbin)

    new_data = vstack(bin_list)

    return new_data
