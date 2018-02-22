"""
Python functions related to stellar mass function.
"""

from __future__ import print_function, division, unicode_literals

from astropy.table import Table
from scipy.stats import norm

import numpy as np

from asap_utils import mtot_minn_weight


__all__ = ['get_smf_bootstrap', 'bootstrap_smf', 'bootstrap_resample',
           'compute_smf', 'ngal_weighted_mass_bin_1d',
           'ngal_weighted_mass_bin_2d',
           'smf_sigma_mass_weighted', 'smf_sigma_mass_weighted_2d']


def compute_smf(sm_array, volume, nb, sm_min, sm_max,
                smf_only=False, return_bins=False):
    """
    Parameters
    ----------
    sm_array: ndarray
        Array of stellar mass values in log10 values

    volume : float
        volume of data in comoving Mpc^-3

    nb : number of bins

    sm_min : min of x axis

    sm_max : max of y axis

    Returns
    -------
    x : ndarray
        x axis of SMF in units of log10 M*

    smf : ndarray in units of dn / dlogM* in units of Mpc^-3 dex^-1

    err : ndarray
        Poisson error
    """

    smf, bin_edges = np.histogram(sm_array, bins=nb,
                                  range=[sm_min, sm_max])

    # bin width in dex
    # !! Only works for constant bin size now
    mass_bin_width = (bin_edges[1] - bin_edges[0])

    # Poison error
    if not smf_only:
        err = np.sqrt(smf)
        # Also normalize the err
        err = (err / volume / mass_bin_width)
        # X-axis
        x = bin_edges[:-1] + (mass_bin_width / 2.0)

    # Normalize
    smf = (smf / volume / mass_bin_width)

    if not smf_only:
        if return_bins:
            return x, smf, err, bin_edges
        return x, smf, err
    # For bootstrap run
    return smf


def bootstrap_resample(X, n_boots=1000):
    """Bootstrap resample an array_like.

    Borrowed from: http://nbviewer.jupyter.org/gist/aflaxman/6871948

    Parameters
    ----------

    X : array_like
      data to resample
    n_boots : int, optional
      Number of bootstrap resamples
      default = 1000

    Results
    -------

    returns X_resamples

    """
    return np.vstack(
        X[np.floor(np.random.rand(len(X))*len(X)).astype(int)]
        for ii in np.arange(n_boots)).T


def bootstrap_smf(sm_array, volume, nb, sm_min, sm_max,
                  n_boots=1000, sm_err=None,
                  resample_err=False):
    """Stellar mass function using bootstrap resampling.

    Parameters
    ----------

    sm_array: ndarray
        Array of stellar mass values in log10 values

    volume : float
        volume of data in comoving Mpc^-3

    nb : number of bins

    sm_min : min of x axis

    sm_max : max of y axis

    sm_err: ndarray, optional
        Array of stellar mass errors


    Returns
    -------

    x : ndarray
        x axis of SMF in units of log10 M*

    smf : ndarray in units of dn / dlogM* in units of Mpc^-3 dex^-1

    err_poison : ndarray
        Poisson error

    smf_boots : ndarray
        Bootstrapped SMFs

    """

    x, smf, err_poison, bins = compute_smf(sm_array, volume, nb,
                                           sm_min, sm_max,
                                           return_bins=True)

    if resample_err:
        msg = "Need to provide the error of stellar mass!"
        assert sm_err is not None, msg
        sm_boots = np.asarray(
            map(lambda mass, err: np.random.normal(mass, err, n_boots),
                sm_array, sm_err))
    else:
        sm_boots = bootstrap_resample(sm_array, n_boots=n_boots)

    smf_boots = np.vstack(
        compute_smf(sm_boots[:, ii], volume, nb, sm_min, sm_max, smf_only=True)
        for ii in range(n_boots)
    )

    return x, smf, err_poison, smf_boots, bins


def get_smf_bootstrap(logms, volume, nbin, min_logms, max_logms,
                      add_err=None, n_boots=5000, bootstrap=True):
    """Estimate the observed SMF and bootstrap errors.

    Parameters
    ----------
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
    smf_boot = bootstrap_smf(logms, volume, nbin,
                             min_logms, max_logms,
                             n_boots=n_boots)
    mass_cen, smf_s, smf_err, smf_b, mass_bins = smf_boot

    # Median values
    if n_boots == 1:
        smf = smf_s
    else:
        smf = np.nanmedian(smf_b, axis=0)
    # 1-sigma errors
    smf_low = np.nanpercentile(smf_b, 16, axis=0,
                               interpolation='midpoint')
    smf_upp = np.nanpercentile(smf_b, 84, axis=0,
                               interpolation='midpoint')

    if add_err is not None:
        smf_err += (smf * add_err)
        smf_low -= (smf * add_err)
        smf_upp += (smf * add_err)

    # Make sure the SMF is above zero
    smf = np.where(smf <= 0.0, 1E-8, smf)
    smf_low = np.where(smf_low <= 0.0, 1E-9, smf_low)
    smf_upp = np.where(smf_upp <= 0.0, 5E-8, smf_upp)

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


def ngal_weighted_mass_bin_1d(logms, sigms, left, right):
    """Weighted count of galaxies in one mass bin."""
    return (norm.sf(left, loc=logms, scale=sigms) -
            norm.sf(right, loc=logms, scale=sigms)).sum()


def ngal_weighted_mass_bin_2d(logms_1, logms_2, sigms, low, upp,
                              left, right):
    """Weigthed count of galaxies in bin using two masses."""
    return np.array(mtot_minn_weight(logms_1, logms_2, sigms,
                                     low, upp, left, right)).sum()


def smf_sigma_mass_weighted(logms, sigms, volume, nbin,
                            min_logms, max_logms,
                            edges=None, return_count=False):
    """Stelar mass function weighted by mass error."""
    # Now only deal with constant bin size
    if edges is None:
        edges = np.linspace(min_logms, max_logms, nbin + 1)

    # Bin width in dex
    bin_width = edges[1:] - edges[0:-1]

    # SMF using weighted count in each bin
    smf = np.array([
        ngal_weighted_mass_bin_1d(logms, sigms, left, right)
        for left, right in zip(edges[0:-1], edges[1:])])

    # Normalize
    if return_count:
        return smf

    return smf / volume / bin_width


def smf_sigma_mass_weighted_2d(logms_1, logms_2, sigms,
                               volume, nbin, min_logms, max_logms,
                               low=11.5, upp=12.5,
                               edges=None, return_count=False):
    """Stelar mass function weighted by mass error."""
    # Now only deal with constant bin size
    if edges is None:
        edges = np.linspace(min_logms, max_logms, nbin + 1)

    # Bin width in dex
    bin_width = edges[1:] - edges[0:-1]

    # SMF using weighted count in each bin
    smf = np.array([
        ngal_weighted_mass_bin_2d(logms_1, logms_2, sigms,
                                  low, upp, left, right)
        for left, right in zip(edges[0:-1], edges[1:])])

    # Normalize
    if return_count:
        return smf

    return smf / volume / bin_width
