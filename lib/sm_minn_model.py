""" Module storing functions used to paint M10 values onto model galaxies
"""
import numpy as np

from halotools.empirical_models import conditional_abunmatch


def logms_inn_cam(logms_tot_obs,
                  logms_inn_obs,
                  logms_tot_mod,
                  secondary_haloprop,
                  logms_tot_bins,
                  sigma=0,
                  num_required_gals_per_massbin=25):
    """
    Conditional Abundance Matching model for inner region stellar mass
    from an input large aperture stellar mass.

    Parameters
    ----------
    logms_tot_obs : ndarray
        Numpy array of shape (num_gals, ) storing log10(Mtot) of the HSC
        catalog

    logms_inn_obs : ndarray
        Numpy array of shape (num_gals, ) storing log10(Minn) of the HSC
        catalog

    logms_tot_mod : ndarray
        Numpy array of shape (num_halos, ) storing log10(Mtot) of the
        model galaxies

    secondary_haloprop : ndarray
        Numpy array of shape (num_halos, ) storing whatever secondary halo
        property is used to do the CAM modeling of Minn in bins of Mtot,
        e.g., ``secondary_haloprop`` could be the in-situ fraction predicted
        by UniverseMachine

    logms_tot_bins  ndarray
        Numpy array of shape (num_bin_edges, ) storing the edges of the Mtot
        bins used to discretize P(Minn | Mtot) in both data and model during
        CAM

    sigma : float, optional
        Level of CAM scatter. Default value is zero for a perfect, monotonic
        correspondence between Minn and secondary_haloprop at fixed M100

    num_required_gals_per_massbin : int, optional
        Number of galaxies required in each M100 bin
        to ensure good characterization of P(Minn | Mtot).
        Default is to require 25 galaxies per bin.

    Returns
    -------
    logms_inn_mod : float or ndarray
        Numpy array of shape (num_halos, ) storing the values of M10 predicted
        by the CAM model
    """
    msg = ("``logms_tot_bins`` must strictly span the min/max values of"
           "``logms_tot_mod``")
    assert np.all(logms_tot_mod >= logms_tot_bins[0]), msg
    assert np.all(logms_tot_mod < logms_tot_bins[-1]), msg

    msg2 = "Only {0} galaxies in the logms_tot_bins bin ({1:.3f}, {2:.3f})"
    logms_inn_mod = np.zeros_like(secondary_haloprop)

    for low, high in zip(logms_tot_bins[:-1], logms_tot_bins[1:]):

        data_mask_ibin = (logms_tot_obs >= low) & (logms_tot_obs < high)
        num_gals_ibin = np.count_nonzero(data_mask_ibin)
        assert num_gals_ibin > num_required_gals_per_massbin, msg2.format(
                num_gals_ibin, low, high)
        logms_inn_ibin = logms_inn_obs[data_mask_ibin]

        model_mask_ibin = ((logms_tot_mod >= low) &
                           (logms_tot_mod < high))
        num_halos_ibin = np.count_nonzero(model_mask_ibin)

        if num_halos_ibin > 0:
            sec_haloprop_ibin = secondary_haloprop[model_mask_ibin]
            logms_inn_mod[model_mask_ibin] = conditional_abunmatch(
                    sec_haloprop_ibin, logms_inn_ibin, sigma=sigma)

    return logms_inn_mod
