""" Module storing functions used to paint M10 values onto model galaxies
"""
import numpy as np

from halotools.empirical_models import conditional_abunmatch


def minn_cam(log10_sm100_data,
             log10_sm10_data,
             log10_sm100_model,
             secondary_haloprop,
             log10_sm100_bins,
             sigma=0,
             num_required_gals_per_massbin=25):
    """
    Conditional Abundance Matching model for inner 10 kpc stellar mass
    from an input 100 kpc aperture stellar mass.

    Parameters
    ----------
    log10_sm100_data : ndarray
        Numpy array of shape (num_gals, ) storing log10(M100) of the HSC
        catalog

    log10_sm10_data : ndarray
        Numpy array of shape (num_gals, ) storing log10(M10) of the HSC catalog

    log10_sm100_model : ndarray
        Numpy array of shape (num_halos, ) storing log10(M100) of the
        model galaxies

    secondary_haloprop : ndarray
        Numpy array of shape (num_halos, ) storing whatever secondary halo
        property is used to do the CAM modeling of M10 in bins of M100,
        e.g., ``secondary_haloprop`` could be the in-situ fraction predicted
        by UniverseMachine

    log10_sm100_bins  ndarray
        Numpy array of shape (num_bin_edges, ) storing the edges of the M100
        bins used to discretize P(M10 | M100) in both data and model during CAM

    sigma : float, optional
        Level of CAM scatter. Default value is zero for a perfect, monotonic
        correspondence between M10 and secondary_haloprop at fixed M100

    num_required_gals_per_massbin : int, optional
        Number of galaxies required in each M100 bin
        to ensure good characterization of P(M10 | M100).
        Default is to require 25 galaxies per bin.

    Returns
    -------
    log10_sm10_model : float or ndarray
        Numpy array of shape (num_halos, ) storing the values of M10 predicted
        by the CAM model
    """
    msg = ("``log10_sm100_bins`` must strictly span the min/max values of"
           "``log10_sm100_model``")
    assert np.all(log10_sm100_model >= log10_sm100_bins[0]), msg
    assert np.all(log10_sm100_model < log10_sm100_bins[-1]), msg

    msg2 = "Only {0} galaxies in the log10_sm100_bins bin ({1:.3f}, {2:.3f})"
    log10_sm10_model = np.zeros_like(secondary_haloprop)

    for low, high in zip(log10_sm100_bins[:-1], log10_sm100_bins[1:]):

        data_mask_ibin = (log10_sm100_data >= low) & (log10_sm100_data < high)
        num_gals_ibin = np.count_nonzero(data_mask_ibin)
        assert num_gals_ibin > num_required_gals_per_massbin, msg2.format(
                num_gals_ibin, low, high)
        data_sm10_ibin = log10_sm10_data[data_mask_ibin]

        model_mask_ibin = ((log10_sm100_model >= low) &
                           (log10_sm100_model < high))
        num_halos_ibin = np.count_nonzero(model_mask_ibin)
        if num_halos_ibin > 0:
            sec_haloprop_ibin = secondary_haloprop[model_mask_ibin]
            log10_sm10_model[model_mask_ibin] = conditional_abunmatch(
                    sec_haloprop_ibin, data_sm10_ibin, sigma=sigma)

    return log10_sm10_model
