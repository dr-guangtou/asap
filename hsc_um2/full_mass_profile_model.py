"""
Module storing functions used to paint values of M10, M100 and M*_tot onto
model galaxies
"""

from sm_tot_model import smtot_from_mhalo_log_linear
from sm_m100_model import sm100_from_smtot
from sm_m10_model import sm10_cam


def sm_profile_from_mhalo(log_mhalo,
                          log_mhalo_coeff,
                          normalization_param,
                          random_scatter_in_dex,
                          frac_sm100_by_smtot,
                          frac_sm10_by_sm100,
                          logms_100_data,
                          logms_10_data,
                          logms_100_bins,
                          log_mass=True,
                          ngal_min=25):
    """
    Parameters
    ----------
    mhalo : float or ndarray
        Float or Numpy array of shape (num_gals, ) of the halo mass of the
        galaxy in units of Msun
        (*not* in scaled h=1 units, but instead in "straight up" units
        calculated assuming little h equals the value appropriate for
        its cosmology)

    log_mhalo_coeff : float
        Power law scaling index of smtot with mhalo

    normalization_param : float
        Normalization of the power law scaling between mhalo and smtot

    random_scatter_in_dex : float
        Dispersion of the log-normal random noise added to smtot at fixed mhalo

    frac_sm100_by_smtot: ndarray

    frac_sm10_by_sm100: ndarray

    log_m100_data : ndarray
        Numpy array of shape (num_gals, ) storing log10(M100) of the
        HSC catalog

    log_m10_data : ndarray
        Numpy array of shape (num_gals, ) storing log10(M10) of the
        HSC catalog

    log_m100_bins : array
        Bins for log_m100_model

    ngal_min : int
        Minimum number of galaxy in each M100 bin.
        Default: 25

    Returns
    -------
    log_m10 : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass within 10 kpc aperture

    log_m100 : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass within 100 kpc aperture

    log_mtot : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass in the halo, including BCG, IHL and total satellite galaxy mass
        in units of Msun
    """
    logms_tot = smtot_from_mhalo_log_linear(log_mhalo,
                                            log_mhalo_coeff,
                                            normalization_param,
                                            random_scatter_in_dex,
                                            log_mass=log_mass)

    logms_100 = sm100_from_smtot(logms_tot,
                                 frac_sm100_by_smtot,
                                 log_mass=log_mass)

    mask_m100 = ((logms_100 >= logms_100_bins[0]) &
                 (logms_100 <= logms_100_bins[-1]))

    logms_10 = sm10_cam(logms_100_data,
                        logms_10_data,
                        logms_100[mask_m100],
                        frac_sm10_by_sm100[mask_m100],
                        logms_100_bins,
                        sigma=0,
                        num_required_gals_per_massbin=ngal_min)

    return (logms_10, logms_100,
            logms_tot, mask_m100)
