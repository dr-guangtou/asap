""" Module storing functions used to paint M*_tot values onto model galaxies
"""
import numpy as np


def smtot_from_mhalo_log_linear(mhalo,
                                log_mhalo_coeff,
                                normalization_param,
                                random_scatter_in_dex,
                                log_mass=True):
    r""" Power law model for total stellar mass in a halo as a function of halo mass.
    The output ``smtot`` variable includes stellar mass of the central, the IHM, and
    the also the total galaxy + IHM mass bound in satellites of the halo.


    log_smtot = log_mhalo_coeff*log_mhalo + normalization_param

    Parameters
    ----------
    mhalo : float or ndarray
        Float or Numpy array of shape (num_gals, ) of the halo mass of the galaxy
        in units of Msun (*not* in scaled h=1 units, but instead in "straight up" units
        calculated assuming little h equals the value appropriate for its cosmology)

    log_mhalo_coeff : float
        Power law scaling index of smtot with mhalo

    normalization_param : float
        Normalization of the power law scaling between mhalo and smtot

    random_scatter_in_dex : float
        Dispersion of the log-normal random noise added to smtot at fixed mhalo

    Returns
    -------
    smtot : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar mass
        in the halo, including BCG, IHL and total satellite galaxy mass in units of Msun
    """
    if log_mass:
        log_mhalo = mhalo
    else:
        log_mhalo = np.log10(mhalo)

    mean_log_sm = (log_mhalo_coeff * log_mhalo + normalization_param)

    log_smtot = np.random.normal(loc=mean_log_sm,
                                 scale=random_scatter_in_dex)

    if log_mass:
        return log_smtot
    else:
        return (10.0 ** log_smtot)
