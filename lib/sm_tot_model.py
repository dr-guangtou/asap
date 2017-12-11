""" Module storing functions used to paint M*_tot values onto model galaxies
"""
import numpy as np


def smtot_from_mhalo_log_linear(log_mhalo,
                                shmr_a,
                                shmr_b,
                                random_scatter_in_dex,
                                log_mass=True):
    r""" Power law model for total stellar mass in a halo as a function of halo mass.
    The output ``smtot`` variable includes stellar mass of the central, the IHM, and
    the also the total galaxy + IHM mass bound in satellites of the halo.


    log_smtot = shmr_a*log_mhalo + shmr_b

    Parameters
    ----------
    mhalo : float or ndarray
        Float or Numpy array of shape (num_gals, ) of the halo mass of the galaxy
        in units of Msun (*not* in scaled h=1 units, but instead in "straight up" units
        calculated assuming little h equals the value appropriate for its cosmology)

    shmr_a : float
        Power law scaling index of smtot with mhalo

    shmr_b : float
        Normalization of the power law scaling between mhalo and smtot

    random_scatter_in_dex : float
        Dispersion of the log-normal random noise added to smtot at fixed mhalo

    Returns
    -------
    smtot : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar mass
        in the halo, including BCG, IHL and total satellite galaxy mass in units of Msun
    """
    log_mhalo = log_mhalo if log_mass else np.log10(log_mhalo)

    return np.random.normal(loc=(shmr_a * log_mhalo + shmr_b),
                            scale=random_scatter_in_dex)
