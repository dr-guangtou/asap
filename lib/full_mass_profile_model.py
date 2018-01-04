"""
Module storing functions used to paint values of M10, M100 and M*_tot onto
model galaxies
"""

from sm_halo_model import logms_halo_from_logmh_log_linear
from sm_minn_model import logms_inn_cam
from sm_mtot_model import logms_tot_from_logms_halo


def sm_profile_from_mhalo(logmh,
                          shmr_a,
                          shmr_b,
                          random_scatter_in_dex,
                          frac_tot_by_halo,
                          frac_inn_by_tot,
                          logms_tot_obs,
                          logms_inn_obs,
                          logms_tot_bins,
                          log_mass=True,
                          ngal_min=25):
    """
    Parameters
    ----------
    logmh : float or ndarray
        Float or Numpy array of shape (num_gals, ) of the halo mass of the
        galaxy in units of log10(Msun)
        (*not* in scaled h=1 units, but instead in "straight up" units
        calculated assuming little h equals the value appropriate for
        its cosmology)

    shmr_a : float
        Power law scaling index of smtot with mhalo

    shmr_b : float
        Normalization of the power law scaling between mhalo and smtot.

    random_scatter_in_dex : float
        Dispersion of the log-normal random noise added to smtot at fixed
        mhalo.

    frac_tot_by_halo: ndarray
        Fraction of the stellar mass of (central + ICL) to the total
        stellar mass within the halo (including satellites galaxies).
        (e.g., the ones predicted by UniverseMachine model.)

    frac_inn_by_tot: ndarray
        Fraction of the stellar mass in the inner region of galaxy
        to the total stellar mass of the galaxy (central + ICL).
        (e.g., the ones predicted by UniverseMachine model.)

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
    logms_inn_mod : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass in the inner region (e.g., within 10 kpc aperture),
        in unit of log10(Msun).

    logms_tot_mod : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass within a very large aperture (e.g., 100 kpc aperture),
        in unit of log10(Msun).

    logms_halo_mod : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar
        mass in the halo, including central galaxy, ICL and total satellite
        galaxy mass in units of log10(Msun).

    mask_tot : boolen
        Flags for useful items in the predicted stellar masses.
    """
    # Given the prameters for stellar mass halo mass relation, and the
    # random scatter of stellar mass, predict the stellar mass of all
    # galaxies (central + ICL + satellites) in haloes.
    logms_halo_mod = logms_halo_from_logmh_log_linear(logmh,
                                                      shmr_a,
                                                      shmr_b,
                                                      random_scatter_in_dex,
                                                      log_mass=log_mass)

    # Given the modelled fraction of Ms,tot/Ms,halo from UM2,
    # predict the total stellar mass of galaxies (central + ICL).
    logms_tot_mod = logms_tot_from_logms_halo(logms_halo_mod,
                                              frac_tot_by_halo,
                                              log_mass=log_mass)

    # Only keep the ones with Ms,tot within the obseved range.
    mask_tot = ((logms_tot >= logms_tot_bins[0]) &
                (logms_tot <= logms_tot_bins[-1]))

    # Given the modelled fraction of Ms,cen/Ms,tot from UM2,
    # predict the stellar mass in the inner region using
    # conditional abundance matching method.
    logms_inn_mod = logms_inn_cam(logms_tot_obs,
                                  logms_inn_obs,
                                  logms_tot_mod[mask_tot],
                                  frac_inn_by_tot[mask_tot],
                                  logms_tot_bins,
                                  sigma=0,
                                  num_required_gals_per_massbin=ngal_min)

    return (logms_inn_mod, logms_tot_mod,
            logms_halo_mod, mask_tot)
