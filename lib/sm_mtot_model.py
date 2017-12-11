""" Module storing functions used to paint M100 values onto model galaxies
"""
import numpy as np


def logms_tot_from_logms_halo(logms_halo, frac_tot_by_halo, log_mass=True):
    """ Calculate logm_tot, total stellar mass of the galaxy (e.g. logms_100kpc),
    from an input total stellar mass ``logms_halo``, in unit of log10(Msun).

    The variable ``logms_halo`` includes central galaxy mass and IHM,
    and also the sum of all stellar mass in the satellites, including IHM contribution
    to the satellites.

    The variable ``frac_tot_by_halo`` can be taken directly
    from UniverseMachine outputs, or modeled independently in some other way, such as
    via a parameterized power law scaling.

    Parameters
    ----------
    logms_halo : float or ndarray
        Numpy array of shape (num_gals, ) storing the total amount of stellar mass
        in each halo, including total satellite galaxy mass

    frac_tot_by_halo : float or ndarray
        For each model galaxy, ``frac_tot_by_halo`` stores the
        fraction of galaxy within 100 kpc scaled by ``logms_halo``.
        In UniverseMachine, ``sm100`` is treated as galaxy mass + IHM,
        while ``logms_halo`` needs to be calculated by additionally
        including contributions from all satellites

    Returns
    -------
    sm100 : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar mass
        within 100 kpc of the halo, including BCG and IHM contributions, in units of Msun

    Examples
    --------
    >>> num_gals = int(1e3)
    >>> logms_halo = np.logspace(10, 12.5, num_gals)
    >>> frac_tot_by_halo = 0.5
    >>> sm100 = logms_tot_from_logms_halo(logms_halo, frac_tot_by_halo)
    """
    if log_mass:
        return (logms_halo + np.log10(frac_tot_by_halo))
    else:
        return (logms_halo * frac_tot_by_halo)
