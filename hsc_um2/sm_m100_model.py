""" Module storing functions used to paint M100 values onto model galaxies
"""
import numpy as np


def sm100_from_smtot(smtot, frac_sm100_by_smtot, log_mass=True):
    """ Calculate sm100, total stellar mass within 100 kpc,
    from an input total stellar mass ``smtot``,

    The variable ``smtot`` includes central galaxy mass and IHM,
    and also the sum of all stellar mass in the satellites, including IHM contribution
    to the satellites. The variable ``frac_sm100_by_smtot`` can be taken directly
    from UniverseMachine outputs, or modeled independently in some other way, such as
    via a parameterized power law scaling.

    Parameters
    ----------
    smtot : float or ndarray
        Numpy array of shape (num_gals, ) storing the total amount of stellar mass
        in each halo, including total satellite galaxy mass

    frac_sm100_by_smtot : float or ndarray
        For each model galaxy, ``frac_sm100_by_smtot`` stores the
        fraction of galaxy within 100 kpc scaled by ``smtot``.
        In UniverseMachine, ``sm100`` is treated as galaxy mass + IHM,
        while ``smtot`` needs to be calculated by additionally
        including contributions from all satellites

    Returns
    -------
    sm100 : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing the total stellar mass
        within 100 kpc of the halo, including BCG and IHM contributions, in units of Msun

    Examples
    --------
    >>> num_gals = int(1e3)
    >>> smtot = np.logspace(10, 12.5, num_gals)
    >>> frac_sm100_by_smtot = 0.5
    >>> sm100 = sm100_from_smtot(smtot, frac_sm100_by_smtot)
    """
    if log_mass:
        return (smtot + np.log10(frac_sm100_by_smtot))
    else:
        return (smtot * frac_sm100_by_smtot)
