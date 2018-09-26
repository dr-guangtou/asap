"""Module contains different types of priors."""

import numpy as np

from scipy.stats import t as student_t
from scipy.stats import uniform

__all__ = ['student_t_prior', 'student_t_transform',
           'flat_prior', 'flat_prior_transform']


def student_t_prior(param, loc, scale, df=1):
    """Return the natural log prior probability of parameter at `param`.

    Parameters
    ----------
    param : numpy array.
        Array of the parameter values.
    loc : float
        Mean of the distribution.
    scale : float
        Scale of the distribution. Similar to the standard deviation.
    df : int, optional
        Degree of freedom. Default: 1

    Return
    ------
        The natural log of the prior probability at `param` values.

    """
    return np.log(student_t.pdf(param, df=df, loc=loc, scale=scale))


def student_t_transform(unit_arr, loc, scale, df=1):
    """
    Go from a value of the CDF (between 0 and 1) to the corresponding
    parameter value.

    Parameters
    ----------
    unit_arr : numpy array.
        Array of values of the CDF (between 0 and 1).
    loc : float
        Mean of the distribution.
    scale : float
        Scale of the distribution. Similar to the standard deviation.
    df : int, optional
        Degree of freedom. Default: 1

    Return
    ------
        The parameter value corresponding to the value of the CDF given by `unit_arr`.

    """
    return student_t.ppf(unit_arr, df=df, loc=loc, scale=scale)


def flat_prior(param, low, upp):
    """A simple flat or tophat prior for parameters.

    Parameters
    ----------
    param : float or array
        Parameter values.
    low : float
        Lower boundary of the distribution.
    upp : float
        Upper boundary of the distribution.

    Return
    ------
        The natural log of the prior probability at `param` values.

    """
    with np.errstate(divide='ignore'):
        return np.log(uniform.pdf(param, loc=low, scale=(upp - low)))


def flat_prior_transform(unit_arr, low, upp):
    """
    Go from a value of the CDF (between 0 and 1) to the corresponding
    parameter value.

    Parameters
    ----------
    unit_arr : numpy array.
        Array of values of the CDF (between 0 and 1).
    low : float
        Lower boundary of the distribution.
    upp : float
        Upper boundary of the distribution.

    Return
    ------
        The parameter value corresponding to the value of the CDF given by `unit_arr`.

    """
    return uniform.ppf(unit_arr, loc=low, scale=(upp - low))
