"""Module contains different types of priors."""

import numpy as np

from scipy.stats import t
from scipy.stats import uniform

__all__ = ['StudentT', 'TopHat']


class TopHat(object):
    """TopHat prior object.

    Based on the `prior` from `prospector`:
        https://github.com/bd-j/prospector/blob/master/prospect/models/priors.py

    """
    def __init__(self, low=0.0, upp=1.0):
        """Constructor.

        Parameters
        ----------
        low : float
            Lower limit of the flat distribution.
        upp : float
            Upper limit of the flat distribution.
        """
        self._low = low
        self._upp = upp

    def lnp(self, x):
        """Compute the value of the probability desnity function at x and
        return the ln of that.

        Parameters
        ----------
        x : float or numpy array
            Parameter values.

        Return
        ------
        lnp : float or numpy array
            The natural log of the prior probability at x
        """
        return uniform.logpdf(x, loc=self.loc, scale=self.scale)

    def unit_transform(self, x):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).

        Return
        ------
            The parameter value corresponding to the value of the CDF given by `unit_arr`.

        """
        return uniform.ppf(x, loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x):
        """Go from the parameter value to the unit coordinate using the cdf.

        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).

        Return
        ------
            The corresponding value in unit coordinate.

        """
        return uniform.cdf(x, loc=self.loc, scale=self.scale)

    def sample(self, nsample):
        """Sample the distribution.

        Parameter
        ---------
        nsample : int
            Number of samples to return.

        Return
        ------
        sample : arr
            `nsample` values that follow the distribution.

        """
        return uniform.rvs(loc=self.loc, scale=self.scale, size=nsample)

    @property
    def low(self):
        """Lower limit of the distribution."""
        return self._low

    @property
    def upp(self):
        """Upper limit of the distribution."""
        return self._upp

    @property
    def scale(self):
        """The `scale` parameter of the distribution."""
        return self._upp - self._low

    @property
    def loc(self):
        """The `loc` parameter of the distribution."""
        return self._low

    @property
    def range(self):
        """The range of the distribution."""
        return (self._low, self._upp)


class StudentT(object):
    """Student-t prior object.

    Based on the `prior` from `prospector`:
        https://github.com/bd-j/prospector/blob/master/prospect/models/priors.py

    """
    def __init__(self, loc=0.0, scale=1.0, df=1):
        """Constructor.

        Parameters
        ----------
        loc : float, optional
            Mean of the distribution. Default: 0.0
        scale : float, optional
            Scale of the flat distribution. Default: 1.0
        df : int, optional
            Degree of freedom. Default: 1
        """
        self._loc = loc
        self._scale = scale
        self._df = df

    def lnp(self, x):
        """Compute the value of the probability desnity function at x and
        return the ln of that.

        Parameters
        ----------
        x : float or numpy array
            Parameter values.

        Return
        ------
        lnp : float or numpy array
            The natural log of the prior probability at x
        """
        return t.logpdf(x, loc=self.loc, scale=self.scale, df=self.df)

    def unit_transform(self, x):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).

        Return
        ------
            The parameter value corresponding to the value of the CDF given by `unit_arr`.

        """
        return t.ppf(x, loc=self.loc, scale=self.scale, df=self.df)

    def inverse_unit_transform(self, x):
        """Go from the parameter value to the unit coordinate using the cdf.

        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).

        Return
        ------
            The corresponding value in unit coordinate.

        """
        return t.cdf(x, loc=self.loc, scale=self.scale, df=self.df)

    def sample(self, nsample, limit=False):
        """Sample the distribution.

        Parameter
        ---------
        nsample : int
            Number of samples to return.
        limit : bool, optional
            Whether to limit the random variables within the range.

        Return
        ------
        sample : arr
            `nsample` values that follow the distribution.

        """
        if not limit:
            return t.rvs(loc=self.loc, scale=self.scale, df=self.df,
                         size=nsample)
        else:
            sample = []
            while len(sample) < nsample:
                rv = t.rvs(df=self.df, loc=self.loc, scale=self.scale)
                if self.low < rv <= self.upp:
                    sample.append(rv)

            return np.asarray(sample)

    @property
    def scale(self):
        """The `scale` parameter of the distribution."""
        return self._scale

    @property
    def loc(self):
        """The `loc` parameter of the distribution."""
        return self._loc

    @property
    def df(self):
        """The `df` parameter of the distribution."""
        return self._df

    @property
    def low(self):
        """Lower limit of the distribution."""
        return self.loc - 5.0 * self.scale

    @property
    def upp(self):
        """Upper limit of the distribution."""
        return self.loc + 5.0 * self.scale

    @property
    def range(self):
        """The range of the distribution."""
        return (self.low, self.upp)
