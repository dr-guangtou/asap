"""
Parameterized models of the stellar mass - halo mass relation (SMHM).

"""

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import os

import numpy as np

from astropy.table import Table

__all__ = ['behroozi10_ms_to_mh', 'behroozi10_evolution',
           'leauthaud12_ms_to_mh', 'moster13_mh_to_ms',
           'moster13_ms_mh_ratio', 'moster13_evolution',
           'behroozi13_f', 'behroozi13_mh_to_ms',
           'behroozi13_evolution', 'behroozi_mh_to_ms_icl',
           'puebla15_mh_to_ms', 'vanuitert16_mh_to_ms',
           'puebla17_p', 'puebla17_q', 'puebla17_g',
           'puebla17_evolution', 'puebla17_mh_to_ms',
           'puebla17_ms_to_mh', 'shan17_ms_to_mh',
           'tinker17_shmr', 'kravtsov18_m500_to_mbcg',
           'kravtsov18_mh_to_ms', 'moster18_mh_to_ms',
           'moster18_ms_mh_ratio', 'moster18_evolution',
           'small_h_corr', 'imf_corr_to_chab',
           'sps_corr_to_bc03', 'm500c_to_m200c']

DATA_DIR = '/Users/song/Dropbox/work/project/hsc_massive/hsc_massive/data/shmr'


def behroozi10_ms_to_mh(logms, mh_1=12.35, ms_0=10.72, beta=0.44,
                        delta=0.57, gamma=1.56, redshift=None,
                        **kwargs):
    """Halo mass from stellar mass based on Behroozi+10.

    Parameters:
        mh_1:  Characteristic halo mass (log10)
        ms_0:  Characteristic stellar mass (log10)
        beta:  Faint-end power law
        delta: Massive-end power law
        gamma: Transition width between faint- and massive-end relations

    Redshift evolution:
        When `redshift` is not `None`, will use `behroozi10_evolution`
        function to get the best-fit parameter at desired redshift.
    """
    if redshift is not None:
        mh_1, ms_0, beta, delta, gamma = behroozi10_evolution(redshift, **kwargs)

    mass_ratio  = (10.0 ** logms) / (10.0 ** ms_0)

    term_1 = np.log10(mass_ratio) * beta
    term_2 = mass_ratio ** delta
    term_3 = (mass_ratio ** -gamma) + 1.0

    return mh_1 + term_1 + (term_2 / term_3) - 0.50

def behroozi10_evolution(redshift, free_mu_kappa=True):
    """Parameterize the evolution in term of scale factor.

    Using the best-fit parameters in Behroozi10.

    The default parameter works for 0 < z < 1, and assume
    free (mu, kappa) parameters about the shifting of SMF
    at different redshifts.
    """
    scale_minus_one = -redshift / (1.0 + redshift)

    if free_mu_kappa:
        if redshift <= 1.0:
            """ Free mu, kappa; 0<z<1
            mh_1_0=12.35+0.07-0.16, mh_1_a=0.28+0.19-0.97
            ms_0_0=10.72+0.22-0.29, ms_0_a=0.55+0.18-0.79
            beta_0=0.44+0.04-0.06, beta_a=0.18+0.08-0.34
            delta_0=0.57+0.15-0.06, delta_a=0.17+0.42-0.41
            gamma_0=1.56+0.12-0.38, gamma_a=2.51+0.15-1.83
            """
            mh_1_0, mh_1_a = 12.35, 0.28
            ms_0_0, ms_0_a = 10.72, 0.55
            beta_0, beta_a = 0.44, 0.18
            delta_0, delta_a = 0.57, 0.17
            gamma_0, gamma_a = 1.56, 2.51
        elif redshift > 4.0:
            raise Exception("# Only works for z < 4.0")
        else:
            """ Free mu, kappa; 0.8<z<4.0
            mh_1_0=12.27+0.59-0.27, mh_1_a=-0.84+0.87-0.58
            ms_0_0=11.09+0.54-0.31, ms_0_a=0.56+0.89-0.44
            beta_0=0.65+0.26-0.20, beta_a=0.31+0.38-0.47
            delta_0=0.56+1.33-0.29, delta_a=-0.12+0.76-0.50
            gamma_0=1.12+7.47-0.36, gamma_a=-0.53+7.87-2.50
            """
            mh_1_0, mh_1_a = 12.27, -0.84
            ms_0_0, ms_0_a = 11.09, 0.56
            beta_0, beta_a = 0.65, 0.31
            delta_0, delta_a = 0.56, -0.12
            gamma_0, gamma_a = 1.12, -0.53
    else:
        if redshift > 1:
            raise Exception("# Only works for z < 1.0")
        else:
            """ mu = kappa = 0; 0<z<1
            mh_1_0=12.35+0.02-0.15, mh_1_a=0.30+0.14-1.02
            ms_0_0=10.72+0.02-0.12, ms_0_a=0.59+0.15-0.85
            beta_0=0.43+0.01-0.05, beta_a=0.18+0.06-0.34
            delta_0=0.56+0.14-0.05, delta_a=0.18+0.41-0.42
            gamma_0=1.54+0.03-0.40, gamma_a=2.52+0.03-1.89
            """
            mh_1_0, mh_1_a = 12.35, 0.30
            ms_0_0, ms_0_a = 10.72, 0.59
            beta_0, beta_a = 0.43, 0.18
            delta_0, delta_a = 0.56, 0.18
            gamma_0, gamma_a = 1.54, 2.52

    mh_1 = mh_1_0 + mh_1_a * scale_minus_one
    ms_0 = ms_0_0 + ms_0_a * scale_minus_one
    beta = beta_0 + beta_a * scale_minus_one
    delta = delta_0 + delta_a * scale_minus_one
    gamma = gamma_0 + gamma_a * scale_minus_one

    return mh_1, ms_0, beta, delta, gamma

def leauthaud12_ms_to_mh(logms, mh_1=12.520, ms_0=10.916, beta=0.457, delta=0.566,
                         gamma=1.53, redshift=None, sigmod=1):
    """Halo mass from stellar mass based on Leauthaud+2012."""
    if redshift is not None:
        if 0.22 <= redshift < 0.48:
            if sigmod == 1:
                mh_1, ms_0, beta, delta, gamma = 12.520, 10.916, 0.457, 0.566, 1.53
            elif sigmod == 2:
                mh_1, ms_0, beta, delta, gamma = 12.518, 10.917, 0.456, 0.582, 1.48
            else:
                raise Exception("# Wrong sig_mod ! Options are [1, 2]")
        elif 0.48 <= redshift < 0.74:
            if sigmod == 1:
                mh_1, ms_0, beta, delta, gamma = 12.725, 11.038, 0.466, 0.610, 1.95
            elif sigmod == 2:
                mh_1, ms_0, beta, delta, gamma = 12.724, 11.038, 0.466, 0.620, 1.93
            else:
                raise Exception("# Wrong sig_mod ! Options are [1, 2]")
        elif 0.74 <= redshift < 1.0:
            if sigmod == 1:
                mh_1, ms_0, beta, delta, gamma = 12.722, 11.100, 0.470, 0.393, 2.51
            elif sigmod == 2:
                mh_1, ms_0, beta, delta, gamma = 12.726, 11.100, 0.470, 0.470, 2.38
            else:
                raise Exception("# Wrong sig_mod ! Options are [1, 2]")
        else:
            raise Exception("# Wrong redshift range ! Should be between [0, 1]")

    return behroozi10_ms_to_mh(logms, mh_1=mh_1, ms_0=ms_0, beta=beta,
                               delta=delta, gamma=gamma)

def moster13_mh_to_ms(logmh, mh_1=11.59, n=0.0351, beta=1.376, gamma=0.608,
                      redshift=None):
    """Stellar mass from halo mass based on Moster et al. 2013."""
    ms_ratio = moster13_ms_mh_ratio(logmh, mh_1=mh_1, n=n, beta=beta, gamma=gamma,
                                    redshift=redshift)

    return logmh + np.log10(ms_ratio)

def moster13_ms_mh_ratio(logmh, mh_1=11.59, n=0.0351, beta=1.376, gamma=0.608,
                         redshift=None):
    """Stellar-to-halo mass ratio based on Moster et al. 2013."""
    if redshift is not None:
        mh_1, n, beta, gamma = moster13_evolution(redshift)

    print(mh_1, n, beta, gamma)

    mass_ratio = 10.0 ** logmh / 10.0 ** mh_1

    term1 = 2.0 * n
    term2 = mass_ratio ** -beta
    term3 = mass_ratio ** gamma

    return term1 / (term2 + term3)

def moster13_evolution(redshift, m10=11.59, m11=1.195, n10=0.0351, n11=-0.0247,
                       beta10=1.376, beta11=-0.826, gamma10=0.608, gamma11=0.329):
    """Redshift dependent of parameters in Moster et al. 2013 model.

    Best-fit parameter:

    M10, M11: 11.59+/-0.236, 1.195+/-0.353
    N10, N11: 0.0351+/- 0.0058, -0.0247+/-0.0069
    beta10, beta11: 1.376+/-0.153, -0.826+/-0.225
    gamma10, gamma11: 0.608+/-0.059, 0.329+/-0.173
    """
    z_factor = redshift / (1.0 + redshift)

    mh_1 = m10 + m11 * z_factor
    n = n10 + n11 * z_factor
    beta = beta10 + beta11 * z_factor
    gamma = gamma10 + gamma11 * z_factor

    return mh_1, n, beta, gamma

def behroozi13_f(x, alpha, delta, gamma):
    """The f(x) function used in Behroozi+13."""
    term_1 = -1.0 * np.log10(10.0 ** (alpha * x) + 1.0)

    term_2 = delta * (np.log10(1.0 + np.exp(x)) ** gamma) / (1.0 + np.exp(10.0 ** -x))

    return term_1 + term_2

def behroozi13_mh_to_ms(logmh, mh_1=11.514, epsilon=-1.777,
                        alpha=-1.412, delta=3.508, gamma=0.316,
                        redshift=None, **kwargs):
    """Stellar mass from halo mass based on Behroozi et al. 2013.

    Parameters:
        mh_1:     Characteristic halo mass (log10)
        epsilon:  Characteristic stellar mass to halo mass ratio (log10)
        alpha:    Faint-end slope of SMHM relation
        delta:    Strength of subpower law at massive end of SMHM relation
        gamma:    Index of subpower law at massive end of SMHM relation

    Redshift evolution:
        When `redshift` is not `None`, will use `behroozi13_evolution`
        function to get the best-fit parameter at desired redshift.
    """
    if redshift is not None:
        mh_1, epsilon, alpha, delta, gamma = behroozi15_evolution(redshift, **kwargs)

    mhalo_ratio = logmh - mh_1

    return mh_1 + epsilon + (behroozi13_f(mhalo_ratio, alpha, delta, gamma) -
                             behroozi13_f(0.0, alpha, delta, gamma))

def behroozi13_evolution(redshift):
    """Parameterize the evolution in term of scale factor.

    Using the best-fit parameters in Behroozi15.

    The default parameter works for 0 < z < 1, and assume
    free (mu, kappa) parameters about the shifting of SMF
    at different redshifts.
    """
    scale = 1.0 / (1.0 + redshift)
    scale_minus_one = -redshift / (1.0 + redshift)

    # mh_1_0 = 11.514 + 0.053 - 0.009
    # mh_1_a = -1.793 + 0.315 - 0.330
    # mh_1_z = -0.251 + 0.012 - 0.125
    mh_1_0, mh_1_a, mh_1_z = 11.514, -1.793, -0.251

    # epsilon_0 = -1.777 + 0.133 - 0.146
    # epsilon_a = -0.006 + 0.113 - 0.361
    # epsilon_z = -0.000 + 0.003 - 0.104
    # epsilon_a_2 = -0.119 + 0.061 - 0.012
    epsilon_0, epsilon_a = -1.777, -0.006
    epsilon_z, epsilon_a_2 = -0.000, -0.119

    # alpha_0 = -1.412 + 0.020 - 0.105
    # alpha_a =  0.731 + 0.344 - 0.296
    alpha_0, alpha_a = -1.412, 0.731

    # delta_0 = 3.508 + 0.087 - 0.369
    # delta_a = 2.608 + 2.446 - 1.261
    # delta_z = -0.043 + 0.958 - 0.071
    delta_0, delta_a, delta_z = 3.508, 2.608, -0.043

    # gamma_0 = 0.316 + 0.076 - 0.012
    # gamma_a = 1.319 + 0.584 - 0.505
    # gamma_z = 0.279 + 0.256 - 0.081
    gamma_0, gamma_a, gamma_z = 0.316, 1.319, 0.279

    nu_a = np.exp(-4.0 * (scale ** 2.0))

    mh_1 = mh_1_0 + ((mh_1_a * scale_minus_one) + mh_1_z * redshift) * nu_a
    epsilon = epsilon_0 + ((epsilon_a * scale_minus_one) + epsilon_z * redshift) + epsilon_a_2 * scale_minus_one
    alpha = alpha_0 + (alpha_a * scale_minus_one) * nu_a
    delta = delta_0 + (delta_a * scale_minus_one + delta_z * redshift) * nu_a
    gamma = gamma_0 + (gamma_a * scale_minus_one + gamma_z * redshift) * nu_a

    return mh_1, epsilon, alpha, delta, gamma

def behroozi_mh_to_ms_icl(loc=DATA_DIR):
    """SHMR with ICL included.

    Only for redshift at 0.2, 0.3, 0.4, 0.5.
    """
    b13_icl_z1= Table.read(
        os.path.join(loc, 'behroozi_2013/smhm_z0.2.dat'),
        format='ascii')
    b13_icl_z2= Table.read(
        os.path.join(loc, 'behroozi_2013/smhm_z0.3.dat'),
        format='ascii')
    b13_icl_z3= Table.read(
        os.path.join(loc, 'behroozi_2013/smhm_z0.4.dat'),
        format='ascii')
    b13_icl_z4= Table.read(
        os.path.join(loc, 'behroozi_2013/smhm_z0.5.dat'),
        format='ascii')

    return b13_icl_z1, b13_icl_z2, b13_icl_z3, b13_icl_z4

def puebla15_mh_to_ms(logmh, mh_1=11.367, epsilon=-2.143, alpha=-2.858,
                      delta=6.026, gamma=0.303, kind=None):
    """Stellar mass from halo mass based on Rodriguez-Puebla et al. 2015.

    Default results are for red central galaxy.
    """
    if kind == 'red':
        """
        mh_1    = 11.361 +/- 0.100
        epsilon = -2.143 +/- 0.086
        alpha   = -2.858 +/- 0.479
        delta   =  6.026 +/- 0.544
        gamma   =  0.303 +/- 0.023
        """
        mh_1, epsilon, alpha, delta, gamma = 11.361, -2.143, -2.858, 6.026, 0.303
    elif kind == 'blue':
        """
        mh_1    = 11.581 +/- 0.034
        epsilon = -1.593 +/- 0.042
        alpha   = -1.500 +/- 0.148
        delta   =  4.293 +/- 0.271
        gamma   =  0.396 +/- 0.035
        """
        mh_1, epsilon, alpha, delta, gamma = 11.581, -1.593, -1.500, 4.293, 0.396
    else:
        raise Exception("# Wrong kind: [red / blue]")

    return behroozi13_mh_to_ms(logmh, mh_1=mh_1, epsilon=epsilon,
                               alpha=alpha, delta=delta, gamma=gamma)

def vanuitert16_mh_to_ms(logmh, mh_1=12.06, ms_0=11.16, beta1=5.4, beta2=0.15,
                         all=False, sat=False):
    """Stellar mass based on halo mass from van Uitert et al. 2016.

         logmh_1         logms_0          beta_1       beta_2
    All  10.97+0.34-0.25 10.58+0.22-0.15  7.5+3.8-2.7  0.25+0.04-0.06
    Cen  12.06+0.72-0.80 11.16+0.40-0.62  5.4+5.3-3.4  0.15+0.31-0.14
    Sat  11.70+0.70-0.84 11.22+0.12-0.22  4.5+4.6-2.9  0.05+0.07-0.04
    """
    if all:
        mh_1, ms_0, beta1, beta2 = 10.97, 10.58, 7.5, 0.25

    if sat:
        mh_1, ms_0, beta1, beta2 = 11.70, 11.22, 4.5, 0.05

    mass_ratio = 10.0 ** logmh / 10.0 ** mh_1

    term1 = 10.0 ** ms_0
    term2 = mass_ratio ** beta1 / (1.0 + mass_ratio) ** (beta1 - beta2)

    return np.log10(term1 * term2)

def puebla17_p(x, y, z):
    """The P(x, y, z) function used in Rodriguez-Puebla+17."""
    return y * z - (x * z) / (1.0 + z)

def puebla17_q(z):
    """The Q(z) function used in Rodriguez-Puebla+17."""
    return np.exp(-4.0 / (1.0 + z) ** 2.0)

def puebla17_g(x, alpha, delta, gamma):
    """The g(x) function used in Behroozi+13."""
    term_1 = -np.log10(10.0 ** (-alpha * x) + 1.0)

    term_2 = delta * (np.log10(1.0 + np.exp(x)) ** gamma) / (1.0 + np.exp(10.0 ** -x))

    return term_1 + term_2

def puebla17_evolution(redshift):
    """Parameterize the evolution in term of scale factor.

    Using the best-fit parameters in Rodriguez-Puebla+17.
    """
    # mh_1_0 = 11.548 +/- 0.049
    # mh_1_1 = -1.297 +/- 0.225
    # mh_1_2 = -0.026 +/- 0.043
    mh_1_0, mh_1_1, mh_1_2 = 11.548, -1.297, -0.026

    # epsilon_0 = -1.758 +/- 0.040
    # epsilon_1 =  0.110 +/- 0.166
    # epsilon_2 = -0.061 +/- 0.029
    # epsilon_3 = -0.023 +/- 0.009
    epsilon_0, epsilon_1 = -1.758, 0.110
    epsilon_2, epsilon_3 = -0.061, -0.023

    # alpha_0 = 1.975 +/- 0.074
    # alpha_1 = 0.714 +/- 0.165
    # alpha_2 = 0.042 +/- 0.017
    alpha_0, alpha_1, alpha_2 = 1.975, 0.714, 0.042

    # delta_0 =  3.390 +/- 0.281
    # delta_1 = -0.472 +/- 0.899
    # detla_2 = -0.931 +/- 0.147
    delta_0, delta_1, delta_2 = 3.390, -0.472, -0.931

    # gamma_0 =  0.498 +/- 0.044
    # gamma_1 = -0.157 +/- 0.122
    gamma_0, gamma_1 = 0.498, -0.157

    mh_1 = mh_1_0 + puebla17_p(mh_1_1, mh_1_2, redshift) * puebla17_q(redshift)
    epsilon = epsilon_0 + (puebla17_p(epsilon_1, epsilon_2, redshift) * puebla17_q(redshift) +
                           puebla17_p(epsilon_3, 0.0, redshift))
    alpha = alpha_0 + puebla17_p(alpha_1, alpha_2, redshift) * puebla17_q(redshift)
    delta = delta_0 + puebla17_p(delta_1, delta_2, redshift) * puebla17_q(redshift)
    gamma = gamma_0 + puebla17_p(gamma_1, 0.0, redshift) * puebla17_q(redshift)

    return mh_1, epsilon, alpha, delta, gamma

def puebla17_mh_to_ms(logmh, mh_1=11.514, epsilon=-1.777,
                      alpha=-1.412, delta=3.508, gamma=0.316,
                      redshift=None, **kwargs):
    """Stellar mass from halo mass based on Puebla et al. 2017.

    Parameters:
        mh_1:     Characteristic halo mass (log10)
        epsilon:  Characteristic stellar mass to halo mass ratio (log10)
        alpha:    Faint-end slope of SMHM relation
        delta:    Strength of subpower law at massive end of SMHM relation
        gamma:    Index of subpower law at massive end of SMHM relation

    Redshift evolution:
        When `redshift` is not `None`, will use `puebla17_evolution`
        function to get the best-fit parameter at desired redshift.
    """
    if redshift is not None:
        mh_1, epsilon, alpha, delta, gamma = puebla17_evolution(redshift, **kwargs)

    mhalo_ratio = logmh - mh_1

    print(mh_1, epsilon, alpha, delta, gamma)

    return mh_1 + epsilon + (puebla17_g(mhalo_ratio, alpha, delta, gamma) -
                             puebla17_g(0.0, alpha, delta, gamma))

def puebla17_ms_to_mh(logms, mh_1=12.58, ms_0=10.90, beta=0.48, delta=0.29, gamma=1.52,
                      redshift=None):
    """Halo mass from stellar mass based on Rodriguez-Puebla et al. 2017."""
    if redshift is not None:
        if 0.0 < redshift <= 0.20:
            mh_1, ms_0, beta, delta, gamma = 12.58, 10.90, 0.48, 0.29, 1.52
        elif 0.20 < redshift <= 0.40:
            mh_1, ms_0, beta, delta, gamma = 12.61, 10.93, 0.48, 0.27, 1.46
        elif 0.40 < redshift <= 0.60:
            mh_1, ms_0, beta, delta, gamma = 12.68, 10.99, 0.48, 0.23, 1.39
        elif 0.60 < redshift <= 0.90:
            mh_1, ms_0, beta, delta, gamma = 12.77, 11.08, 0.50, 0.18, 1.33
        elif 0.90 < redshift <= 1.20:
            mh_1, ms_0, beta, delta, gamma = 12.89, 11.19, 0.51, 0.12, 1.27
        elif 1.20 < redshift <= 1.40:
            mh_1, ms_0, beta, delta, gamma = 13.01, 11.31, 0.53, 0.03, 1.22
        elif 1.40 < redshift <= 1.60:
            mh_1, ms_0, beta, delta, gamma = 13.15, 11.47, 0.54, -0.10, 1.17
        elif 1.60 < redshift <= 1.80:
            mh_1, ms_0, beta, delta, gamma = 13.33, 11.73, 0.55, -0.34, 1.16
        else:
            raise Exception("# Wrong redshift range: [0.0, 1.8]")

    mass_ratio  = (10.0 ** logms) / (10.0 ** ms_0)

    term_1 = np.log10(mass_ratio) * beta
    term_2 = mass_ratio ** delta
    term_3 = (mass_ratio ** -gamma) + 1.0

    return mh_1 + term_1 + (term_2 / term_3) - 0.50

def shan17_ms_to_mh(logms, mh_1=12.52, ms_0=10.98, beta=0.47, delta=0.55,
                    gamma=1.43, redshift=None):
    """Halo mass from stellar mass based on Shan+2017."""
    if redshift is not None:
        if 0.2 <= redshift < 0.4:
            mh_1, ms_0, beta, delta, gamma = 12.52, 10.98, 0.47, 0.55, 1.43
        elif 0.4 <= redshift < 0.6:
            mh_1, ms_0, beta, delta, gamma = 12.70, 11.11, 0.50, 0.54, 1.72
        else:
            raise Exception("# Wrong redshift range ! Should be between [0, 1]")

    return behroozi10_ms_to_mh(logms, mh_1=mh_1, ms_0=ms_0, beta=beta,
                               delta=delta, gamma=gamma)

def tinker17_shmr(loc=DATA_DIR):
    """SHMR from Tinker+2017."""
    tinker17_mh_to_ms = Table.read(
        os.path.join(loc, 'tinker_2017/tinker2017_mh_to_ms.txt'),
        format='ascii')
    tinker17_ms_to_mh = Table.read(
        os.path.join(loc, 'tinker_2017/tinker2017_ms_to_mh.txt'),
        format='ascii')

    return tinker17_mh_to_ms, tinker17_ms_to_mh

def kravtsov18_m500_to_mbcg(m500, a=0.39, b=12.15,
                            with_g13=False, tot=False, sat=False):
    """BCG stellar mass from halo mass based on Kravtsov+2018.

    * 9 clusters:
        Relation        Slope         Normalization    Scatter
        M*_BCG - M500   0.39+/-0.17   12.15+/-0.08     0.21+/-0.09
        M*_Sat - M500   0.87+/-0.15   12.42+/-0.07     0.10+/-0.12
        M*_Tot - M500   0.69+/-0.09   12.63+/-0.04     0.09+/-0.05

    * 21 clusters (+ Gonzalaz et al. 2013)
        Relation        Slope         Normalization    Scatter
        M*_BCG - M500   0.33+/-0.11   12.24+/-0.04     0.17+/-0.03
        M*_Sat - M500   0.75+/-0.09   12.52+/-0.03     0.10+/-0.03
        M*_Tot - M500   0.59+/-0.08   12.71+/-0.03     0.11+/-0.03

    """
    m_norm = m500 - 14.5

    if not with_g13:
        if tot:
            a, b = 0.69, 12.63
        elif sat:
            a, b = 0.87, 12.42
    else:
        if tot:
            a, b = 0.59, 12.71
        elif sat:
            a, b = 0.75, 12.52
        else:
            a, b = 0.33, 12.24

    return a * m_norm + b

def kravtsov18_mh_to_ms(logmh, mh_1=11.35, epsilon=-1.642, alpha=-1.779, delta=4.394, gamma=0.547,
                        kind=None, scatter=False):
    """Central stellar mass from halo mass based on Kravtsov et al. 2018."""
    if kind is not None:
        if kind == '200c':
            if not scatter:
                mh_1, epsilon, alpha, delta, gamma = 11.39, -1.618, -1.795, 4.345, 0.619
            else:
                mh_1, epsilon, alpha, delta, gamma = 11.35, -1.642, -1.779, 4.394, 0.547
        elif kind == '500c':
            if not scatter:
                mh_1, epsilon, alpha, delta, gamma = 11.32, -1.527, -1.856, 4.376, 0.644
            else:
                mh_1, epsilon, alpha, delta, gamma = 11.28, -1.566, -1.835, 4.437, 0.567
        elif kind == '200m':
            if not scatter:
                mh_1, epsilon, alpha, delta, gamma = 11.45, -1.702, -1.736, 4.273, 0.613
            else:
                mh_1, epsilon, alpha, delta, gamma = 11.41, -1.720, -1.727, 4.305, 0.544
        elif kind == 'vir':
            if not scatter:
                mh_1, epsilon, alpha, delta, gamma = 11.43, -1.663, -1.750, 4.290, 0.595
            else:
                mh_1, epsilon, alpha, delta, gamma = 11.39, -1.685, -1.740, 4.335, 0.531
        else:
            raise Exception("# Wrong definition of mass: [200c, 500c, 200m, vir]")

    mhalo_ratio = logmh - mh_1

    return mh_1 + epsilon + (behroozi13_f(mhalo_ratio, alpha, delta, gamma) -
                             behroozi13_f(0.0, alpha, delta, gamma))

def moster18_mh_to_ms(logmh, mh_1=11.339, n=0.005, beta=3.344, gamma=0.966,
                      fb=0.156, redshift=None):
    """Stellar mass from halo mass based on Moster et al. 2018."""
    ms_ratio = moster18_ms_mh_ratio(logmh, mh_1=mh_1, n=n, beta=beta, gamma=gamma,
                                    redshift=redshift)

    return logmh + np.log10(fb) + np.log10(ms_ratio)

def moster18_ms_mh_ratio(logmh, mh_1=11.339, n=0.005, beta=3.344, gamma=0.966,
                         redshift=None):
    """Stellar-to-halo mass ratio based on Moster et al. 2013."""
    if redshift is not None:
        mh_1, n, beta, gamma = moster18_evolution(redshift)

    mass_ratio = 10.0 ** logmh / 10.0 ** mh_1

    term1 = 2.0 * n
    term2 = mass_ratio ** -beta
    term3 = mass_ratio ** gamma

    return term1 / (term2 + term3)

def moster18_evolution(z, kind='cen'):
    """Redshift dependent of parameters in Moster et al. 2018 model.

    Based on the best-fit parameters in Table 8 of Moster et al. 2018:
    """
    if 0.0 <= z < 0.3:
        if kind == 'cen':
            mh_1, n, beta, gamma = 11.80, 0.14, 1.75, 0.57
        elif kind == 'qe':
            mh_1, n, beta, gamma = 11.65, 0.17, 1.80, 0.57
        elif kind == 'sf':
            mh_1, n, beta, gamma = 11.75, 0.12, 1.75, 0.57
        elif kind == 'all':
            mh_1, n, beta, gamma = 11.78, 0.15, 1.78, 0.57
        else:
            raise Exception("# Wrong kind: [cen, qe, sf, all]")
    elif 0.3 <= z < 0.8:
        if kind == 'cen':
            mh_1, n, beta, gamma = 11.85, 0.16, 1.70, 0.58
        elif kind == 'qe':
            mh_1, n, beta, gamma = 11.75, 0.19, 1.75, 0.58
        elif kind == 'sf':
            mh_1, n, beta, gamma = 11.80, 0.14, 1.70, 0.58
        elif kind == 'all':
            mh_1, n, beta, gamma = 11.86, 0.18, 1.67, 0.58
        else:
            raise Exception("# Wrong kind: [cen, qe, sf, all]")
    elif 0.8 <= z < 1.5:
        if kind == 'cen':
            mh_1, n, beta, gamma = 11.95, 0.18, 1.60, 0.60
        elif kind == 'qe':
            mh_1, n, beta, gamma = 11.85, 0.21, 1.65, 0.60
        elif kind == 'sf':
            mh_1, n, beta, gamma = 11.90, 0.15, 1.60, 0.60
        elif kind == 'all':
            mh_1, n, beta, gamma = 11.98, 0.19, 1.53, 0.59
        else:
            raise Exception("# Wrong kind: [cen, qe, sf, all]")
    elif 1.5 <= z < 2.5:
        if kind == 'cen':
            mh_1, n, beta, gamma = 12.00, 0.18, 1.55, 0.62
        elif kind == 'qe':
            mh_1, n, beta, gamma = 11.90, 0.21, 1.60, 0.60
        elif kind == 'sf':
            mh_1, n, beta, gamma = 11.95, 0.16, 1.55, 0.62
        elif kind == 'all':
            mh_1, n, beta, gamma = 11.99, 0.19, 1.46, 0.59
        else:
            raise Exception("# Wrong kind: [cen, qe, sf, all]")
    elif 2.5 <= z < 5.5:
        if kind == 'cen':
            mh_1, n, beta, gamma = 12.05, 0.19, 1.50, 0.64
        elif kind == 'qe':
            mh_1, n, beta, gamma = 12.00, 0.21, 1.55, 0.64
        elif kind == 'sf':
            mh_1, n, beta, gamma = 12.05, 0.18, 1.50, 0.64
        elif kind == 'all':
            mh_1, n, beta, gamma = 12.07, 0.20, 1.36, 0.60
        else:
            raise Exception("# Wrong kind: [cen, qe, sf, all]")
    elif 5.5 <= z <= 8.0:
        if kind == 'cen':
            mh_1, n, beta, gamma = 12.10, 0.24, 1.30, 0.64
        elif kind == 'qe':
            mh_1, n, beta, gamma = 12.10, 0.28, 1.30, 0.64
        elif kind == 'sf':
            mh_1, n, beta, gamma = 12.10, 0.24, 1.30, 0.64
        elif kind == 'all':
            mh_1, n, beta, gamma = 12.10, 0.24, 1.30, 0.60
        else:
            raise Exception("# Wrong kind: [cen, qe, sf, all]")
    else:
        raise Exception("# Wrong redshift range: 0 < z < 8")

    return mh_1, n, beta, gamma

def small_h_corr(h, h_ref=0.7, mh=False):
    """Correction factor for small h on stellar or halo mass."""
    if mh:
        return h / h_ref
    else:
        return (h / h_ref) ** 2.0

def imf_corr_to_chab(kind='kroupa'):
    """Correct the stellar mass to Chabrier IMF."""
    if kind == 'kroupa':
        return -0.05
    elif kind == 'salpeter':
        return -0.25
    elif kind == 'diet-salpeter':
        return -0.1
    else:
        raise Exception("# Wrong IMF type: [kroupa, salpeter, diet-salpeter]")

def sps_corr_to_bc03(kind='fsps'):
    """Correct the stellar mass to BC03 SPS model."""
    if kind == 'fsps':
        return -0.05
    elif kind == 'bc07' or kind == 'cb07':
        return 0.13
    elif kind == 'pegase':
        return -0.05
    elif kind == 'm05':
        return 0.2
    else:
        raise Exception('# Wrong SPS type: [fsps, bc07, pegase, m05]')

def m500c_to_m200c():
    """Convert M500c to M200c based on White 2001."""
    return -np.log10(0.72)