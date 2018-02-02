import numpy as np
import scipy.stats
import scipy.optimize


def get_fit_binning(x_data):
    step = 0.1
    # np.arange doesn't go above the max
    edges = np.arange(np.min(x_data), np.max(x_data) + step, step)
    midpoints = edges[:-1] + np.diff(edges) / 2
    return edges, midpoints

def drop_nans(bin_midpoints, y):
    indexes = np.isfinite(y)
    return bin_midpoints[indexes], y[indexes]

# Returns the parameters needed to fit SM (on the x axis) to HM (on the y axis)
# Uses the functional form in the paper...
def get_fit(catalog):
    x = np.log10(catalog["icl"] + catalog["sm"])
    y = np.log10(catalog["m"])

    # Find various stats on our data
    sm_bin_edges, sm_bin_midpoints = get_fit_binning(x)
    mean_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=sm_bin_edges)

    # Drop nans (from empty bins)
    sm_bin_midpoints, mean_hm = drop_nans(sm_bin_midpoints, mean_hm)

    # Start with the default values from the paper
    m1 = 10**12.73
    sm0 = 10**11.04
    beta = 0.47
    delta = 0.60
    gamma = 1.96

    # Now try fit
    popt, _ = scipy.optimize.curve_fit(
            f_shmr_inverse,
            sm_bin_midpoints, # log
            mean_hm, # log
            p0=[m1, sm0, beta, delta, gamma],
            bounds=(
                [m1/1e7, sm0/1e7, 0, 0, 0],
                [m1*1e7, sm0*1e7, 10, 10, 20],
            ),
            # m1 will be smaller because we have total stellar mass (not galaxy mass). So smaller halos will have galaxies of mass M
            # sm0 will be larger for a similar reason as ^
            # beta should be unchanged - it only affects the low mass end
            # delta has large freedom
    )
    return popt

def get_fit_2(catalog):
    y = np.log10(catalog["icl"] + catalog["sm"])
    x = np.log10(catalog["m"])

    # Find various stats on our data
    hm_bin_edges, hm_bin_midpoints  = get_fit_binning(x)
    mean_sm, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=hm_bin_edges)

    # Drop nans (from empty bins)
    hm_bin_midpoints, mean_sm = drop_nans(hm_bin_midpoints, mean_sm)

    # Start with the default values from the paper
    m1 = 10**12.73
    sm0 = 10**11.04
    beta = 0.47
    delta = 0.60
    gamma = 1.96

    # Now try fit
    popt, _ = scipy.optimize.curve_fit(
            f_shmr,
            hm_bin_midpoints, # log
            mean_sm, # log
            p0=[m1, sm0, beta, delta, gamma],
            bounds=(
                [m1/1e7, sm0/1e7, 0, 0, 0],
                [m1*1e7, sm0*1e7, 10, 10, 20],
            ),
            # m1 will be smaller because we have total stellar mass (not galaxy mass). So smaller halos will have galaxies of mass M
            # sm0 will be larger for a similar reason as ^
            # beta should be unchanged - it only affects the low mass end
            # delta has large freedom
    )
    return popt

# The functional form from https://arxiv.org/pdf/1103.2077.pdf
# This is the fitting function
# f_shmr finds SM given HM. As the inverse, this find HM given SM
def f_shmr_inverse(log_stellar_masses, m1, sm0, beta, delta, gamma):
    if np.max(log_stellar_masses) > 100:
        raise Exception("You are probably not passing log masses!")

    stellar_masses = np.power(10, log_stellar_masses)

    usm = stellar_masses / sm0 # unitless stellar mass is sm / characteristic mass
    log_halo_mass = np.log10(m1) + (beta * np.log10(usm)) + ((np.power(usm, delta)) / (1 + np.power(usm, -gamma))) - 0.5
    return log_halo_mass

# d log10(halo_mass) / d log10(stellar_mass)
# http://www.wolframalpha.com/input/?i=d%2Fdx+B*log10(x%2FS)+%2B+((x%2FS)%5Ed)+%2F+(1+%2B+(x%2FS)%5E-g)+-+0.5
# https://math.stackexchange.com/questions/504997/derivative-with-respect-to-logx
def f_shmr_inverse_der(log_stellar_masses, sm0, beta, delta, gamma):
    if np.max(log_stellar_masses) > 100:
        raise Exception("You are probably not passing log masses to der!")

    stellar_masses = np.power(10, log_stellar_masses)
    usm = stellar_masses / sm0 # unitless stellar mass is sm / characteristic mass
    denom = (usm**-gamma) + 1
    return stellar_masses * np.log(10) * (
        (beta / (stellar_masses * np.log(10))) +
        ((delta * np.power(usm, delta - 1)) / (sm0 * denom)) +
        ((gamma * np.power(usm, delta - gamma - 1)) / (sm0 * np.power(denom, 2))))


# Given a list of halo masses, find the expected stellar mass
# Does this by guessing stellar masses and plugging them into the inverse
# Scipy is so sick . . .
def f_shmr(log_halo_masses, m1, sm0, beta, delta, gamma):
    if np.max(log_halo_masses) > 100:
        raise Exception("You are probably not passing log halo masses!")
    # Function to minimize
    def f(stellar_masses_guess):
        return np.sum(
                np.power(
                    f_shmr_inverse(stellar_masses_guess, m1, sm0, beta, delta, gamma) - log_halo_masses,
                    2,
                )
        )
    # Gradient of the function to minimize
    def f_der(stellar_masses_guess):
        return 2 * (
                (f_shmr_inverse(stellar_masses_guess, m1, sm0, beta, delta, gamma) - log_halo_masses) *
                f_shmr_inverse_der(stellar_masses_guess, sm0, beta, delta, gamma)
        )

    x = scipy.optimize.minimize(
            f,
            log_halo_masses - 2,
            method="CG",
            jac=f_der,
            tol=1e-12, # roughly seems to be as far as we go without loss of precision
    )
    if not x.success:
        raise Exception("Failure to invert {}".format(x.message))
    return x.x
