"""Utilities for A.S.A.P model."""
from __future__ import print_function, division, unicode_literals

import copy
import numbers

from scipy.stats import mvn, norm

from astropy.table import Column, vstack

import numpy as np


__all__ = ["mcmc_save_results", "mcmc_samples_stats", "mcmc_load_results",
           "mass_gaussian_weight_2d", "mass_gaussian_weight",
           "rank_splitting_sample", "weighted_avg_and_std", "ellipse_split_2d",
           "ellipse_distance", "ellipse_selector", "split_sample_along_axis_angle"]


def mcmc_samples_stats(mcmc_samples):
    """1D marginalized parameter constraints."""
    return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
               zip(*np.percentile(mcmc_samples,
                                  [16, 50, 84], axis=0)))


def mcmc_load_results(mcmc_file):
    """Retrieve the MCMC results from .npz file."""
    mcmc_data = np.load(mcmc_file)

    return (mcmc_data['samples'], mcmc_data['chains'],
            mcmc_data['lnprob'], mcmc_data['best'],
            mcmc_data['position'], mcmc_data['acceptance'])


def mcmc_save_results(mcmc_results, mcmc_sampler, mcmc_file,
                      mcmc_ndims, verbose=True):
    """Save the MCMC run results."""
    (mcmc_position, mcmc_lnprob, _) = mcmc_results

    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape(
        (-1, mcmc_ndims))
    mcmc_chains = mcmc_sampler.chain
    mcmc_lnprob = mcmc_sampler.lnprobability
    ind_1, ind_2 = np.unravel_index(np.argmax(mcmc_lnprob, axis=None),
                                    mcmc_lnprob.shape)
    mcmc_best = mcmc_chains[ind_2, ind_1, :]
    mcmc_params_stats = mcmc_samples_stats(mcmc_samples)

    np.savez(mcmc_file,
             samples=mcmc_samples, lnprob=np.array(mcmc_lnprob),
             best=np.array(mcmc_best), chains=mcmc_chains,
             position=np.asarray(mcmc_position),
             acceptance=np.array(mcmc_sampler.acceptance_fraction))

    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(mcmc_sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" % np.max(mcmc_lnprob))
        print(mcmc_best)
        print("#------------------------------------------------------")
        for param_stats in mcmc_params_stats:
            print(param_stats)
        print("#------------------------------------------------------")


def mass_gaussian_weight_2d(logms1, logms2, sigms1, sigms2,
                            bin1_l, bin1_r, bin2_l, bin2_r):
    """Weight of galaaxy using two stellar masses.

    Notes
    -----
        Now this step is actually the bottom neck.

    """
    p, _ = mvn.mvnun([bin1_l, bin2_l], [bin1_r, bin2_r], [logms1, logms2],
                     [[sigms1 ** 2, sigms1 * sigms2], [sigms2 * sigms1, sigms2 ** 2]])

    return p


def mtot_minn_weight(logm_tot, logm_inn, sig,
                     mtot_0, mtot_1, minn_0, minn_1):
    """Two-dimensional weight of galaxy in Mtot-Minn box."""
    return [mass_gaussian_weight_2d(m1, m2, ss, ss, mtot_0, mtot_1, minn_0, minn_1)
            for m1, m2, ss in zip(logm_tot, logm_inn, sig)]


def mass_gaussian_weight(logms, sigms, left, right):
    """Weights of stellar in bin."""
    return (norm.sf(left, loc=logms, scale=sigms) -
            norm.sf(right, loc=logms, scale=sigms))


def rank_splitting_sample(cat, X_col, Y_col, n_bins=5, n_sample=2,
                          X_min=None, X_max=None, X_bins=None,
                          id_each_bin=True):
    """Split sample into N_sample with fixed distribution in X, but different
    rank orders in Y.

    Parameters:
    -----------
    cat : astropy.table
        Table for input catalog
    X_col : string
        Name of the column for parameter that should have fixed distribution
    Y_col : string
        Name of the column for parameter that need to be split
    n_bins : int
        Number of bins in X
    n_sample : int
        Number of bins in Y
    X_min : float
        Minimum value of X for the binning
    X_max: float
        Maximum value of X for the binning
    X_bins : array
        Edges of X bins, provided by the users
        Usefull for irregular binnings

    Return
    ------

    """
    data = copy.deepcopy(cat)

    X = data[X_col]
    X_len = len(X)
    if X_bins is None:
        if X_min is None:
            X_min = np.nanmin(X)
        if X_max is None:
            X_max = np.nanmax(X)

        msg = '# Sample size should be much larger than number of bins in X'
        assert X_len > (2 * n_bins), msg

        X_bins = np.linspace(X_min, X_max, (n_bins + 1))
    else:
        n_bins = (len(X_bins) - 1)

    # Place holder for sample ID
    data.add_column(Column(data=(np.arange(X_len) * 0),
                           name='sample_id'))
    data.add_column(Column(data=np.arange(X_len), name='index_ori'))

    # Create index array for object in each bin
    X_idxbins = np.digitize(X, X_bins, right=True)

    bin_list = []
    for ii in range(n_bins):
        subbin = data[X_idxbins == (ii + 1)]
        subbin.sort(Y_col)

        subbin_len = len(subbin)
        subbin_size = int(np.ceil(subbin_len / n_sample))

        idx_start, idx_end = 0, subbin_size
        for jj in range(n_sample):
            if idx_end > subbin_len:
                idx_end = subbin_len
            if id_each_bin:
                subbin['sample_id'][idx_start:idx_end] = ((jj + 1) +
                                                          (ii * n_sample))
            else:
                subbin['sample_id'][idx_start:idx_end] = (jj + 1)
            idx_start = idx_end
            idx_end += subbin_size

        bin_list.append(subbin)

    new_data = vstack(bin_list)

    return new_data


def check_random_state(seed):
    """Turn seed into a `numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : `None`, int, or `numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.

    Returns
    -------
    random_state : `numpy.random.RandomState`
        RandomState object.

    Notes
    -----
    This routine is from scikit-learn.  See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def random_cmap(ncolors=256, random_state=None):
    """Generate a matplotlib colormap consisting of random (muted) colors.

    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.
    random_state : int or `~numpy.random.RandomState`, optional
        The pseudo-random number generator state used for random
        sampling.  Separate function calls with the same
        ``random_state`` will generate the same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.Colormap`
        The matplotlib colormap with random colors.
    """

    from matplotlib import colors

    prng = check_random_state(random_state)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    return colors.ListedColormap(rgb)


def weighted_avg_and_std(values, weights):
    """Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    Based on:
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)

    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)

    return average, np.sqrt(variance)


def counter_clockwise_2d_rotation_matrix(rot_angle_in_degrees):
    """Rotate 2-D matrix in counter clockwise direction."""
    rot_angle_in_radians = rot_angle_in_degrees * np.pi / 180.
    c, s = np.cos(rot_angle_in_radians), np.sin(rot_angle_in_radians)

    return np.array(((c, s), (-s, c))).T


def rotate_2d(v, rot_angle_in_degrees):
    """ Rotate a 2-d vector ``v`` in the counter-clockwise direction by some angle

    Parameters
    ----------
    v : ndarray
        Array of shape (2, )
    rot_angle_in_degrees : float
        Counter-clockwise rotation angle in units of degrees

    Returns
    -------
    rotated_v : ndarray
        Array of shape (2, )
    """
    return np.dot(counter_clockwise_2d_rotation_matrix(
        rot_angle_in_degrees), v)


def ellipse_distance(xpts, ypts, xc, yc, major_axis_angle, b_by_a):
    """ Calculate the distance between (xc, yc) and each point in (xpts, ypts)
    using an elliptical distance metric, with the ellipse defined by the
    axis ``major_axis_angle`` (in degrees) and ``b_by_a`` major-to-minor axis
    length
    """
    major_axis = rotate_2d((1, 0), major_axis_angle)
    minor_axis = rotate_2d(major_axis, 90)
    pts = np.vstack((xpts - xc, ypts - yc)).T
    a = np.dot(pts, major_axis)
    b = np.dot(pts, minor_axis)
    return np.sqrt(a ** 2 + b_by_a * b ** 2)


def ellipse_selector(xpts, ypts, xc, yc, major_axis_angle, b_by_a, nkeep):
    """ From an input set of points in the xy plane, select a sample of
    ``nkeep`` points defined by the ellipse defined by the input
    ``major_axis_angle`` and ``b_by_a``.

    Parameters
    ----------
    xpts, ypts : ndarrays
        Numpy arrays of shape (ngals, ) storing, for example, the values of
        M10 and M100

    xc, yc : floats
        Centroid of the ellipse

    major_axis_angle : float
        Angle defining the major axis direction of the ellipse.
        Angle should be in degrees in the counter-clockwise direction off of
            the positive x-axis

    b_by_a : float
        Major-to-minor axis ratio

    nkeep : int
        Number of galaxies in the output sample

    Returns
    -------
    idx : ndarray
        Numpy array of shape (nkeep, ) that can be used as a mask to select the
        galaxies passing the elliptical selection

    Notes
    -----
    See ellipse_selection_demo.ipynb for usage demo

    """
    d = ellipse_distance(xpts, ypts, xc, yc, major_axis_angle, b_by_a)
    idx_sorted_d = np.argsort(d)
    return idx_sorted_d[:nkeep]


def split_sample_along_axis_angle(xpts, ypts, xc, yc, axis_angle):
    """ From an input set of points in the xy plane, split the sample into
    points on one side or the other of the line passing through (xc, yc) with
    angle ``axis_angle``.

    Parameters
    ----------
    xpts, ypts : ndarrays
        Numpy arrays of shape (ngals, ) storing, for example, the values of
        M10 and M100

    xc, yc : floats
        Centroid of the ellipse

    axis_angle : float
        Angle defining the axis direction along which the sample will be split.
        Angle should be in degrees in the counter-clockwise direction off of
        the positive x-axis

    Returns
    -------
    idx : ndarray
        Numpy array of shape (nkeep, ) that can be used as a mask to divide the
        input sample

    Notes
    -----
    The output mask just directly splits the sample according to which side of
    the line the points fall on.
    This does *nothing* to guarantee that there will be an equal number of
    points in the resulting subsamples.

    See ellipse_selection_demo.ipynb for usage demo
    """
    splitting_axis = rotate_2d((1, 0), axis_angle)
    if splitting_axis[0] != 0:
        slope = (splitting_axis[1] / splitting_axis[0])
        b = (yc - slope * xc)
        ycut = (slope * xpts + b)
        return ypts > ycut
    else:
        return xpts > xc


def ellipse_split_2d(x_arr, y_arr, x_cen, y_cen, pa, ba,
                     nsample=300):
    """Select and split samples in an elliptical region."""
    dist_ellip = ellipse_distance(x_arr, y_arr, x_cen, y_cen,
                                  pa, ba)
    idx_sorted_dist_ellip = np.argsort(dist_ellip)

    min_dist_ellip = dist_ellip[idx_sorted_dist_ellip[nsample - 1]]

    splitting_mask = split_sample_along_axis_angle(
        x_arr, y_arr, x_cen, y_cen, pa)

    group1_mask = (splitting_mask & (dist_ellip <= min_dist_ellip))
    group2_mask = (~splitting_mask & (dist_ellip <= min_dist_ellip))

    return group1_mask, group2_mask
