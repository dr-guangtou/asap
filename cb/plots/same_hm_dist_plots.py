import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import colossus.cosmology.cosmology
import colossus.halo.mass_so
from plots import labels as l


def ks_test(sm_cut_catalog, hm_cut_catalog, key, f, cuts):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    sm_cut_catalog = sm_cut_catalog[key]["data"]
    hm_cut_catalog = hm_cut_catalog[key]["data"]

    sm_sample = _get_sm_sample(sm_cut_catalog, cuts)
    sm_sample_vals = f(sm_sample)

    # We need to flatten this (can't just reuse) because of our sampling method.
    sample2 = f(_get_sample_with_matching_halo_dist(sm_sample, hm_cut_catalog, 40).flatten())
    pvalues = np.array([scipy.stats.ks_2samp(np.random.choice(sample2, size=len(sm_sample_vals)), sm_sample_vals).pvalue for i in range(40)])

    # pvalue_ks = scipy.stats.ks_2samp(sample2, sm_sample_vals).pvalue
    # This explodes some times
    # _, _, significance_ad = scipy.stats.anderson_ksamp((sample2, sm_sample_vals))
    return np.mean(pvalues), np.std(pvalues)

def calc_median_shift(sm_cut_catalog, hm_cut_catalog, key, f, cuts):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])

    sm_cut_catalog = sm_cut_catalog[key]["data"]
    hm_cut_catalog = hm_cut_catalog[key]["data"]
    sm_sample = _get_sm_sample(sm_cut_catalog, cuts)

    sm_sample_vals = np.sort(f(sm_sample))
    sample2 = f(_get_sample_with_matching_halo_dist(sm_sample, hm_cut_catalog, 100).flatten())

    # Just throw away nans - there should be a similar number in each sample that it
    # won't make a huge difference
    sample2 = sample2[np.isfinite(sample2)]
    sm_sample_vals = sm_sample_vals[np.isfinite(sm_sample_vals)]

    h_cut = np.percentile(sample2, 50)
    s_cut = np.percentile(sm_sample_vals, 50)
    print("Stellar cut", "Halo cut", "S - H")
    print(s_cut, h_cut, s_cut - h_cut)


def plot_cdf(sm_cut_catalog, hm_cut_catalog, key, f, cuts):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    _, ax = plt.subplots()

    sm_cut_catalog = sm_cut_catalog[key]["data"]
    hm_cut_catalog = hm_cut_catalog[key]["data"]
    sm_sample = _get_sm_sample(sm_cut_catalog, cuts)

    sm_sample_vals = np.sort(f(sm_sample))
    ax.plot(sm_sample_vals)

    sample2 = f(_get_sample_with_matching_halo_dist(sm_sample, hm_cut_catalog, 100))
    sample2 = np.sort(sample2, axis=1)

    sd = 34 # distance of 1sd from the median
    upper = np.percentile(sample2, 50 + sd, axis=0)
    lower = np.percentile(sample2, 50 - sd, axis=0)
    median = np.percentile(sample2, 50, axis=0)

    ax.plot(median)
    ax.fill_between(
            np.arange(len(sm_sample)),
            lower,
            upper,
            facecolor="tab:orange",
            alpha=0.25,
    ) # this region ends up being tiny (within the line)

    print(scipy.stats.ks_2samp(median, sm_sample_vals))
    print(np.percentile(sm_sample_vals, 50), np.percentile(sample2, 50))
    return ax

def plot_pdf(sm_cut_catalog, hm_cut_catalog, key, f, cuts, bins=None, ax=None):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    if ax is None:
        _, ax = plt.subplots()

    sm_cut_catalog = sm_cut_catalog[key]["data"]
    hm_cut_catalog = hm_cut_catalog[key]["data"]
    sm_sample = _get_sm_sample(sm_cut_catalog, cuts)

    # Plot
    _, bin_edges, _ = ax.hist(
            f(sm_sample),
            bins=bins,
            alpha=0.3,
            density=True,
            label=l.m_star_x_axis(key),
            color='b',
    )

    # Now get random dataset with matching HM dist and plot it.
    sample2 = _get_sample_with_matching_halo_dist(sm_sample, hm_cut_catalog, 100)
    bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])
    s2_hists = np.array([np.histogram(f(i), bins=bin_edges, density=True)[0] for i in sample2])
    ax.errorbar(
            bin_mid,
            np.mean(s2_hists, axis=0),
            yerr = np.std(s2_hists, axis=0),
            label="Random",
    )
    ax.legend(fontsize="xx-small")
    return ax

def _get_sm_sample(catalog, cuts):
    # Make the stellar mass cut
    sm_sample = catalog[
        (catalog["sm"] + catalog["icl"] > 10**cuts[0]) &
        (catalog["sm"] + catalog["icl"] < 10**cuts[1])
    ]
    # For cen/halo the median/mean should be close (should maybe be closer?) but std is different.
    bad = sm_sample["m"] < 10**11.5
    assert np.count_nonzero(bad) / len(sm_sample) < 0.01
    sm_sample = sm_sample[np.logical_not(bad)]
    print("SM sample size: {0} ({1} bad)".format(len(sm_sample), np.count_nonzero(bad)))
    return sm_sample


# Given a sample, randomly selects n_resamples with the same halo mass distribution
def _get_sample_with_matching_halo_dist(sm_sample, catalog, n_resamples):
    sample2 = []
    for _ in range(n_resamples):
        # Put our original sample in x bins and count the number in each. We will match this
        bin_counts, bin_edges = np.histogram(np.log10(sm_sample["m"]), bins=np.random.randint(40, 60))
        # Which bin each of our hm_sample fall into.
        # 1 is the first bin (not 0)
        which_bin = np.digitize(catalog["m"], np.power(10, bin_edges), right=True)
        s = []
        for i in range(len(bin_counts)):
            # if the original data had nothing in this bin, continue
            if bin_counts[i] == 0:
                continue
            valid_indexes = np.where( which_bin == i+1 )[0]
            if len(valid_indexes) == 0:
                print(bin_counts[i], i)
                raise Exception("You need some options for this bin... Probably need to increase bin size")
            s.append(
                np.random.choice(catalog[valid_indexes], size=bin_counts[i])
            )
        sample2.append(np.concatenate(s))
    return np.array(sample2)

# Helper functions, all with the same signature.
def f_concentration(sample):
    return sample["rvir"] / sample["rs"]

# Normalised in the same way that benedict did it.
def f_acc(sample, n_dyn=None):
    cosmo = colossus.cosmology.cosmology.setCosmology('planck15')

    a_now = 0.712400
    z_now = (1 / a_now) - 1
    t_now = cosmo.age(z_now)

    if n_dyn is None or n_dyn == 1:
        t_ago = colossus.halo.mass_so.dynamicalTime(z_now, "vir", "crossing")
        key = "Acc_Rate_1*Tdyn"
    elif n_dyn == 2:
        t_ago = 2*colossus.halo.mass_so.dynamicalTime(z_now, "vir", "crossing")
        key = "Acc_Rate_2*Tdyn"
    elif n_dyn == "100Myr":
        t_ago = 0.1
        key = "Acc_Rate_100Myr"
    else:
        raise Exception("bugger")


    t_then = t_now - t_ago
    z_then = cosmo.age(t_then, inverse=True)
    a_then = 1/(1 + z_then)
    if np.any(a_then < 0):
        print("Some are less than 0")

    delta_mass = sample[key] * t_ago * 1e9 # t_dyn is in Gyr
    x = np.count_nonzero(sample["m"] - delta_mass < 0)
    if x > 0:
        print(x, "here are less than 0")

    res = (
            np.log10(sample["m"]) - np.log10(sample["m"] - delta_mass)) / (
            np.log10(a_now) - np.log10(a_then))
    return res


def f_age(sample, plot=False):
    ages = sample["Halfmass_Scale"]
    return ages
    # return smooth_discrete_scales(ages, plot)

def f_mm(sample, plot=False):
    ages = sample["scale_of_last_MM"]
    return ages
    # return smooth_discrete_scales(ages, plot)

def smooth_discrete_scales(ages, plot):
    subtractive = {}
    for i in range(1, len(_all_scales)):
        subtractive[_all_scales[i]] = _all_scales[i] - _all_scales[i-1]

    smooth_ages = np.copy(ages).flatten()
    for i in range(len(smooth_ages)):
        smooth_ages[i] -= np.random.random() * subtractive[smooth_ages[i]]

    if plot:
        _, ax = plt.subplots()
        ax.hist(smooth_ages, alpha=0.3, bins=100)
        ax.hist(ages, alpha=0.3, bins=100)
        subset_ages = smooth_ages[(smooth_ages > _all_scales[43]) & (smooth_ages < _all_scales[44])]
        _, ax = plt.subplots()
        ax.hist(subset_ages, alpha=0.3, bins=3)

    return np.reshape(smooth_ages, ages.shape)

# This was pulled from the file names of the hlists. We then removed trailing zeros and reduced the sig figs in a couple of places (mostly at low scales - < 0.01).
_all_scales = np.array([0, 0.0511, 0.0556, 0.0601, 0.0646, 0.0691, 0.0736, 0.0781, 0.0826, 0.0894, 0.0967, 0.104, 0.109, 0.113, 0.118, 0.123, 0.128, 0.133, 0.139, 0.145, 0.151, 0.157, 0.164, 0.171, 0.178, 0.186, 0.194, 0.202, 0.21, 0.219, 0.228, 0.238, 0.248, 0.258, 0.269, 0.281, 0.292, 0.305, 0.318, 0.331, 0.345, 0.359, 0.375, 0.39, 0.407, 0.424, 0.442, 0.46, 0.48, 0.5, 0.53, 0.5444, 0.5504, 0.5563, 0.5623, 0.5684, 0.5743, 0.5803, 0.5864, 0.5924, 0.5983, 0.6043, 0.6104, 0.6163, 0.6223, 0.6284, 0.6344, 0.6403, 0.6464, 0.6524, 0.6583, 0.6643, 0.6704, 0.6763, 0.6823, 0.6884, 0.6944, 0.7003, 0.7064, 0.7124, 0.7183, 0.7243, 0.7363, 0.7423, 0.7544, 0.7603, 0.7724, 0.7783, 0.7873, 0.7904, 0.8023, 0.8084, 0.8144, 0.8173, 0.8234, 0.8263, 0.8324, 0.8353, 0.8414, 0.8443, 0.8504, 0.8533, 0.8594, 0.8623, 0.8684, 0.8713, 0.8773, 0.8803, 0.8863, 0.8893, 0.8953, 0.8984, 0.9043, 0.9074, 0.92515, 0.956, 0.97071, 1.00000])
