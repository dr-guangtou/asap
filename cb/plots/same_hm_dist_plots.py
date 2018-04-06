import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def ks_test(catalog, key, f, cuts):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    catalog = catalog[key]["data"]
    sm_sample = _get_sm_sample(catalog, cuts)
    sm_sample_vals = np.sort(f(sm_sample))

    sample2 = f(_get_sample_with_matching_halo_dist(sm_sample, catalog, 100))
    sample2 = np.sort(sample2, axis=1)
    median = np.percentile(sample2, 50, axis=0)
    return scipy.stats.ks_2samp(median, sm_sample_vals).pvalue

def plot_cdf(catalog, key, f, cuts):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    _, ax = plt.subplots()

    catalog = catalog[key]["data"]
    sm_sample = _get_sm_sample(catalog, cuts)

    sm_sample_vals = np.sort(f(sm_sample))
    ax.plot(sm_sample_vals)

    sample2 = f(_get_sample_with_matching_halo_dist(sm_sample, catalog, 100))
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
    return ax

def plot_pdf(catalog, key, f, cuts, bins=None, ax=None):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    if ax is None:
        _, ax = plt.subplots()

    catalog = catalog[key]["data"]
    sm_sample = _get_sm_sample(catalog, cuts)

    # Plot
    _, bin_edges, _ = ax.hist(
            f(sm_sample),
            bins=bins,
            alpha=0.3,
            density=True,
            label=r"$M_{\ast}^{" + key + r"}$",
            color='b',
    )

    # Now get random dataset with matching HM dist and plot it.
    sample2 = _get_sample_with_matching_halo_dist(sm_sample, catalog, 100)
    bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])
    s2_hists = np.array([np.histogram(f(i), bins=bin_edges, density=True)[0] for i in sample2])
    ax.errorbar(
            bin_mid,
            np.mean(s2_hists, axis=0),
            yerr = np.std(s2_hists, axis=0),
            label="Random",
    )
    ax.legend()
    return ax

def _get_sm_sample(catalog, cuts):
    # Make the stellar mass cut
    sm_sample = catalog[
        (catalog["sm"] + catalog["icl"] > 10**cuts[0]) &
        (catalog["sm"] + catalog["icl"] < 10**cuts[1])
    ]
    # For cen/halo the median/mean should be close (should maybe be closer?) but std is different.
    print("SM sample size: {0}\tSM median halo mass: {1:.2e}\tSM std halo mass: {2:.2e}".format(
        len(sm_sample),
        np.median(sm_sample["m"]),
        np.std(sm_sample["m"]),
    ))
    return sm_sample


# Given a sample, randomly selects n_resamples with the same halo mass distribution
def _get_sample_with_matching_halo_dist(sm_sample, catalog, n_resamples):
    # Put our original sample in 50 bins and count the number in each. We will match this
    count1, bin_edges = np.histogram(np.log10(sm_sample["m"]), bins=50)

    bins = np.digitize(catalog["m"], np.power(10, bin_edges), right=True)
    sample2 = []
    for _ in range(n_resamples):
        s = []
        for i in range(len(count1)):
            valid_indexes = np.where( bins == i+1 )[0]
            if count1[i] == 0:
                continue # If the original data had nothing in this bin, continue
            if len(valid_indexes) == 0:
                raise Exception("You need some options for this bin... Probably need to increase bin size")
            s.append(
                np.random.choice(catalog[valid_indexes], size=count1[i])
            )
        sample2.append(np.concatenate(s))
    return np.array(sample2)

# Helper functions, all with the same signature.
def f_concentration(sample):
    return sample["rvir"] / sample["rs"]

def f_acc(sample):
    return sample["Acc_Rate_1*Tdyn"]

def f_age(sample, plot=False):
    ages = sample["Halfmass_Scale"]
    return smooth_discrete_scales(ages, plot)

def f_mm(sample, plot=False):
    ages = sample["scale_of_last_MM"]
    return smooth_discrete_scales(ages, plot)

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
