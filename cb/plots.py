import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize

import smhm_fit

solarMassUnits = r"($M_{\odot}$)"

m_vir_x_axis = r"$log\ M_{vir}$"
hm_scatter = r"$\sigma(log\ M_{vir})$"

sm_scatter_simple = r"$\sigma(log\ M_{\ast})$"
m_star_x_axis_simple = r"$log\ M_{\ast}$"
def m_star_x_axis(n_sats):
    return r"$log\ M_{\ast}^{" + str(n_sats) + "}$"
def sm_scatter(n_sats):
    return r"$\sigma(log\ M_{\ast}^{" + str(n_sats) + "})$"

# Be very careful with when you are in log and when not in log...
# All plotters should plot using log10(value)
# Whether they take in data in that format or convert it depends so keep track of that

def resample_scatter(x, y, bins):
    bin_indexes = np.digitize(x, bins)
    stds, stdstds = np.zeros(len(bins)-1), np.zeros(len(bins)-1)
    # Assert no empty bins
    for i in range(len(bins) - 1):
        indexes_in_bin = np.where(bin_indexes == i + 1)[0] # digitize is 1 indexed
        count_in_bin = len(indexes_in_bin)
        if count_in_bin < 5:
            print("Warning - {} items in bin {}".format(count_in_bin, i+1))

        # Calculate stats for that bin
        iterations = 1000
        this_bin_std = np.zeros(iterations)
        for j in range(iterations):
            ci = np.random.choice(indexes_in_bin, len(indexes_in_bin)) # chosen indexes
            this_bin_std[j] = np.std(y[ci])
        stds[i] = np.mean(this_bin_std)
        stdstds[i] = np.std(this_bin_std)
    return stds, stdstds

# This is simlar to ^ except it resamples everything which doesn't guarantee that
# the number of points in each bin is conserved. It *appears* to be the same.
# Trade off here is simple code, but the chance of having empty bins which I am not
# 100% sure how to deal with...
def resample_scatter_simple(x, y, bins):
    stds = []
    while len(stds) < 1000:
        si = np.random.choice(len(x), len(x))
        std, _, _ = scipy.stats.binned_statistic(x[si], y[si], statistic="std", bins=bins)
        if np.any(std == 0):
            print("warning, empty bin. Not an issue unless you see a lot (10?) of these")
            continue
        stds.append(std)
    stds = np.array(stds)
    return np.mean(stds, axis=0), np.std(stds, axis=0)

# plots m_star_all_halo, m_star_all_cen, m_star_insitu
def hm_vs_sm_scatter_variant(central_catalogs, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)

    for cat in ["insitu", "cen", "halo"]:
        v = central_catalogs[cat]
        halo_masses = np.log10(v["data"]["m"])
        stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
        predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *v["fit"])
        delta_stellar_masses = stellar_masses - predicted_stellar_masses

        bins = np.arange(
                np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
                np.max(halo_masses) + 0.2, # to ensure that the last point is included
                0.2)[:-1] # The last bin has only one data point. Can't have that.
        bin_midpoints = bins[:-1] + np.diff(bins) / 2

        std, stdstd = resample_scatter(halo_masses, delta_stellar_masses, bins)
        ax.errorbar(bin_midpoints, std, yerr=stdstd, label=r"$M_{\ast}^{" + str(cat) + "}$", capsize=1.5, linewidth=1)
    ax.set(
        xlabel=m_vir_x_axis,
        ylabel=sm_scatter_simple,
    )
    ax.legend(loc="upper left")
    ax.set_ylim(top = 0.62) # to make room for the legend
    return ax

# central_catalogs look like: {label1: {data: [ ], fit: [ ]}, label2: ...}
def hm_vs_sm_scatter(central_catalogs, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)

    for k, v in central_catalogs.items():
        if k == "insitu":
            continue
        halo_masses = np.log10(v["data"]["m"])
        stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
        predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *v["fit"])
        delta_stellar_masses = stellar_masses - predicted_stellar_masses

        bins = np.arange(
                np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
                np.max(halo_masses) + 0.2, # to ensure that the last point is included
                0.2)[:-1]
        bin_midpoints = bins[:-1] + np.diff(bins) / 2

        std, stdstd = resample_scatter(halo_masses, delta_stellar_masses, bins)
        ax.errorbar(bin_midpoints, std, yerr=stdstd, label=r"$M_{\ast}^{" + str(k) + "}$", capsize=1.5, linewidth=1)
    ax.set(
        xlabel=m_vir_x_axis,
        ylabel=sm_scatter_simple,
    )
    ax.legend()
    return ax

def sm_vs_hm_scatter(central_catalogs, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)

    for k, v in central_catalogs.items():
        if k == "insitu":
            continue
        stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
        halo_masses = np.log10(v["data"]["m"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v["fit"])
        delta_halo_masses = halo_masses - predicted_halo_masses

        bins = np.arange(
                np.floor(10*np.min(stellar_masses))/10, # round down to nearest tenth
                np.max(stellar_masses) + 0.2, # to ensure that the last point is included
                0.2)[:-1]
        bin_midpoints = bins[:-1] + np.diff(bins) / 2

        std, stdstd = resample_scatter(stellar_masses, delta_halo_masses, bins)
        ax.errorbar(bin_midpoints, std, yerr=stdstd, label=r"$M_{\ast}^{" + str(k) + "}$", capsize=1.5, linewidth=1)
    ax.set(
        xlabel=m_star_x_axis_simple,
        ylabel=hm_scatter,
    )
    ax.legend()
    return ax


def sanity_check_scatter(sc_centrals, hc_centrals):
    log_halo_masses = np.log10(hc_centrals["data"]["m"])

    hm_bins = np.arange(np.floor(np.min(log_halo_masses)), np.max(log_halo_masses), 0.2)
    sm_bins = smhm_fit.f_shmr(hm_bins, *hc_centrals["fit"])
    sm_bin_midpoints = sm_bins[:-1] + np.diff(sm_bins) / 2

    # calculate SM scatter at fixed HM
    halo_masses = np.log10(hc_centrals["data"]["m"])
    stellar_masses = np.log10(hc_centrals["data"]["icl"] + hc_centrals["data"]["sm"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *hc_centrals["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses
    std_sm, _, _ = scipy.stats.binned_statistic(halo_masses, delta_stellar_masses, statistic="std", bins=hm_bins)

    # calculate HM scatter at fixed SM
    stellar_masses = np.log10(sc_centrals["data"]["icl"] + sc_centrals["data"]["sm"])
    halo_masses = np.log10(sc_centrals["data"]["m"])
    predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *sc_centrals["fit"])
    delta_halo_masses = halo_masses - predicted_halo_masses
    std_hm, _, _ = scipy.stats.binned_statistic(stellar_masses, delta_halo_masses, statistic="std", bins=sm_bins)

    # calculate derivative at the center of the bins
    d_hm_d_sm = smhm_fit.f_shmr_inverse_der(sm_bin_midpoints, *sc_centrals["fit"][1:])

    _, ax = plt.subplots()
    ax.plot(sm_bin_midpoints, std_hm / std_sm, label=r"$\sigma_{hm}/\sigma_{sm}$")
    ax.plot(sm_bin_midpoints, d_hm_d_sm, label=r"$dlog_{hm}/dlog_{sm}$")
    ax.set(
            xlabel="Stellar Mass",
    )
    ax.legend()

    return ax

# HM (y axis) at fixed SM (x axis)
def dm_vs_sm(catalog, fit=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
    x = np.log10(catalog["icl"] + catalog["sm"])
    y = np.log10(catalog["m"])

    # Find various stats on our data
    sm_bin_edges = np.arange(np.min(x), np.max(x), 0.2)
    sm_bin_midpoints = sm_bin_edges[:-1] + np.diff(sm_bin_edges) / 2
    mean_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=sm_bin_edges)
    std_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="std", bins=sm_bin_edges)

    # Plot data and colored error regions
    ax.plot(sm_bin_midpoints, mean_hm, marker="o", label="Universe Machine", linewidth=1)
    ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm+std_hm, alpha=0.5, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm-(2*std_hm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm+std_hm, mean_hm+(2*std_hm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm-(2*std_hm), mean_hm-(3*std_hm), alpha=0.125, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm+(2*std_hm), mean_hm+(3*std_hm), alpha=0.125, facecolor="tab:blue")
    ax.set(
        xlabel=m_star_x_axis_simple,
        ylabel=m_vir_x_axis,
    )

    if fit is not None:
        ax.plot(sm_bin_midpoints, smhm_fit.f_shmr_inverse(sm_bin_midpoints, *fit), label="Best Fit", linewidth=1)
    ax.legend()

    return ax

# SM (y axis) at fixed HM (x axis)
def sm_vs_dm(catalog, n_sats, fit=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
    y = np.log10(catalog["icl"] + catalog["sm"])
    x = np.log10(catalog["m"])

    # Find various stats on our data
    hm_bin_edges = np.arange(np.min(x), np.max(x), 0.2)
    hm_bin_midpoints = hm_bin_edges[:-1] + np.diff(hm_bin_edges) / 2
    mean_sm, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=hm_bin_edges)
    std_sm, _, _ = scipy.stats.binned_statistic(x, y, statistic="std", bins=hm_bin_edges)

    # Plot data and colored error regions
    ax.plot(hm_bin_midpoints, mean_sm, marker="o", label="Universe Machine", linewidth=1)
    ax.fill_between(hm_bin_midpoints, mean_sm-std_sm, mean_sm+std_sm, alpha=0.5, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm-std_sm, mean_sm-(2*std_sm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm+std_sm, mean_sm+(2*std_sm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm-(2*std_sm), mean_sm-(3*std_sm), alpha=0.125, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm+(2*std_sm), mean_sm+(3*std_sm), alpha=0.125, facecolor="tab:blue")
    ax.set(
        xlabel=m_vir_x_axis,
        ylabel=m_star_x_axis(n_sats),
    )

    if fit is not None:
        ax.plot(hm_bin_midpoints, smhm_fit.f_shmr(hm_bin_midpoints, *fit), label="Best Fit", linewidth=1)
    ax.legend(loc="lower right")

    return ax
