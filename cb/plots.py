import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize

import smhm_fit

solarMassUnits = r"($M_{\odot}$)"
smhm_ratio_scatter = r"$\sigma\ [log\ M_{*}/M_{vir}]$"

m_vir_x_axis = r"$log\ M_{vir}$"
hm_scatter = r"$\sigma_{log\ M_{vir}}$"

def m_star_x_axis(n_sats):
    return r"$log\ M_{*}^{" + str(n_sats) + "}$"
def sm_scatter(n_sats):
    return r"$\sigma_{log\ M_{*}^{" + str(n_sats) + "}}$"

# Be very careful with when you are in log and when not in log...
# All plotters should plot using log10(value)
# Whether they take in data in that format or convert it depends so keep track of that

# central_catalogs look like: {label1: {data: [ ], fit: [ ]}, label2: ...}
def hm_vs_sm_scatter(central_catalogs, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)

    for k, v in central_catalogs.items():
        halo_masses = np.log10(v["data"]["m"])
        stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
        predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *v["fit"])
        delta_stellar_masses = stellar_masses - predicted_stellar_masses

        bins = np.arange(
                np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
                np.max(halo_masses) + 0.2, # to ensure that the last point is included
                0.2)
        bin_midpoints = bins[:-1] + np.diff(bins) / 2
        std, _, _ = scipy.stats.binned_statistic(halo_masses, delta_stellar_masses, statistic="std", bins=bins)
        ax.plot(bin_midpoints[:-1], std[:-1], label=r"$M^{" + str(k) + "}$") # fixme: hack to remove last bin where there is only one point
    ax.set(
        xlabel=m_vir_x_axis,
        ylabel=sm_scatter,
    )
    ax.legend()
    return ax

def sm_vs_hm_scatter(central_catalogs, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)

    for k, v in central_catalogs.items():
        halo_masses = np.log10(v["data"]["m"])
        stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v["fit"])
        delta_halo_masses = halo_masses - predicted_halo_masses

        bins = np.arange(
                np.floor(10*np.min(stellar_masses))/10, # round down to nearest tenth
                np.max(stellar_masses) + 0.2, # to ensure that the last point is included
                0.2)
        bin_midpoints = bins[:-1] + np.diff(bins) / 2
        std, _, _ = scipy.stats.binned_statistic(stellar_masses, delta_halo_masses, statistic="std", bins=bins)
        ax.plot(bin_midpoints[:-1], std[:-1], label=r"$M^{" + str(k) + "}$") # fixme: hack to remove last bin where there is only one point
    ax.set(
        xlabel=m_star_x_axis,
        ylabel=hm_scatter,
    )
    ax.legend()
    return ax


def dm_vs_all_sm_error(catalogs, x_axis, labels=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    label = None
    for i, catalog in enumerate(catalogs):
        # Scatter on HM given SM bins
        if x_axis == "sm":
            x = np.log10(catalog["icl"] + catalog["sm"])
            y = np.log10(catalog["m"])
            y_diff = y - smhm_fit.f_shmr_inverse(x, *smhm_fit.get_fit(catalog))
        # Scatter on SM given HM bins
        elif x_axis == "hm":
            x = np.log10(catalog["m"])
            y = np.log10(catalog["icl"] + catalog["sm"])
            y_diff = y - smhm_fit.f_shmr(x, *smhm_fit.get_fit(catalog))
        else:
            raise Exception("x_axis must be 'sm' or 'hm', got {}".format(x_axis))

        bins = np.arange(np.min(x), np.max(x), 0.2)
        std, _, _ = scipy.stats.binned_statistic(x, y_diff, statistic="std", bins=bins)
        bin_midpoints = bins[:-1] + np.diff(bins) / 2
        if labels is not None:
            label = labels[i]
        ax.plot(bin_midpoints, std, label=label)
    if x_axis == "sm":
        ax.set(
            xlabel=r"$M_{*}\ [log\ M_{*}/M_{\odot}]$",
            ylabel=r"$\sigma\ [log\ M_{vir}/M_{\odot}]$",
            title="Scatter in Total Stellar Mass - Peak Halo Mass Ratio",
        )
    elif x_axis == "hm":
        ax.set(
            xlabel=m_vir_x_axis,
            ylabel=smhm_ratio_scatter,
            title="Scatter in Total Stellar Mass - Peak Halo Mass Ratio",
        )
    ax.legend()
    return ax


# HM (y axis) at fixed SM (x axis)
def dm_vs_sm(catalog, fit=None, ax=None):
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
    ax.plot(sm_bin_midpoints, mean_hm, marker="o")
    ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm+std_hm, alpha=0.5, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm-(2*std_hm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm+std_hm, mean_hm+(2*std_hm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm-(2*std_hm), mean_hm-(3*std_hm), alpha=0.125, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm+(2*std_hm), mean_hm+(3*std_hm), alpha=0.125, facecolor="tab:blue")
    ax.set(
        xlabel=r"$M_{*}\ [log\ M_{*}/M_{\odot}]$",
        ylabel=r"$M_{vir}\ [log\ M_{vir}/M_{\odot}]$",
    )

    if fit is not None:
        ax.plot(sm_bin_midpoints, smhm_fit.f_shmr_inverse(sm_bin_midpoints, *fit))

    return ax

# SM (y axis) at fixed HM (x axis)
def sm_vs_dm(catalog, fit=None, ax=None):
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
    ax.plot(hm_bin_midpoints, mean_sm, marker="o")
    ax.fill_between(hm_bin_midpoints, mean_sm-std_sm, mean_sm+std_sm, alpha=0.5, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm-std_sm, mean_sm-(2*std_sm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm+std_sm, mean_sm+(2*std_sm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm-(2*std_sm), mean_sm-(3*std_sm), alpha=0.125, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm+(2*std_sm), mean_sm+(3*std_sm), alpha=0.125, facecolor="tab:blue")
    ax.set(
        xlabel=r"$M_{vir}\ [log\ M_{vir}/M_{\odot}]$",
        ylabel=r"$M_{*}\ [log\ M_{*}/M_{\odot}]$",
    )

    if fit is not None:
        ax.plot(hm_bin_midpoints, smhm_fit.f_shmr(hm_bin_midpoints, *fit))

    return ax
