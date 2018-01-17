import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize

import smhm_fit

solarMassUnits = r"($M_{\odot}$)"
smhm_ratio_scatter = r"$\sigma\ [log\ M_{*}/M_{vir}]$"

m_vir_x_axis = r"$M_{vir}\ [log\ M_{vir}/M_{\odot}]$"
m_star_x_axis = r"$M_{*}\ [log\ M_{*}/M_{\odot}]$"
sm_scatter = r"$\sigma\ [log\ M_{*}]$"
hm_scatter = r"$\sigma\ [log\ M_{vir}]$"

# Be very careful with when you are in log and when not in log...
# All plotters should plot using log10(value)
# Whether they take in data in that format or convert it depends so keep track of that

def dm_vs_all_sm_error(catalogs, x_axis, labels=None, ):
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
