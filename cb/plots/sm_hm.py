"""
Graphs plotting the stellar mass against the halo mass
"""
import numpy as np
import matplotlib.pyplot as plt
import smhm_fit
import scipy.stats
from plots import labels as l

# HM (y axis) at fixed SM (x axis). Was dm_vs_sm
def hm_at_fixed_sm(catalog, n_sats, fit=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
        # fig.set_size_inches(18.5, 10.5)
    x = np.log10(catalog["icl"] + catalog["sm"])
    y = np.log10(catalog["m"])

    # Find various stats on our data
    sm_bin_edges = np.arange(np.min(x), np.max(x), 0.1)
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
        xlabel=l.m_star_x_axis(n_sats),
        ylabel=l.m_vir_x_axis,
    )

    if fit is not None:
        ax.plot(sm_bin_midpoints, smhm_fit.f_shmr_inverse(sm_bin_midpoints, *fit), label="Best Fit", linewidth=1)
    ax.legend()

    return ax

# SM (y axis) at fixed HM (x axis). Was sm_vs_dm
def sm_at_fixed_hm(catalog, n_sats, fit=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    catalog = catalog[catalog["m"] > 1e13]

    y = np.log10(catalog["icl"] + catalog["sm"])
    x = np.log10(catalog["m"])

    # Find various stats on our data
    hm_bin_edges = np.arange(np.min(x), np.max(x), 0.1)
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
        xlabel=l.m_vir_x_axis,
        ylabel=l.m_star_x_axis(n_sats),
    )

    if fit is not None:
        ax.plot(hm_bin_midpoints, smhm_fit.f_shmr(hm_bin_midpoints, *fit), label="Best Fit", linewidth=1)
    ax.legend(loc="lower right")

    return ax

def sm_cen_at_fixed_sm_halo(z, fit=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    y = np.log10(z["sm_cen"] + z["icl_cen"])
    x = np.log10(z["sm_tot"] + z["icl_tot"])

    # Find various stats on our data
    hm_bin_edges = np.arange(np.min(x), np.max(x), 0.1)
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
        xlabel=l.m_star_x_axis("halo"),
        ylabel=l.m_star_x_axis("cen")
    )

    if fit is not None:
        ax.plot(hm_bin_midpoints, smhm_fit.f_shmr(hm_bin_midpoints, *fit), label="Best Fit", linewidth=1)
    ax.legend(loc="lower right")

    return ax
