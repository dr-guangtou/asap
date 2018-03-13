"""
These plot secondary parameters on the SHMR plot.
Should show whether these secondary parameters are the cause (or are correlated
with the cause) of the scatter)
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

import smhm_fit
from plots import labels as l

def correlation_matrix(catalog):
    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    # We have our delta stellar masses which tell us how far away from the
    # expected it is. Lets see what correlates with this!

    concentrations = cdata["rvir"] / cdata["rs"]
    richness = catalog["richness"]["richness"]
    mm = cdata["scale_of_last_MM"]
    ages = cdata["Halfmass_Scale"]
    matrix = np.vstack((delta_stellar_masses, concentrations, richness, mm, ages))
    corr = np.corrcoef(matrix)

    # Tell us about it...
    print(corr)
    fig, ax = plt.subplots()
    img = ax.imshow(corr)
    fig.colorbar(img)
    ax.set_xticklabels(["", "SM Bias", "Concentration", "Richness", "MM Scale", "Halfmass scale"],
            fontsize="xx-small")
    ax.set_yticklabels(["", "SM Bias", "Concentration", "Richness", "MM Scale", "Halfmass scale"],
            fontsize="xx-small")


def sm_at_fixed_hm_conc_split(catalog):
    fig, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    concentrations = cdata["rvir"] / cdata["rs"]

    concentration_bins = [
            np.floor(np.min(concentrations)),
            np.floor(np.percentile(concentrations, 40)),
            np.floor(np.percentile(concentrations, 60)),
            np.ceil(np.max(concentrations)),
    ]
    hm_bin_edges = np.arange(np.min(halo_masses), np.max(halo_masses), 0.1)
    hm_bin_midpoints = hm_bin_edges[:-1] + np.diff(hm_bin_edges) / 2

    # Find various stats on our data
    mean_sm, _, _, _ = scipy.stats.binned_statistic_2d(concentrations, halo_masses, stellar_masses, statistic="mean", bins=[concentration_bins, hm_bin_edges])
    std_sm, _, _, _ = scipy.stats.binned_statistic_2d(concentrations, halo_masses, stellar_masses, statistic="std", bins=[concentration_bins, hm_bin_edges])
    ax.plot(hm_bin_midpoints, mean_sm[0], label="Low concentration")
    ax.plot(hm_bin_midpoints, mean_sm[-1], label="High concentration")
    ax.fill_between(hm_bin_midpoints, mean_sm[0]-std_sm[0], mean_sm[0]+std_sm[0], alpha=0.5, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints, mean_sm[-1]-std_sm[-1], mean_sm[-1]+std_sm[-1], alpha=0.5, facecolor="tab:orange")
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.m_star_x_axis("cen"),
    )
    ax.legend()

def conc_sm_heatmap_at_fixed_hm(catalog, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    concentrations = cdata["rvir"] / cdata["rs"]
    # ax.hist(concentrations)
    matrix = np.vstack((delta_stellar_masses, concentrations))
    print(matrix.shape)
    print(np.cov(matrix))

    _, _, _, img = ax.hist2d(concentrations, delta_stellar_masses, bins=20)
    fig.colorbar(img)


    # # Find various stats on our data
    # sm_bin_edges = np.arange(np.min(x), np.max(x), 0.1)
    # sm_bin_midpoints = sm_bin_edges[:-1] + np.diff(sm_bin_edges) / 2
    # mean_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=sm_bin_edges)
    # std_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="std", bins=sm_bin_edges)

    # # Plot data and colored error regions
    # ax.plot(sm_bin_midpoints, mean_hm, marker="o", label="Universe Machine", linewidth=1)
    # ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm+std_hm, alpha=0.5, facecolor="tab:blue")
    # ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm-(2*std_hm), alpha=0.25, facecolor="tab:blue")
    # ax.fill_between(sm_bin_midpoints, mean_hm+std_hm, mean_hm+(2*std_hm), alpha=0.25, facecolor="tab:blue")
    # ax.fill_between(sm_bin_midpoints, mean_hm-(2*std_hm), mean_hm-(3*std_hm), alpha=0.125, facecolor="tab:blue")
    # ax.fill_between(sm_bin_midpoints, mean_hm+(2*std_hm), mean_hm+(3*std_hm), alpha=0.125, facecolor="tab:blue")
    # ax.set(
    #     xlabel=l.m_star_x_axis(n_sats),
    #     ylabel=l.m_vir_x_axis,
    # )

    # if fit is not None:
    #     ax.plot(sm_bin_midpoints, smhm_fit.f_shmr_inverse(sm_bin_midpoints, *fit), label="Best Fit", linewidth=1)
    # ax.legend()

    # return ax

