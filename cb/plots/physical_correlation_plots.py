"""
These plot secondary parameters on the SHMR plot.
Should show whether these secondary parameters are the cause (or are correlated
with the cause) of the scatter)
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

import smhm_fit
from plots import labels as l
from stats import partial_corr

def _pretty_corr(corr, labels):
    # print(corr)
    fig, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    lim = max(np.nanmax(corr), np.abs(np.nanmin(corr)))
    img = ax.imshow(corr, cmap="coolwarm", vmin=-lim, vmax=lim)
    lim = np.floor(10 * lim) / 10
    fig.colorbar(img, ticks=[-lim, 0, lim])

    # I'm not sure why but the first and last ticks are beyond our data?
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize="xx-small", rotation=45)
    ax.set_yticklabels(labels, fontsize="xx-small")

def _build_partial_corr(matrix):
    p_corr = partial_corr(matrix.T)
    p_corr[np.triu_indices(len(p_corr), 0)] = np.nan
    return p_corr

def _build_corr(matrix):
    corr = np.corrcoef(matrix)
    corr[np.triu_indices(len(corr), 0)] = np.nan
    return corr

def _get_data_for_correlation_matrix(catalog):
    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    concentrations = cdata["rvir"] / cdata["rs"]
    mm = cdata["scale_of_last_MM"]
    ages = cdata["Halfmass_Scale"]
    insitu_fraction= cdata["sm"] / (cdata["icl"] + cdata["sm"])

    # These require some mutation so we copy first
    richness = np.copy(catalog["richness"]["richness"]) # mutations!
    richness[richness == 0] = np.exp(-0.5) # to fix log issues
    richness = np.log10(richness)
    acc_rate_m_peak = np.copy(cdata["Acc_Rate_Mpeak"])
    acc_rate_m_peak[acc_rate_m_peak == 0] = np.min(acc_rate_m_peak[np.nonzero(acc_rate_m_peak)])
    acc_rate_m_peak = np.log10(acc_rate_m_peak)

    data = {
            "SM Bias": delta_stellar_masses,
            "Halo mass": halo_masses,
            "In-situ fraction": insitu_fraction,
            "Concentration": concentrations,
            "Acc Rate at Mpeak": acc_rate_m_peak,
            "Richness": richness,
            "Last MM Scale": mm,
            "Halfmass Scale": ages,
    }
    labels = data.keys()
    matrix = np.vstack(data.values())
    return data, labels, matrix

def cen_vs_halo_correlation_matrix(cen_catalog, halo_catalog):
    cen = _get_data_for_correlation_matrix(cen_catalog)
    halo = _get_data_for_correlation_matrix(halo_catalog)

    cen_corr = _build_corr(cen[2])
    halo_corr = _build_corr(halo[2])

    _pretty_corr(halo_corr - cen_corr, cen[1])

def marginalized_heatmap(catalog):
    data, _, matrix = _get_data_for_correlation_matrix(catalog)
    corr = _build_corr(matrix)

    fig, axes = plt.subplots(len(data), len(data))
    fig.set_size_inches(36.5, 20.5)

    d_items = list(data.items())
    for i in range(len(d_items)):
        for j in range(i):
            _, xedges, _, _ = axes[i][j].hist2d(d_items[i][1], d_items[j][1], bins=20, cmap="OrRd", norm=mpl.colors.LogNorm())
            axes[i][j].annotate(s="{:.2f}".format(corr[i][j]), xy=(0.85,1.02), xycoords="axes fraction")
            if i == len(d_items) - 1:
                axes[i][j].set_xlabel(d_items[j][0])
            if j == 0:
                axes[i][j].set_ylabel(d_items[i][0])
        for j in range(i, len(d_items)):
            axes[i][j].axis("off")

    plt.tight_layout()

def partial_correlation_matrix(catalog):
    _, labels, matrix = _get_data_for_correlation_matrix(catalog)
    p_corr = _build_partial_corr(matrix)
    _pretty_corr(p_corr, labels)


def correlation_matrix(catalog):
    _, labels, matrix = _get_data_for_correlation_matrix(catalog)
    corr = _build_corr(matrix)
    _pretty_corr(corr, labels)


#### Non correlation matrix stuff

def sm_at_fixed_hm_mm_split(catalog):
    _, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    mm_scale = cdata["scale_of_last_MM"]

    mm_scale_bins = [
            np.min(mm_scale),
            np.percentile(mm_scale, 20),
            np.percentile(mm_scale, 80),
            np.max(mm_scale),
    ]
    print(mm_scale_bins)
    ax = sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, mm_scale, mm_scale_bins, ["Old halos", "Young halos"])
    return ax

def sm_at_fixed_hm_age_split(catalog):
    _, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    hm_scale = cdata["Halfmass_Scale"]

    hm_scale_bins = [
            np.min(hm_scale),
            np.percentile(hm_scale, 20),
            np.percentile(hm_scale, 80),
            np.max(hm_scale),
    ]
    print(hm_scale_bins)
    ax = sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, hm_scale, hm_scale_bins, ["Old halos", "Young halos"])
    return ax

def sm_at_fixed_hm_conc_split(catalog):
    _, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    concentrations = cdata["rvir"] / cdata["rs"]

    concentration_bins = [
            np.floor(np.min(concentrations)),
            np.floor(np.percentile(concentrations, 20)),
            np.floor(np.percentile(concentrations, 80)),
            np.ceil(np.max(concentrations)),
    ]
    print(concentration_bins)
    ax = sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, concentrations, concentration_bins, ["Low concentration", "High concentration"])
    return ax

def _cleanup_nans(y, yerr, x):
    good_y, good_yerr, good_x = [], [], []
    for i in range(len(y)):
        good_idxs = ((np.isfinite(y[i])) & (yerr[i] != 0))
        good_y.append(y[i][good_idxs])
        good_yerr.append(yerr[i][good_idxs])
        good_x.append(x[good_idxs])
    return np.array(good_y), np.array(good_yerr), np.array(good_x)

def sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, split_params, split_bins, legend):
    hm_bin_edges = np.arange(np.min(halo_masses), np.max(halo_masses), 0.1)
    hm_bin_midpoints = hm_bin_edges[:-1] + np.diff(hm_bin_edges) / 2

    # Find various stats on our data
    mean_sm, _, _, _ = scipy.stats.binned_statistic_2d(split_params, halo_masses, stellar_masses, statistic="mean", bins=[split_bins, hm_bin_edges])
    std_sm, _, _, _ = scipy.stats.binned_statistic_2d(split_params, halo_masses, stellar_masses, statistic="std", bins=[split_bins, hm_bin_edges])

    mean_sm, std_sm, hm_bin_midpoints = _cleanup_nans(mean_sm, std_sm, hm_bin_midpoints)

    ax.plot(hm_bin_midpoints[0], mean_sm[0], label=legend[0])
    ax.plot(hm_bin_midpoints[-1], mean_sm[-1], label=legend[1])
    ax.fill_between(hm_bin_midpoints[0], mean_sm[0]-std_sm[0], mean_sm[0]+std_sm[0], alpha=0.5, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints[-1], mean_sm[-1]-std_sm[-1], mean_sm[-1]+std_sm[-1], alpha=0.5, facecolor="tab:orange")
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.m_star_x_axis("cen"),
    )
    ax.legend()
    return ax

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
