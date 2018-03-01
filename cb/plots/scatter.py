"""
Graphs plotting the scatter in one of stellar mass or halo mass against the other
Or now also plotting the scatter in some other observable (richness) against HM.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize

import fits
import smhm_fit
import data
from plots.lit_scatter import lit
from plots import labels as l

# See https://arxiv.org/pdf/0810.1885.pdf
def resample_scatter(x, y, bins):
    bin_indexes = np.digitize(x, bins)
    stds, stdstds = np.zeros(len(bins)-1), np.zeros(len(bins)-1)

    cnts = []
    for i in range(len(bins) - 1):
        # digitize is 1 indexed
        indexes_in_bin = np.where(bin_indexes == i + 1)[0]
        count_in_bin = len(indexes_in_bin)
        cnts.append(count_in_bin)
        if count_in_bin < 5:
            print("Warning - {} items in bin {}".format(count_in_bin, i+1))

        # Calculate stats for that bin
        iterations = 1000
        this_bin_std = np.zeros(iterations)
        for j in range(iterations):
            ci = np.random.choice(indexes_in_bin, len(indexes_in_bin)) # chosen indexes
            this_bin_std[j] = np.std(y[ci], ddof=1)
        stds[i] = np.mean(this_bin_std)
        stdstds[i] = np.std(this_bin_std, ddof=1)
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
    return np.mean(stds, axis=0), np.std(stds, axis=0, ddof=1)

# plots m_star_all_halo, m_star_all_cen, m_star_insitu
def hm_vs_sm_scatter_variant(central_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()
        # fig.set_size_inches(18.5, 10.5)

    for cat in ["insitu", "cen", "halo"]:
        v = central_catalogs[cat]
        if cat == "insitu": cat = "in" #hack hack hack
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
        xlabel=l.m_vir_x_axis,
        ylabel=l.sm_scatter_simple,
    )
    for _, v in lit.items():
        ax.plot(v["x"], v["y"], label=v["label"])
    ax.legend(loc="upper left")
    ax.set_ylim(top = 0.62) # to make room for the legend
    return ax

def in_sm_at_fixed_hm(central_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()

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

        y, yerr = resample_scatter(halo_masses, delta_stellar_masses, bins)
        ax.errorbar(bin_midpoints, y, yerr=yerr, label=r"$M_{\ast}^{" + str(k) + "}$", capsize=1.5, linewidth=1)

    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.sm_scatter_simple,
    )
    ax.legend(fontsize="xx-small", loc="upper right")
    return ax


# this is number density by stellar mass
def in_hm_at_fixed_number_density(combined_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    number_densities = np.logspace(-1.9, -4.2, num=10)
    number_densities_mid = number_densities[:-1] + (number_densities[1:] - number_densities[:-1]) / 2
    for k in data.cut_config.keys():
        # Convert number densities to SM so that we can use that
        sm_bins = np.array([fits.mass_at_density(combined_catalogs, k, d) for d in number_densities])

        v = combined_catalogs[k]
        stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
        halo_masses = np.log10(v["data"]["m"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v["fit"])
        delta_halo_masses = halo_masses - predicted_halo_masses

        y, yerr = resample_scatter(stellar_masses, delta_halo_masses, sm_bins)

        ax.errorbar(number_densities_mid, y, yerr=yerr, label=r"$M_{\ast}^{" + str(k) + "}$", capsize=1.5, linewidth=1)

    ax.set(
            xscale="log",
            ylim=0,
            xlabel=l.number_density,
            ylabel=l.hm_scatter,
    )
    ax.invert_xaxis()
    ax.legend(fontsize="xx-small", loc="upper right")


    # Add the mass at the top
    ax2 = ax.twiny()
    masses = [11.8, 12, 12.2, 12.4] # manually found

    lims = [fits.mass_at_density(combined_catalogs, "cen", m) for m in ax.get_xlim()]
    ax2.set(
            xlim=lims,
            xticks=masses,
            xticklabels=masses,
            xlabel=l.m_star_x_axis("cen"),
    )

    return ax

# this is
def in_hm_at_fixed_richness_number_density(combined_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    v = combined_catalogs["cen"]

    number_densities = np.logspace(-3.5, -6.5, num=12)
    number_densities_mid = number_densities[:-1] + (number_densities[1:] - number_densities[:-1]) / 2
    # Convert number density to richness so that we can use that
    r_bins = np.array([fits.richness_at_density(combined_catalogs, "cen", d) for d in number_densities])

    # Delta halo masses
    stellar_masses = np.log10(v["data"]["icl"] + v["data"]["sm"])
    halo_masses = np.log10(v["data"]["m"])
    predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v["fit"])
    delta_halo_masses = halo_masses - predicted_halo_masses
    richnesses = v["richness"]["richness"]

    y, yerr = resample_scatter(richnesses, delta_halo_masses, r_bins)

    ax.errorbar(number_densities_mid, y, yerr=yerr)
    ax.set(
            xscale="log",
            ylim=0,
            xlabel=l.number_density_richness,
            ylabel=l.hm_scatter,
    )
    ax.invert_xaxis()
    ax.legend(fontsize="xx-small", loc="upper right")

    # Add the mass at the top
    ax2 = ax.twiny()
    richnesses = [0, 10, 20, 30, 40, 50]

    lims = [fits.richness_at_density(combined_catalogs, "cen", m) for m in ax.get_xlim()]
    ax2.set(
            xlim=lims,
            xticks=richnesses,
            xticklabels=richnesses,
            xlabel=l.richness,
    )
    return ax




def in_richness_at_fixed_hm(combined_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    catalog = combined_catalogs["cen"]["richness"]
    halo_masses = np.log10(catalog["m"])
    bins = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)[:-1]
    bin_midpoints = bins[:-1] + np.diff(bins) / 2

    y, yerr = resample_scatter(halo_masses, catalog["richness"], bins)
    ax.errorbar(bin_midpoints, y, yerr=yerr)

    return ax


#### Would consider the following "non-prod"


# We want to use the same bin midpoints, but create new bins to have the same number
# in each bin as in cen.
# Note that we need to start from the most massive (
def bins_for_const_num_den(bin_counts, x_data):
    bins = []
    x_data = np.flip(np.sort(x_data), 0)
    for i in range(len(bin_counts), -1, -1):
        bins.append(x_data[np.sum(bin_counts[i:])])
    return np.array(bins[::-1])


def sanity_check_scatter(sc_centrals, hc_centrals):
    log_halo_masses = np.log10(hc_centrals["data"]["m"])

    hm_bins = np.arange(np.floor(np.min(log_halo_masses)), np.max(log_halo_masses), 0.1)

    # calculate SM scatter at fixed HM
    halo_masses = np.log10(hc_centrals["data"]["m"])
    stellar_masses = np.log10(hc_centrals["data"]["icl"] + hc_centrals["data"]["sm"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *hc_centrals["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses
    std_sm, _, _ = scipy.stats.binned_statistic(halo_masses, delta_stellar_masses, statistic="std", bins=hm_bins)
    count, _ = np.histogram(halo_masses, hm_bins)

    # More data in HM cuts
    std_sm, count = std_sm[8:], count[8:]

    # We need this many items in our SM bins too
    log_stellar_masses = np.log10(sc_centrals["data"]["icl"] + sc_centrals["data"]["sm"])
    sm_bins = bins_for_const_num_den(count, log_stellar_masses)

    # sm_bins = smhm_fit.f_shmr(hm_bins, *hc_centrals["fit"])
    sm_bin_midpoints = sm_bins[:-1] + np.diff(sm_bins) / 2

    # calculate HM scatter at fixed SM
    halo_masses = np.log10(sc_centrals["data"]["m"])
    predicted_halo_masses = smhm_fit.f_shmr_inverse(log_stellar_masses, *sc_centrals["fit"])
    delta_halo_masses = halo_masses - predicted_halo_masses
    std_hm, _, _ = scipy.stats.binned_statistic(log_stellar_masses, delta_halo_masses, statistic="std", bins=sm_bins)

    std_hm, std_sm, sm_bin_midpoints = std_hm[:-3], std_sm[:-3], sm_bin_midpoints[:-3] # Last bins are pretty empty...

    # calculate derivative at the center of the bins
    d_hm_d_sm = smhm_fit.f_shmr_inverse_der(sm_bin_midpoints, *sc_centrals["fit"][1:])


    _, ax = plt.subplots()
    ax.plot(sm_bin_midpoints, std_hm / std_sm, label=r"$\sigma_{hm}/\sigma_{sm}$")
    ax.plot(sm_bin_midpoints, d_hm_d_sm, label=r"$dlog_{hm}/dlog_{sm}$")
    ax.set(
            xlabel="The number density of a Stellar Mass of X",
    )
    ax.legend()

    return ax
