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
from plots.lit_scatter import plot_lit
from plots import labels as l

from importlib import reload
sim_volume = 400**3 # (400 Mpc/h)
data_key = "data"#_cut"
fit_key = "fit"#_cut"


# See https://arxiv.org/pdf/0810.1885.pdf
def resample_scatter(x, y, bins):
    bin_indexes = np.digitize(x, bins)
    # print(np.histogram(bin_indexes))
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
    print(cnts)
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
def in_sm_at_fixed_hm_incl_lit(central_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()
        # fig.set_size_inches(18.5, 10.5)

    our_lines = []
    for cat in ["in", "cen", "tot"]:
        v = central_catalogs[cat]
        indexes = v[data_key]["m"] > 1e13
        halo_masses = np.log10(v[data_key]["m"][indexes])
        stellar_masses = np.log10(v[data_key]["icl"][indexes] + v[data_key]["sm"][indexes])

        predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *v[fit_key])
        delta_stellar_masses = stellar_masses - predicted_stellar_masses

        bins = np.arange(
                np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
                np.max(halo_masses) + 0.2, # to ensure that the last point is included
                0.2)[:-1] # The last bin has only one data point. Can't have that.
        bin_midpoints = bins[:-1] + np.diff(bins) / 2

        std, stdstd = resample_scatter(halo_masses, delta_stellar_masses, bins)
        print(std)
        our_lines.append(
            ax.errorbar(bin_midpoints, std, yerr=stdstd, label=l.m_star_legend(cat), capsize=1.5, linewidth=1)
        )
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.sm_scatter("x"),
        xlim=ax.get_xlim(), # don't let the lit values move this around
    )
    ax.add_artist(ax.legend(handles=our_lines, loc="upper right", fontsize="xx-small"))

    # And now for the lit values
    ax = plot_lit(ax)
    ax.set_ylim(top = 0.68) # to make room for the legend
    return ax

def in_sm_at_fixed_hm(central_catalogs, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    for k, v in central_catalogs.items():
        if k == "in":
            continue
        indexes = v[data_key]["m"] > 1e13
        halo_masses = np.log10(v[data_key]["m"][indexes])
        stellar_masses = np.log10(v[data_key]["icl"][indexes] + v[data_key]["sm"][indexes])
        predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *v[fit_key])
        delta_stellar_masses = stellar_masses - predicted_stellar_masses

        bins = np.arange(
                np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
                np.max(halo_masses) + 0.2, # to ensure that the last point is included
                0.2)[:-1]
        bin_midpoints = bins[:-1] + np.diff(bins) / 2

        y, yerr = resample_scatter(halo_masses, delta_stellar_masses, bins)
        ax.errorbar(bin_midpoints, y, yerr=yerr, label=l.m_star_legend(k), capsize=1.5, linewidth=1)

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

    cum_counts = np.logspace(0.9, 4.3, num=10)
    cum_counts_mid = cum_counts[:-1] + (cum_counts[1:] - cum_counts[:-1]) / 2
    number_densities_mid = cum_counts_mid / sim_volume

    for k in data.cut_config.keys():
        # Convert number densities to SM so that we can use that
        sm_bins = fits.mass_at_density(combined_catalogs[k], cum_counts)

        v = combined_catalogs[k]
        stellar_masses = np.log10(v[data_key]["icl"] + v[data_key]["sm"])
        halo_masses = np.log10(v[data_key]["m"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v[fit_key])
        delta_halo_masses = halo_masses - predicted_halo_masses

        y, yerr = resample_scatter(stellar_masses, delta_halo_masses, sm_bins)

        ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.m_star_legend(k), capsize=1.5, linewidth=1)
    ax.set(
            xscale="log",
            ylim=0,
            xlabel=l.cum_number_density,
            ylabel=l.hm_scatter,
    )
    ax.invert_xaxis()
    ax.legend(fontsize="xx-small", loc="upper right")


    # Add the mass at the top
    ax2 = ax.twiny()
    halo_masses = [13, 14, 14.5, 14.8] # remeber to change the xlabel if you change this
    ticks = np.array(fits.density_at_hmass(combined_catalogs["cen"], halo_masses)) / sim_volume
    # cen_masses = [11.5, 11.8, 12.1, 12.4]
    # ticks = fits.density_at_mass(combined_catalogs["cen"], cen_masses) / sim_volume
    ax2.set(
            xlim=np.log10(ax.get_xlim()),
            xticks=np.log10(ticks),
            xticklabels=halo_masses,
            # xlabel=l.m_star_x_axis("cen"),
            xlabel=l.m_vir_x_axis,
    )

    return ax

def in_hm_at_fixed_number_density_incl_richness(combined_catalogs, richness, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    cum_counts = np.logspace(0.9, 4.3, num=10)
    cum_counts_mid = cum_counts[:-1] + (cum_counts[1:] - cum_counts[:-1]) / 2
    number_densities_mid = cum_counts_mid / sim_volume

    # Photoz richness
    r_bins = fits.photoz_richness_at_density(richness, cum_counts)
    y, yerr = resample_scatter(
            richness["richness"]["photoz_richness"],
            np.log10(richness["richness"]["m"]),
            r_bins,
    )
    ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.scatter_photoz, capsize=1.5, linewidth=1)

    # True richness
    r_bins = fits.richness_at_density(richness, cum_counts)
    y, yerr = resample_scatter(
            richness["richness"]["richness"],
            np.log10(richness["richness"]["m"]),
            r_bins,
    )
    ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.scatter_ideal, capsize=1.5, linewidth=1)

    # Specz richness
    r_bins = fits.specz_richness_at_density(richness, cum_counts)
    y, yerr = resample_scatter(
            richness["richness"]["specz_richness"],
            np.log10(richness["richness"]["m"]),
            r_bins,
    )
    ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.scatter_specz, capsize=1.5, linewidth=1)


    # # True richness
    # richnesses = richness["richness"]["richness"]
    # r_bins = np.array([2, 3, 4, 6, 8, 10, 13, 16, 20, 24, 28, 39, 50])
    # r_bins_mid = _bins_mid(r_bins)
    # number_densities_mid = np.array(fits.density_at_richness(richness, "", r_bins_mid)) / sim_volume
    # y, yerr = resample_scatter(richnesses, np.log10(richness["richness"]["m"]), r_bins)
    # ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.scatter_intrinsic, capsize=1.5, linewidth=1)
    next(ax._get_lines.prop_cycler)


    cum_counts = np.logspace(0.9, 4.3, num=10)
    cum_counts_mid = cum_counts[:-1] + (cum_counts[1:] - cum_counts[:-1]) / 2
    number_densities_mid = cum_counts_mid / sim_volume

    # 1 -> 2 but I can't get a fit...
    for k in ["cen", 1, "tot"]:
        # Convert number densities to SM so that we can use that
        sm_bins = fits.mass_at_density(combined_catalogs[k], cum_counts)

        v = combined_catalogs[k]
        stellar_masses = np.log10(v[data_key]["icl"] + v[data_key]["sm"])
        halo_masses = np.log10(v[data_key]["m"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v[fit_key])
        delta_halo_masses = halo_masses - predicted_halo_masses

        y, yerr = resample_scatter(stellar_masses, delta_halo_masses, sm_bins)

        ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.m_star_legend(k), capsize=1.5, linewidth=1)

    # Rozo2014
    minx = fits.density_at_richness(richness, 20) / sim_volume
    maxx = fits.density_at_richness(richness, 70) / sim_volume
    line = ax.plot([minx, maxx], [0.11, 0.11], color=l.r2014, linestyle="dashed", label="Rozo2014")[0]
    ax.fill_between([minx, maxx], 0.09, 0.13, alpha=0.2, facecolor=line.get_color())

    ax.set(
            xscale="log",
            ylim=0,
            xlabel=l.cum_number_density,
            ylabel=l.hm_scatter,
            xlim=(3.5e-4, 1.2e-7),
    )

    # ax.invert_xaxis()
    ax.legend(fontsize="xx-small", loc="upper right")
    return ax


def in_hm_at_fixed_richness_number_density(richness, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    # Photoz richness
    r_bins = np.array([4, 6, 8, 10, 13, 16, 20, 24, 28, 39, 50])
    r_bins_mid = _bins_mid(r_bins)
    y, yerr = resample_scatter(richness["richness"]["photoz_richness"], np.log10(richness["richness"]["m"]), r_bins)
    ax.errorbar(r_bins_mid, y, yerr=yerr, label=l.scatter_photoz)

    # True richness
    r_bins = np.array([4, 6, 8, 10, 13, 16, 20, 24, 28, 39, 50])
    r_bins_mid = _bins_mid(r_bins)
    y, yerr = resample_scatter(richness["richness"]["richness"], np.log10(richness["richness"]["m"]), r_bins)
    ax.errorbar(r_bins_mid, y, yerr=yerr, label=l.scatter_ideal)

    # Specz richness
    r_bins = np.array([4, 6, 8, 10, 13, 16, 20, 24, 28, 39])
    r_bins_mid = _bins_mid(r_bins)
    y, yerr = resample_scatter(richness["richness"]["specz_richness"], np.log10(richness["richness"]["m"]), r_bins)
    ax.errorbar(r_bins_mid, y, yerr=yerr, label=l.scatter_specz)


    ax.set(
            ylim=0,
            xlabel=l.ngals,
            ylabel=l.hm_scatter,
            xlim=(0, 50),
    )


    # Add the ND at the top
    ax2 = ax.twiny()
    powers = np.arange(-7, -3.9) # [-7, -6, -5, -4]
    number_densities = 10**powers
    cumulative_counts = number_densities * sim_volume
    ticks = fits.richness_at_density(richness, cumulative_counts)

    ax2.set(
            # xscale="log",
            xlim=ax.get_xlim(),
            xticks=ticks,
            xticklabels=[r"$10^{" + str(int(i)) + "}$" for i in powers],
            xlabel=l.cum_number_density,
    )
    # ax2.invert_xaxis()

    line = ax.plot([10, 70], [0.195, 0.195], color=l.r2009, linestyle="dashed", label="Rozo2009")[0]
    ax.fill_between([10, 70], 0.12, 0.28, alpha=0.2, facecolor=line.get_color())

    line = ax.plot([20, 70], [0.11, 0.11], color=l.r2014, linestyle="dashed", label="Rozo2014")[0]
    ax.fill_between([20, 70], 0.09, 0.13, alpha=0.2, facecolor=line.get_color())

    ax.legend(fontsize="xx-small")#, loc="upper right")
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

def _bins_mid(bins):
    return bins[:-1] + (bins[1:] - bins[:-1]) / 2


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
    log_halo_masses = np.log10(hc_centrals[data_key]["m"])

    hm_bins = np.arange(np.floor(np.min(log_halo_masses)), np.max(log_halo_masses), 0.1)

    # calculate SM scatter at fixed HM
    halo_masses = np.log10(hc_centrals[data_key]["m"])
    stellar_masses = np.log10(hc_centrals[data_key]["icl"] + hc_centrals[data_key]["sm"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *hc_centrals[fit_key])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses
    std_sm, _, _ = scipy.stats.binned_statistic(halo_masses, delta_stellar_masses, statistic="std", bins=hm_bins)
    count, _ = np.histogram(halo_masses, hm_bins)

    # More data in HM cuts
    std_sm, count = std_sm[8:], count[8:]

    # We need this many items in our SM bins too
    log_stellar_masses = np.log10(sc_centrals[data_key]["icl"] + sc_centrals[data_key]["sm"])
    sm_bins = bins_for_const_num_den(count, log_stellar_masses)

    # sm_bins = smhm_fit.f_shmr(hm_bins, *hc_centrals[fit_key])
    sm_bin_midpoints = sm_bins[:-1] + np.diff(sm_bins) / 2

    # calculate HM scatter at fixed SM
    halo_masses = np.log10(sc_centrals[data_key]["m"])
    predicted_halo_masses = smhm_fit.f_shmr_inverse(log_stellar_masses, *sc_centrals[fit_key])
    delta_halo_masses = halo_masses - predicted_halo_masses
    std_hm, _, _ = scipy.stats.binned_statistic(log_stellar_masses, delta_halo_masses, statistic="std", bins=sm_bins)

    std_hm, std_sm, sm_bin_midpoints = std_hm[:-3], std_sm[:-3], sm_bin_midpoints[:-3] # Last bins are pretty empty...

    # calculate derivative at the center of the bins
    d_hm_d_sm = smhm_fit.f_shmr_inverse_der(sm_bin_midpoints, *sc_centrals[fit_key][1:])


    _, ax = plt.subplots()
    ax.plot(sm_bin_midpoints, std_hm / std_sm, label=r"$\sigma_{hm}/\sigma_{sm}$")
    ax.plot(sm_bin_midpoints, d_hm_d_sm, label=r"$dlog_{hm}/dlog_{sm}$")
    ax.set(
            xlabel="The number density of a Stellar Mass of X",
    )
    ax.legend()

    return ax
