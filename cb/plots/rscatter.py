import numpy as np
import matplotlib.pyplot as plt

import fits
from plots import labels as l
import smhm_fit

sim_volume = 400**3 # (400 Mpc/h)
data_key = "data_cut"
fit_key = "fit_cut"

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
    print(cnts)
    return stds, stdstds

def in_hm_at_fixed_number_density_incl_richness(data_stellar_cut_x, richness, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    cum_counts = np.logspace(0.9, 4.3, num=10)
    cum_counts_mid = cum_counts[:-1] + (cum_counts[1:] - cum_counts[:-1]) / 2
    number_densities_mid = cum_counts_mid / sim_volume

    # True richness
    r_bins = fits.richness_at_density(richness, cum_counts)
    y, yerr = resample_scatter(
            richness["richness"]["richness"],
            np.log10(richness["richness"]["m"]),
            r_bins,
    )
    ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.ngals, capsize=1.5, linewidth=1)

    for k in ["cen", 2, "tot"]:
        # Convert number densities to SM so that we can use that
        sm_bins = fits.mass_at_density(data_stellar_cut_x[k], cum_counts, cut=True)

        v = data_stellar_cut_x[k]
        stellar_masses = np.log10(v[data_key]["icl"] + v[data_key]["sm"])
        halo_masses = np.log10(v[data_key]["m"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v[fit_key])
        delta_halo_masses = halo_masses - predicted_halo_masses

        y, yerr = resample_scatter(stellar_masses, delta_halo_masses, sm_bins)

        ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.m_star_legend(k), capsize=1.5, linewidth=1)

    ax.set(
            xscale="log",
            ylim=0,
            # xlabel=l.cum_number_density,
            # ylabel=l.hm_scatter,
            xlim=(3.5e-4, 1.2e-7),
    )

    ax.legend(fontsize="xx-small", loc="upper right")
    return ax
