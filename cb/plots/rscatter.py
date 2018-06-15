import numpy as np
import matplotlib.pyplot as plt

import fits
from plots import labels as l
import smhm_fit
from plots.utils import resample_scatter

sim_volume = 400**3 # (400 Mpc/h)
data_key = "data_cut"
fit_key = "fit_cut"

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
