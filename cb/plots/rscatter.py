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

    cum_counts = np.logspace(0.9, 4.1, num=10)
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
        print(k)
        # Convert number densities to SM so that we can use that
        sm_bins = fits.mass_at_density(data_stellar_cut_x[k], cum_counts, cut=True)

        v = data_stellar_cut_x[k]
        stellar_masses = np.log10(v[data_key]["icl"] + v[data_key]["sm"])
        halo_masses = np.log10(v[data_key]["m"])
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v[fit_key])
        delta_halo_masses = halo_masses - predicted_halo_masses

        y, yerr = resample_scatter(stellar_masses, delta_halo_masses, sm_bins)

        ax.errorbar(number_densities_mid, y, yerr=yerr, label=l.m_star_legend(k),
                capsize=1.5, linewidth=1)

    # Plot Rozo2014 region
    minx = fits.density_at_richness(richness, 20) / sim_volume
    maxx = 1e-9 #fits.density_at_richness(richness, 30) / sim_volume
    line = ax.plot([minx, maxx], [0.11, 0.11], color=l.r2014,
            linestyle="dashed", label=r"$\lambda$ (est)")[0]
    ax.fill_between([minx, maxx], 0.09, 0.13, alpha=0.2, facecolor=line.get_color())

    ax.set(
            xscale="log",
            ylim=(0, 0.47),
            xlabel=l.cum_number_density,
            ylabel=l.hm_scatter,
            xlim=(2e-4, 1.2e-7),
    )
    ax.legend(fontsize="xx-small", loc="upper right")

    # Put the richness at the tops
    ax2 = ax.twiny()
    richness_counts = [5, 10, 20, 30]
    ticks = fits.density_at_richness(richness, richness_counts) / sim_volume
    ax2.set(
            xlim=np.log10(ax.get_xlim()),
            xticks=np.log10(ticks),
            xticklabels=[r"$" + str(i) + "$" for i in richness_counts],
            xlabel=l.ngals,
    )
    return ax, ax2
