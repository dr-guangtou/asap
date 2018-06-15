import numpy as np
import matplotlib.pyplot as plt

from plots.utils import resample_scatter, _bins_mid
import smhm_fit
import fits

sim_volume = 400**3

cum_counts = np.logspace(2, 4.1, num=10)
cum_counts_mid = _bins_mid(cum_counts)
number_densities_mid = cum_counts_mid / sim_volume

def explore_cylinder_length(catalog):
    _, ax = plt.subplots()

    for k, v in catalog.items():
        if int(k) > 40: continue
        sm_bins = fits.mass_at_density(catalog[k], cum_counts, True)
        y, yerr = _get_scatter(v, sm_bins)
        ax.errorbar(number_densities_mid, y, yerr=yerr, label=k)

    ax.legend()
    ax.invert_xaxis()
    ax.set(
            xscale="log",
    )

def parabola_me(catalog):
    _, ax = plt.subplots()

    for i in range(4, 10):
        scatter = []
        x = []
        for k, v in catalog.items():
            if int(k) > 40: continue
            sm_bins = fits.mass_at_density(catalog[k], cum_counts, True)
            y, _ = _get_scatter(v, sm_bins)
            scatter.append(y[i])
            x.append(int(k))

        ax.plot(x, scatter)
    ax.set(
            ylabel="Scatter",
            xlabel="Cylinder length",
    )

def _get_scatter(v, sm_bins):
    data_key, fit_key = "data_cut", "fit_cut"
    halo_masses = np.log10(v[data_key]["m"])
    stellar_masses = np.log10(v[data_key]["icl"] + v[data_key]["sm"])
    predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v[fit_key])
    delta_halo_masses = halo_masses - predicted_halo_masses

    return resample_scatter(stellar_masses, delta_halo_masses, sm_bins)

"""
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
    print(k, y)

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
"""
