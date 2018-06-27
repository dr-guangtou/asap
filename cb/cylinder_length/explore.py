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

    all_scatters = []
    for i in range(0, 9):
        scatter = []
        for k, v in catalog.items():
            sm_bins = fits.mass_at_density(catalog[k], cum_counts, True)
            y, _ = _get_scatter(v, sm_bins)
            scatter.append(y[i])

        all_scatters.append(np.array(scatter))
        ax.plot([int(k) for k in catalog.keys()], scatter)

    ax.plot([int(k) for k in catalog.keys()], np.mean(np.array(all_scatters), axis=0), label="mean")
    ax.set(
            ylabel="Scatter",
            xlabel="Cylinder length",
    )
    ax.legend()
    return ax

def _get_scatter(v, sm_bins):
    data_key, fit_key = "data_cut", "fit_cut"
    halo_masses = np.log10(v[data_key]["m"])
    stellar_masses = np.log10(v[data_key]["icl"] + v[data_key]["sm"])
    try:
        predicted_halo_masses = smhm_fit.f_shmr_inverse(stellar_masses, *v[fit_key])
    except KeyError as e:
        print("No fit, just binning and taking the mean")
        raise e


    delta_halo_masses = halo_masses - predicted_halo_masses

    return resample_scatter(stellar_masses, delta_halo_masses, sm_bins)


# def _window_halo_masses(stellar_masses, halo_masses):
#     sorting_indexes = np.argsort(stellar_masses) # increasing
#     s_sm = stellar_masses[sorting_indexes]
#     s_hm = halo_masses[sorting_indexes]
