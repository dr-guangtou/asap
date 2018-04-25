import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import halo_info
from plots import labels as l
import smhm_fit

def age_and_hm_vs_sm_scatter(centrals, fit, n_sats):
    halo_masses = np.log10(centrals["m"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    ages = centrals["Halfmass_Scale"]
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.linspace(np.min(ages), np.max(ages), num = 8 + 1)
    fig, ax, image, binned_stats = generalised_heatmap(halo_masses, ages, stellar_masses, True, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.2f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_vir_x_axis,
            ylabel=r"Scale factor at halo half mass",
    )
    fig.colorbar(image, label=l.sm_scatter(n_sats))
    return ax

def age_and_sm_vs_hm_scatter(centrals, fit, n_sats):
    halo_masses = np.log10(centrals["m"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    ages = centrals["Halfmass_Scale"]
    x_bin_edges = np.arange(
            np.floor(10*np.min(stellar_masses))/10, # round down to nearest tenth
            np.max(stellar_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.linspace(np.min(ages), np.max(ages), num = 8 + 1)
    fig, ax, image, binned_stats = generalised_heatmap(stellar_masses, ages, halo_masses, False, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.2f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_star_x_axis(n_sats),
            ylabel=r"Scale factor at halo half mass",
    )
    fig.colorbar(image, label=l.hm_scatter)
    return ax

def mm_and_hm_vs_sm_scatter(centrals, fit, n_sats):
    halo_masses = np.log10(centrals["m"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    mm = centrals["scale_of_last_MM"]
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.linspace(np.min(mm), np.max(mm), num = 16 + 1)
    fig, ax, image, binned_stats = generalised_heatmap(halo_masses, mm, stellar_masses, True, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.2f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_vir_x_axis,
            ylabel=r"Scale factor at last major merger",
    )
    fig.colorbar(image, label=l.sm_scatter(n_sats))
    return ax

def mm_and_sm_vs_hm_scatter(centrals, fit, n_sats):
    halo_masses = np.log10(centrals["m"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    mm = centrals["scale_of_last_MM"]
    x_bin_edges = np.arange(
            np.floor(10*np.min(stellar_masses))/10, # round down to nearest tenth
            np.max(stellar_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.linspace(np.min(mm), np.max(mm), num = 16 + 1)
    fig, ax, image, binned_stats = generalised_heatmap(stellar_masses, mm, halo_masses, False, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.2f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_star_x_axis(n_sats),
            ylabel=r"Scale factor at last major merger",
    )
    fig.colorbar(image, label=l.hm_scatter)
    return ax

def concentration_and_hm_vs_sm_scatter(centrals, fit, n_sats):
    halo_masses = np.log10(centrals["m"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    concentrations = centrals["rvir"] / centrals["rs"]
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.arange(np.floor(np.min(concentrations)), np.ceil(np.max(concentrations)), 1)
    fig, ax, image, binned_stats = generalised_heatmap(halo_masses, concentrations, stellar_masses, True, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=binned_stats.y_edge[::2],
            yticklabels=["{0:.0f}".format(i) for i in binned_stats.y_edge[::2]],
            xlabel=l.m_vir_x_axis,
            ylabel=r"Concentration",
    )
    fig.colorbar(image, label=l.sm_scatter(n_sats))
    return ax

def concentration_and_sm_vs_hm_scatter(centrals, fit, n_sats):
    halo_masses = np.log10(centrals["m"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    concentrations = centrals["rvir"] / centrals["rs"]
    x_bin_edges = np.arange(
            np.floor(10*np.min(stellar_masses))/10, # round down to nearest tenth
            np.max(stellar_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.around(np.geomspace(np.min(concentrations), np.max(concentrations), num = 16 + 1), decimals=1)
    fig, ax, image, binned_stats = generalised_heatmap(stellar_masses, concentrations, halo_masses, False, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.1f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_star_x_axis(n_sats),
            ylabel=r"Concentration",
    )
    fig.colorbar(image, label=l.hm_scatter)
    return ax

def richness_and_hm_vs_sm_scatter(centrals, satellites, min_mass_for_richness, fit, n_sats):
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    halo_masses = np.log10(centrals["m"])
    richnesses = halo_info.get_richness(centrals, satellites, min_mass_for_richness)
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    fig, ax, image, binned_stats = generalised_heatmap(halo_masses, richnesses, stellar_masses, True, fit, x_bin_edges, y_bin_edges)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(0, 64, num=8),
            yticklabels=["{0:.0f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_vir_x_axis,
            ylabel=r"$Richness$",
    )
    fig.colorbar(image, label=l.sm_scatter(n_sats))
    return ax

def richness_and_sm_vs_hm_scatter(centrals, satellites, min_mass_for_richness, fit, n_sats):
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    halo_masses = np.log10(centrals["m"])
    richnesses = halo_info.get_richness(centrals, satellites, min_mass_for_richness)
    x_bin_edges = np.arange(
            np.floor(10*np.min(stellar_masses))/10, # round down to nearest tenth
            np.max(stellar_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    fig, ax, image, binned_stats = generalised_heatmap(stellar_masses, richnesses, halo_masses, False, fit, x_bin_edges, y_bin_edges)
    fig.colorbar(image, label=l.hm_scatter)
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(0, 64, num=8),
            yticklabels=["{0:.0f}".format(i) for i in binned_stats.y_edge],
            xlabel=l.m_star_x_axis(n_sats),
            ylabel=r"$Richness$",
    )
    return ax


# This takes some mass on the x axis, some quantity on the y axis and plots the scatter in the other mass
# as color
def generalised_heatmap(x_masses, y_quantity, other_masses, x_is_halo, fit, x_bin_edges, y_bin_edges):
    if np.min(x_masses) > 100 or np.min(other_masses) > 100:
        raise Exception("You are probably not passing log masses!")
    fig, ax = plt.subplots()

    if x_is_halo:
        predicted_other_masses = smhm_fit.f_shmr(x_masses, *fit)
    else:
        predicted_other_masses = smhm_fit.f_shmr_inverse(x_masses, *fit)
    delta_other_masses = other_masses - predicted_other_masses
    assert np.mean(delta_other_masses) < 0.05, "prediction should be ~ the mean"

    binned_stats = scipy.stats.binned_statistic_2d(
            x_masses,
            y_quantity,
            delta_other_masses,
            bins=[x_bin_edges, y_bin_edges],
            statistic="std", # I don't think that this is quite right...
    )

    # Invalidate bins that don't have many members
    binned_stats = invalidate_unoccupied_bins(binned_stats)

    # Plot and add labels, colorbar, etc
    image = ax.imshow(
            binned_stats.statistic.T, # Still don't know why this is Transposed
            origin="lower",
            extent=[binned_stats.x_edge[0], binned_stats.x_edge[-1], binned_stats.y_edge[0], binned_stats.y_edge[-1]],
            aspect="auto",
    )
    return fig, ax, image, binned_stats

def invalidate_unoccupied_bins(binned_stats, c = 5):
    print("Invalidating < {}".format(c))
    x_ind, y_ind = np.unravel_index(binned_stats.binnumber, (len(binned_stats.x_edge) + 1, len(binned_stats.y_edge) + 1))
    bin_counts, _, _ = np.histogram2d(x_ind, y_ind, bins=[len(binned_stats.x_edge)-1, len(binned_stats.y_edge)-1])

    # fig, ax = plt.subplots()
    # image = ax.imshow(
    #         np.log10(bin_counts.T),
    #         origin="lower",
    # )
    # fig.colorbar(image)
    # Could sum things here to find the next extents (if we remove some data at the high or low end our graph looks a bit silly.

    # Could do this better by doing some resampling and only keeping if it is well defined
    # Though I'm not sure we have the data here to make that easy...
    binned_stats.statistic[bin_counts < c] = -np.inf
    return binned_stats
