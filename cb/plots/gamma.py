import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from .heatmaps import invalidate_unoccupied_bins
from plots import labels as l

def _imshow(ax, binned_stats, **kwargs):
    return ax.imshow(
            binned_stats.statistic.T,
            origin="lower",
            extent=[binned_stats.x_edge[0], binned_stats.x_edge[-1], binned_stats.y_edge[0], binned_stats.y_edge[-1]],
            aspect="auto",
            **kwargs,
    )


def one(mag_gap, gammas):
    _, ax = plt.subplots()

    x = ax.hist2d(
            np.log10(mag_gap),
            gammas,
            bins=20,
            cmap="OrRd",
            norm=mpl.colors.LogNorm(),
            range=[[7, 13], [0, 5]]
    )

    mg_bins = x[1]
    mg_bins_cen = mg_bins[:-1] + ((mg_bins[1:] - mg_bins[:-1]) / 2)
    s = scipy.stats.binned_statistic(np.log10(mag_gap), gammas, statistic="mean", bins=mg_bins)

    ax.plot(mg_bins_cen, s.statistic)

    # std = scipy.stats.binned_statistic(np.log10(mag_gap), gammas, statistic="std", bins=mg_bins)
    #ax.fill_between(mg_bins_cen, s.statistic - std.statistic, s.statistic + std.statistic, alpha=0.2)
    ax.set(
        xlabel="Log10(mass gap)",
        ylabel="Gamma",
        ylim=(0, 5),
    )

    return ax


def two(masses, mag_gap, gammas):
    fig, ax = plt.subplots()
    s = scipy.stats.binned_statistic_2d(np.log10(masses), np.log10(mag_gap), gammas, statistic="mean", bins=(10, 20))
    s = invalidate_unoccupied_bins(s)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        xlabel=l.m_star_x_axis("cen"),
        ylabel="Log10(mass gap)",
    )
    fig.colorbar(img, ax=ax)

def gamma_in_mstarcen_mstarhalo_bins(j_cen_mass, j_halo_mass, j_gammas):
    fig, ax = plt.subplots()
    s = scipy.stats.binned_statistic_2d(np.log10(j_cen_mass), np.log10(j_halo_mass), j_gammas, statistic="median", bins=(10, 10))
    s = invalidate_unoccupied_bins(s, 15)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        xlabel=l.m_star_x_axis("cen"),
        ylabel=l.m_star_x_axis("tot"),
        xlim=(np.log10(np.min(j_cen_mass)), 12.5),
        ylim=(np.log10(np.min(j_halo_mass)), 13),
    )
    fig.colorbar(img, ax=ax, label=l.gamma2)

    std = scipy.stats.binned_statistic_2d(np.log10(j_cen_mass), np.log10(j_halo_mass), j_gammas, statistic="std", bins=(10, 10))
    std = invalidate_unoccupied_bins(std, 15)
    std = std.statistic.flatten()
    std = std[np.isfinite(std)]
    print(np.median(std))
    return ax

def gamma_in_mstarcen_mstarhalo(j_cen_mass, j_halo_mass, j_gammas):
    fig, ax = plt.subplots()

    cen_selected = (j_cen_mass.values > 10**11.6) & (j_cen_mass.values < 10**11.7)
    num = 500* 1
    low_tot_selected = (j_halo_mass.values < 10**11.7)
    high_tot_selected = (j_halo_mass.values > 10**12.15)

    low = j_gammas.values[cen_selected & low_tot_selected]
    high = j_gammas.values[cen_selected & high_tot_selected]
    print(len(low))
    print(len(high))

    ax.hist(low, density=True, alpha=0.4, bins=50, label="Lowest " + l.m_star_legend("tot") + " systems")
    ax.hist(high, density=True, alpha=0.4, bins=50, label="Highest " + l.m_star_legend("tot") + " systems")
    ax.set(
            xlim=(-2, 6),
            xlabel=l.gamma2,
            ylabel="Density",
    )

    ax.legend(fontsize="xx-small")
