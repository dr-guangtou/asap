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
    s = scipy.stats.binned_statistic_2d(np.log10(j_cen_mass), np.log10(j_halo_mass), j_gammas, statistic="mean", bins=(10, 10))
    s = invalidate_unoccupied_bins(s, 5)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        xlabel=l.m_star_x_axis("cen"),
        ylabel=l.m_star_x_axis("tot"),
    )
    fig.colorbar(img, ax=ax, label=l.gamma)
    return ax
