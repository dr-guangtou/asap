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

def mhalo_gamma_richness(cutz):
    g = "gammas2"
    cutz2 = cutz[np.isfinite(cutz[g])]

    def _perc(data):
        cutoff = 3
        return np.count_nonzero(data > cutoff) / len(data)

    fig, ax = plt.subplots()
    s = scipy.stats.binned_statistic_2d(np.log10(cutz2["m"]), cutz2[g], cutz2["richness"], statistic="mean", bins=(25, 25))
    s = invalidate_unoccupied_bins(s, 12)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.gamma2,
        xlim=(13.5, 14.75),
        ylim=(0, 6),
    )
    clb = fig.colorbar(img, ax=ax, label=l.ngals)

    contours = ax.contour(
            s.x_edge[:-1] + ((s.x_edge[1:] - s.x_edge[:-1]) / 2),
            s.y_edge[:-1] + ((s.y_edge[1:] - s.y_edge[:-1]) / 2),
            s.statistic.T,
            levels=[1, 2, 4, 6, 8, 10, 13, 16, 19],
            # levels=np.arange(20),
            cmap="copper_r",
    )
    clb.add_lines(contours)

    # Plot applox uncertainties on contours
    ax.plot([14.4, 14.6], [4, 4], color="black", marker="|")
    ax.annotate("Approximate size of the\nuncertainties on the contours",
            [14.1, 4.2],
            fontsize=8,
            )

    # We need to know the std
    fig, ax = plt.subplots()
    s = scipy.stats.binned_statistic_2d(np.log10(cutz2["m"]), cutz2[g], cutz2["richness"], statistic="std", bins=(25, 25))
    s = invalidate_unoccupied_bins(s, 12)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.gamma2,
        xlim=(13.5, 14.75),
        ylim=(0, 6),
    )
    clb = fig.colorbar(img, ax=ax, label=l.ngals)

def mhalo_gamma_richness_with_interp(cutz):
    g = "gammas2"
    cutz2 = cutz[np.isfinite(cutz[g])]

    # Now lets do it with interpolation
    # Somehow need to get training samples in the non square space
    # This won't work as is because i just pulled it from the above function and haven't yet fixed it. cbx
    fig, ax = plt.subplots()
    x_centers = s.x_edge[:-1] + ((s.x_edge[1:] - s.x_edge[:-1]) / 2)
    y_centers = s.y_edge[:-1] + ((s.y_edge[1:] - s.y_edge[:-1]) / 2)
    non_nan = np.copy(s.statistic)
    non_nan[np.isinf(s.statistic)] = 0
    print(non_nan.shape, y_centers.shape)
    print(non_nan)
    interp_f = scipy.interpolate.interp2d(
            x_centers,
            y_centers,
            non_nan.T,
            kind="cubic",
    )

    x = np.linspace(13, 15, num=500)
    y = np.linspace(0, 8, num=500)
    z = interp_f(x, y)
    img = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="OrRd",
    )
    clb = fig.colorbar(img, ax=ax, label="Mean number of satellite galaxys $>$ 0.2M*")
    contours = ax.contour(
            x,
            y,
            z,
            levels=[0, 0.5, 1, 2, 3.7, 5.5, 10, 13, 16, 19],
            # levels=np.arange(20),
            cmap="copper_r",
            )
    clb.add_lines(contours)



    # And just do a scatter plot for fun...
    """
    fig, ax = plt.subplots()

    low = cutz2["richness"] < 10
    ax.scatter(np.log10(cutz2["m"][low]), cutz2[g][low], color="b", s=0.3)
    ax.scatter(np.log10(cutz2["m"][np.logical_not(low)]), cutz2[g][np.logical_not(low)], color="r", s=0.3)
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.gamma,
        xlim=(13.5, 15),
        ylim=(-1, 7.5),
    )
    """

    return ax

def gamma_in_mvir_mstarx_bins(mvir, mstar, gamma, x):
    fig, ax = plt.subplots()
    s = scipy.stats.binned_statistic_2d(
            np.log10(mvir),
            np.log10(mstar),
            gamma,
            statistic="median",
            bins=(10, 10)
    )
    s = invalidate_unoccupied_bins(s, 15)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.m_star_x_axis(x),
        xlim=(np.log10(np.min(mvir)), np.log10(np.max(mvir))),
        ylim=(np.log10(np.min(mstar)), np.log10(np.max(mstar))),
    )
    fig.colorbar(img, ax=ax, label=l.gamma2)
    return ax


# was gamma_in_mstarcen_mstarhalo_bins
def gamma_in_mstarcen_mstartot_bins(mcen, mtot, gamma):
    fig, ax = plt.subplots()
    s = scipy.stats.binned_statistic_2d(np.log10(mtot), np.log10(mcen), gamma, statistic="median", bins=(10, 10))
    s = invalidate_unoccupied_bins(s, 15)
    img = _imshow(ax, s, cmap="OrRd")
    ax.set(
        ylabel=l.m_star_x_axis("cen"),
        xlabel=l.m_star_x_axis("tot"),
        ylim=(np.log10(np.min(mcen)), 13),
        xlim=(np.log10(np.min(mtot)), 13),
    )
    fig.colorbar(img, ax=ax, label=l.gamma2)

    std = scipy.stats.binned_statistic_2d(np.log10(mtot), np.log10(mcen), gamma, statistic="std", bins=(10, 10))
    std = invalidate_unoccupied_bins(std, 15)
    std = std.statistic.flatten()
    std = std[np.isfinite(std)]
    print(np.median(std))
    return ax

# was gamma_in_mstarcen_mstarhalo
def gamma_of_fixed_cen_and_split_tot_pops(mcen, mtot, gamma):
    assert len(mcen) == len(mtot)
    fig, ax = plt.subplots()

    in_cen_range = ((mcen > 10**11.6) & (mcen < 10**11.7)).values
    mcen, mtot, gamma = mcen[in_cen_range], mtot[in_cen_range], gamma[in_cen_range]

    mtot_minus_mcen = (np.log10(mtot) - np.log10(mcen)).values

    low_cut, high_cut = np.percentile(mtot_minus_mcen, [20, 80])
    print(low_cut, high_cut)

    in_low_tot_range = (mtot_minus_mcen <= low_cut)
    in_high_tot_range = (mtot_minus_mcen >= high_cut)

    low = gamma.values[in_low_tot_range]
    high = gamma.values[in_high_tot_range]
    print(len(low))
    print(len(high))

    diff = l.m_star_legend("tot") + r"-" + l.m_star_legend("cen")
    ax.hist(low, density=True, alpha=0.4, bins=50, label="Small " + diff)
    ax.hist(high, density=True, alpha=0.4, bins=50, label="Large " + diff)
    ax.set(
            xlim=(-1, 6),
            xlabel=l.gamma,
            ylabel="Density",
    )

    ax.legend(fontsize="small")
