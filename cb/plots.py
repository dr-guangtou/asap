import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
import cluster_sum

solarMassUnits = r"($M_{\odot}$)"
m_vir_x_axis = r"$M_{vir,peak}\ [log\ M_{vir,peak}/M_{\odot}]$"
smhm_ratio_scatter = r"$\sigma\ [log\ M_{*}/M_{vir,peak}]$"

# Be very careful with when you are in log and when not in log...
# All plotters should plot using log10(value)
# Whether they take in data in that format or convert it depends so keep track of that

def age_vs_scatter(centrals):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    halo_masses = np.log10(centrals["mp"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    smhm = stellar_masses / halo_masses
    age = centrals["Halfmass_Scale"]

    x_bin_edges = np.linspace(np.min(halo_masses), np.max(halo_masses), num=16 + 1)
    y_bin_edges = np.geomspace(np.min(age), np.max(age), num = 16 + 1)
    heat, x_edge, y_edge, _ = scipy.stats.binned_statistic_2d(
            halo_masses,
            age,
            smhm,
            bins=[x_bin_edges, y_bin_edges],
            statistic="std",
    )
    heat[heat == 0] = -np.inf
    image = ax.imshow(
            heat.T,
            origin="lower",
            extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            aspect="auto",
    )
    ax.set(
            xticks=x_bin_edges,
            yticks=np.linspace(np.min(y_bin_edges), np.max(y_bin_edges), num=16 + 1),
            yticklabels=["{0:.2f}".format(i) for i in y_bin_edges],
            xlabel=r"log $M_{vir}$" + solarMassUnits,
            ylabel=r"Scale factor at half mass",
            title="SMHM variance in HM and age bins",
    )
    fig.colorbar(image, label="SMHM variance")
    return ax


def mm_vs_scatter(centrals):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # Define the basic quantities we are interested in
    halo_masses = np.log10(centrals["mp"])
    smhm_ratio = np.log10(
            (centrals["icl"] + centrals["sm"]) / centrals["mp"]
    )
    mm = centrals["scale_of_last_MM"]
    print(len(np.unique(mm)))
    print(np.min(mm), np.max(mm))

    # Bin based on halo mass on the x axis and concentration on the y axis
    # Calculate the std-dev (scatter) of the smhm_ratio in each 2d bin
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.linspace(np.min(mm), np.max(mm), num = 16 + 1)
    binned_stats = scipy.stats.binned_statistic_2d(
            halo_masses,
            mm,
            smhm_ratio,
            bins=[x_bin_edges, y_bin_edges],
            statistic="std",
    )

    # Invalidate bins that don't have many members
    binned_stats = invalidate_unoccupied_bins(binned_stats)
    # binned_stats.statistic[binned_stats.statistic == 0] = -np.inf

    # Plot and add labels, colorbar, etc
    image = ax.imshow(
            binned_stats.statistic.T,
            origin="lower",
            extent=[binned_stats.x_edge[0], binned_stats.x_edge[-1], binned_stats.y_edge[0], binned_stats.y_edge[-1]],
            aspect="auto",
    )
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.2f}".format(i) for i in binned_stats.y_edge],
            xlabel=m_vir_x_axis,
            ylabel=r"Concentration",
            title="SMHM Ratio Scatter binned by Concentration and $M_{vir,peak}$",
    )
    fig.colorbar(image, label=smhm_ratio_scatter)
    return ax


def concentration_vs_scatter(centrals):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # Define the basic quantities we are interested in
    halo_masses = np.log10(centrals["mp"])
    smhm_ratio = np.log10(
            (centrals["icl"] + centrals["sm"]) / centrals["mp"]
    )
    concentrations = centrals["rvir"] / centrals["rs"]

    # Bin based on halo mass on the x axis and concentration on the y axis
    # Calculate the std-dev (scatter) of the smhm_ratio in each 2d bin
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.around(np.geomspace(np.min(concentrations), np.max(concentrations), num = 16 + 1), decimals=1)
    binned_stats = scipy.stats.binned_statistic_2d(
            halo_masses,
            concentrations,
            smhm_ratio,
            bins=[x_bin_edges, y_bin_edges],
            statistic="std",
    )

    # Invalidate bins that don't have many members
    binned_stats = invalidate_unoccupied_bins(binned_stats)

    # Plot and add labels, colorbar, etc
    image = ax.imshow(
            binned_stats.statistic.T,
            origin="lower",
            extent=[binned_stats.x_edge[0], binned_stats.x_edge[-1], binned_stats.y_edge[0], binned_stats.y_edge[-1]],
            aspect="auto",
    )
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(np.min(binned_stats.y_edge), np.max(binned_stats.y_edge), num=len(binned_stats.y_edge)),
            yticklabels=["{0:.1f}".format(i) for i in binned_stats.y_edge],
            xlabel=m_vir_x_axis,
            ylabel=r"Concentration",
            title="SMHM Ratio Scatter binned by Concentration and $M_{vir,peak}$",
    )
    fig.colorbar(image, label=smhm_ratio_scatter)
    return ax

def richness_vs_scatter(centrals, satellites, min_mass_for_richness):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # Define the basic quantities we are interested in
    halo_masses = np.log10(centrals["mp"])
    smhm_ratio = np.log10(
            (centrals["icl"] + centrals["sm"]) / centrals["mp"]
    )
    richnesses = cluster_sum.get_richness(centrals, satellites, min_mass_for_richness)

    # Bin based on halo mass on the x axis and richness on the y axis
    # Manually set up the bin edges because it is a bit tricky
    # Calculate the std-dev (scatter) of the smhm_ratio in each 2d bin
    x_bin_edges = np.arange(
            np.floor(10*np.min(halo_masses))/10, # round down to nearest tenth
            np.max(halo_masses) + 0.2, # to ensure that the last point is included
            0.2)
    y_bin_edges = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    binned_stats = scipy.stats.binned_statistic_2d(
            halo_masses,
            richnesses,
            smhm_ratio,
            bins=[x_bin_edges, y_bin_edges],
            statistic="std",
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
    ax.set(
            xticks=binned_stats.x_edge[::2],
            xticklabels=["{0:.1f}".format(i) for i in binned_stats.x_edge[::2]],
            yticks=np.linspace(0, 64, num=8),
            yticklabels=["{0:.0f}".format(i) for i in binned_stats.y_edge],
            xlabel=m_vir_x_axis,
            ylabel=r"$Richness\ [N_{sats}(log\ M_{*}/M_{\odot} > 10.8)]$",
            title="SMHM Ratio Scatter binned by Richness and $M_{vir,peak}$",
    )
    fig.colorbar(image, label=smhm_ratio_scatter)
    return ax

def invalidate_unoccupied_bins(binned_stats):
    x_ind, y_ind = np.unravel_index(binned_stats.binnumber, (len(binned_stats.x_edge) + 1, len(binned_stats.y_edge) + 1))
    bin_counts, _, _ = np.histogram2d(x_ind, y_ind, bins=[len(binned_stats.x_edge)-1, len(binned_stats.y_edge)-1])

    # Could sum things here to find the next extents (if we remove some data at the high or low end our graph looks a bit silly.

    # Could do this better by doing some resampling and only keeping if it is well defined
    # Though I'm not sure we have the data here to make that easy...
    binned_stats.statistic[bin_counts < 5] = -np.inf
    return binned_stats

def dm_vs_all_sm_error(catalogs, x_axis, labels=None, ):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    label = None
    for i, catalog in enumerate(catalogs):
        smhm_ratio = (catalog["icl"] + catalog["sm"]) / catalog["mp"]
        y = np.log10(smhm_ratio)
        if x_axis == "sm":
            x = np.log10(catalog["icl"] + catalog["sm"])
        elif x_axis == "hm":
            x = np.log10(catalog["mp"])
        else:
            raise Exception("x_axis must be 'sm' or 'hm', got {}".format(x_axis))

        bins = np.arange(np.min(x), np.max(x), 0.2)
        std, _, _ = scipy.stats.binned_statistic(x, y, statistic="std", bins=bins)
        bin_midpoints = bins[:-1] + np.diff(bins) / 2
        if labels is not None:
            label = labels[i]
        ax.plot(bin_midpoints, std, label=label)
    if x_axis == "sm":
        ax.set(
            xlabel=r"$M_{*}\ [log\ M_{*}/M_{\odot}]$",
            ylabel=r"$\sigma\ [log\ M_{*}/M_{vir,peak}]$",
            title="Scatter in Total Stellar Mass - Peak Halo Mass Ratio",
        )
    elif x_axis == "hm":
        ax.set(
            xlabel=m_vir_x_axis,
            ylabel=smhm_ratio_scatter,
            title="Scatter in Total Stellar Mass - Peak Halo Mass Ratio",
        )
    ax.legend()
    return ax


def dm_vs_sm(catalog, ax=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = np.log10(catalog["icl"] + catalog["sm"])
    # x = np.log10(0.1*catalog["icl"] + catalog["sm"])
    y = np.log10(catalog["m"])

    # Find various stats on our data
    sm_bin_edges = np.arange(np.min(x), np.max(x), 0.2)
    sm_bin_midpoints = sm_bin_edges[:-1] + np.diff(sm_bin_edges) / 2
    mean_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=sm_bin_edges)
    std_hm, _, _ = scipy.stats.binned_statistic(x, y, statistic="std", bins=sm_bin_edges)

    # Plot data and colored error regions
    ax.plot(sm_bin_midpoints, mean_hm, marker="o")
    ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm+std_hm, alpha=0.5, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm-std_hm, mean_hm-(2*std_hm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm+std_hm, mean_hm+(2*std_hm), alpha=0.25, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm-(2*std_hm), mean_hm-(3*std_hm), alpha=0.125, facecolor="tab:blue")
    ax.fill_between(sm_bin_midpoints, mean_hm+(2*std_hm), mean_hm+(3*std_hm), alpha=0.125, facecolor="tab:blue")
    ax.set(
        xlabel=r"$M_{*}\ [log\ M_{*}/M_{\odot}]$",
        ylabel=r"$M_{vir}\ [log\ M_{vir}/M_{\odot}]$",
    )

    # Plot with the default values from the paper
    m1 = 12.73
    sm0 = 11.04
    beta = 0.47
    delta = 0.60
    gamma = 1.96
    # mh = f_shmr(sm_bin_midpoints, m1, sm0, beta, delta, gamma)
    # ax.plot(sm_bin_midpoints, mh)

    # Now try fit
    popt, pcov = scipy.optimize.curve_fit(
            f_shmr,
            sm_bin_midpoints,
            mean_hm,
            p0=[m1, sm0, beta, delta, gamma],
            # bounds=(
            #     [5, sm0, beta-1e-9, -np.inf, -np.inf],
            #     [m1, 13, beta+1e-9, np.inf, np.inf],
            # ),
            # m1 will be smaller because we have total stellar mass (not galaxy mass). So smaller halos will have galaxies of mass M
            # smo will be larger for a similar reason as ^
            # beta should be unchanged - it only affects the low mass end
            # delta has large freedom
    )
    print(m1, sm0, beta, delta, gamma)
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    # print(f_shmr(sm_bin_midpoints, m1, sm0, beta, delta, gamma))
    # print(f_shmr(sm_bin_midpoints, *popt))
    ax.plot(sm_bin_midpoints, f_shmr(sm_bin_midpoints, *popt))

    return ax

# The functional form from https://arxiv.org/pdf/1103.2077.pdf
# This is the fitting function
def f_shmr(stellar_masses, m1, sm0, beta, delta, gamma):
    usm = stellar_masses / sm0 # unitless stellar mass is sm / characteristic mass
    return (m1 +
        beta * usm +
        ((usm**delta) / (1 + usm**-gamma)) -
        0.5)
