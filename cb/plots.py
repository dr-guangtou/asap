import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import scipy.stats
import cluster_sum

solarMassUnits = r"($M_{\odot}$)"

# Be very careful with when you are in log and when not in log...
# All plotters should plot using log10(value)
# Whether they take in data in that format or convert it depends so keep track of that

def add_mean_and_stddev(ax, x, y, label=None):
    bins = np.arange(np.min(x), np.max(x), 0.2)
    mean, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=bins)
    std, _, _ = scipy.stats.binned_statistic(x, y, statistic=np.std, bins=bins)

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    ax.errorbar(
        bin_midpoints,
        mean,
        yerr=std,
        linestyle="None",
        marker=".",
        label=label,
        zorder=3)  # https://github.com/matplotlib/matplotlib/issues/1622


def richness_vs_scatter(centrals, satellites, min_mass_for_richness):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    halo_masses = np.log10(centrals["mp"])
    stellar_masses = np.log10(centrals["icl"] + centrals["sm"])
    smhm = stellar_masses / halo_masses
    richnesses = cluster_sum.get_richness(centrals, satellites, min_mass_for_richness)

    x_bin_edges = np.linspace(np.min(halo_masses), np.max(halo_masses), num=16 + 1)
    y_bin_edges = np.array([0, 1, 2, 4, 8, 16, 32, 64])
    heat, x_edge, y_edge, bin_number = scipy.stats.binned_statistic_2d(
            halo_masses,
            richnesses,
            smhm,
            bins=[x_bin_edges, y_bin_edges],
            statistic="std",
    )
    heat[heat == 0] = -np.inf
    image = ax.imshow(
            heat.T, # WHY???
            origin="lower",
            extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            aspect="auto",
    )
    ax.set(
            xticks=x_bin_edges,
            yticks=np.linspace(0, 64, num=8),
            yticklabels=[str(i) for i in y_bin_edges],
            xlabel=r"log $M_{vir}$" + solarMassUnits,
            ylabel=r"Richness",
            title="SMHM variance in HM and Richness bins",
    )
    fig.colorbar(image, label="SMHM variance")

    # Lets also get a sense of how much data we have in each bin
    fig2, ax2 = plt.subplots()
    x_ind, y_ind = np.unravel_index(bin_number,
                                (len(x_edge) + 1, len(y_edge) + 1))
    bin_counts, _, _= np.histogram2d(x_ind, y_ind, bins=[len(x_bin_edges)-1, len(y_bin_edges)-1])
    bin_counts = np.log10(bin_counts)
    image = ax2.imshow(
            bin_counts.T,
            origin="lower",
            extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            aspect="auto",
    )

    fig2.colorbar(image)
    return ax


def dm_vs_all_sm_error(catalogs, labels=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    label = None
    for i, catalog in enumerate(catalogs):
        x = np.log10(catalog["mp"])
        y = np.log10(catalog["icl"] + catalog["sm"])
        bins = np.arange(np.min(x), np.max(x), 0.2)
        std, _, _ = scipy.stats.binned_statistic(x, y, statistic=np.std, bins=bins)
        bin_midpoints = bins[:-1] + np.diff(bins) / 2
        if labels is not None:
            label = labels[i]
        ax.plot(bin_midpoints, std, label=label)
    ax.set(
        xlabel=r"log $M_{vir}$" + solarMassUnits,
        ylabel=r"SM-HM Scatter (dex)",
        title="Stellar Mass - Halo Mass scatter at varying halo masses with various numbers of sats",
    )
    ax.legend()
    return ax


def dm_vs_all_sm(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    x = np.log10(catalog["mp"])
    y = np.log10(catalog["icl"] + catalog["sm"])
    ax.plot(x, y, linestyle="None", marker="o", markersize=0.1)
    ax.set(
        xlabel=r"Peak HM " + solarMassUnits,
        ylabel=r"SM (all) " + solarMassUnits,
        title="Total SM vs Peak HM",
    )
    add_mean_and_stddev(ax, x, y)
    return ax


def dm_vs_insitu(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    x = np.log10(catalog["mp"])
    y = np.log10(catalog["sm"])
    ax.plot(x, y, linestyle="None", marker="o", markersize=0.1)
    ax.set(
        xlabel=r"Peak HM " + solarMassUnits,
        ylabel=r"SM (insitu) " + solarMassUnits,
        title="Insitu SM vs Peak HM",
    )
    add_mean_and_stddev(ax, x, y)
    return ax


def dm_vs_exsitu(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    x = np.log10(catalog["mp"])
    y = np.log10(catalog["icl"], where=catalog["icl"] > 0)
    ax.plot(x, y, linestyle="None", marker="o", markersize=0.1)
    ax.set(
        xlabel=r"Peak HM " + solarMassUnits,
        ylabel=r"SM (exsitu) " + solarMassUnits,
        title="Exsitu SM vs Peak HM",
    )
    add_mean_and_stddev(ax, x, y)
    return ax


def plotly_stuff(data, y_cols):
    scatters = [
        go.Scatter(
            visible=False,
            x=data[i]["mp"],
            y=np.sum(np.array([data[i][col] for col in y_cols]), axis=0),
            mode='markers',
        ) for i in range(len(data))
    ]
    scatters[0]["visible"] = True

    steps = []
    for i in range(len(data)):
        steps.append({
            "method": "restyle",
            "args": ["visible", [i == j for j in range(len(data))]]
        })
    sliders = [{"active": 0, "steps": steps}]

    layout = go.Layout(
        xaxis=dict(type='log', range=[11.9, 16]),
        yaxis=dict(type='log', range=[9, 14]),
        sliders=sliders,
    )
    return go.Figure(data=scatters, layout=layout)
