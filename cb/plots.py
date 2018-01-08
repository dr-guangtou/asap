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


def richness_vs_scatter(binned_halos, satellites, min_mass_for_richness, labels = None):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    label = None

    for i, current_halos in enumerate(binned_halos):
        stellar_masses = np.log10(current_halos["icl"] + current_halos["sm"])
        richness = cluster_sum.get_richness(current_halos, satellites, min_mass_for_richness)
        bins = np.arange(np.min(richness), np.max(richness))
        # mean, _, _ = scipy.stats.binned_statistic(richness, stellar_masses, statistic="mean", bins=bins)
        std, _, _ = scipy.stats.binned_statistic(richness, stellar_masses, statistic=np.std, bins=bins)

        # ax.plot(richness, stellar_masses, linestyle="None", marker="o", markersize=0.1)
        bin_midpoints = bins[:-1] + np.diff(bins) / 2
        if labels is not None:
            label = labels[i]
        ax.plot(bin_midpoints, std, label=label)
    ax.set(
        xlabel=r"Richness (# of satellites with total SM > {})".format(np.log10(min_mass_for_richness)),
        ylabel=r"SM-HM Scatter (dex)",
        title="Total Stellar Mass - Halo Mass scatter at fixed halo mass vs Richness",
    )
    ax.legend()
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
