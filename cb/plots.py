import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import scipy.stats as sps

solarMassUnits = r"($M_{\odot}$)"

def add_scatter_plot(ax, x, y):
    # We do this entirely in log space
    bins = np.arange(np.min(x), np.max(x), 0.2)
    mean, _, _ = sps.binned_statistic(x, y, statistic = "mean", bins = bins)
    std, _, _ = sps.binned_statistic(x, y, statistic = np.std, bins = bins)

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    # And then raise to the power before plotting
    ax.errorbar(bin_midpoints, mean, yerr = std, color = "red", marker = "o", fmt = ".")
    print(std)

def dm_vs_insitu(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    x = np.log10(catalog["mp"])
    y = np.log10(catalog["sm"])
    ax.scatter(x, y, s = 0.01)
    ax.set(
            xlabel = r"Peak HM " + solarMassUnits,
            ylabel = r"SM (insitu) " + solarMassUnits,
            title = "Insitu SM vs Peak HM",
    )
    add_scatter_plot(ax, x, y)
    return ax

def dm_vs_exsitu(catalog, ax = None):
    if ax is None:
        _, ax = plt.subplots()
    x = np.log10(catalog["mp"])
    y = np.log10(catalog["icl"], where = catalog["icl"] > 0)
    ax.scatter(x, y, s = 0.1)
    ax.set(
            xlabel = r"Peak HM " + solarMassUnits,
            ylabel = r"SM (exsitu) " + solarMassUnits,
            title = "Exsitu SM vs Peak HM",
    )
    add_scatter_plot(ax, x, y)
    return ax

def dm_vs_all_sm(catalog, ax = None):
    if ax is None:
        _, ax = plt.subplots()
    x = np.log10(catalog["mp"])
    y = np.log10(catalog["icl"] + catalog["sm"])
    ax.scatter(x, y, s = 0.1)
    ax.set(
            xlabel = r"Peak HM " + solarMassUnits,
            ylabel = r"SM (all) " + solarMassUnits,
            title = "Total SM vs Peak HM",
    )
    add_scatter_plot(ax, x, y)
    return ax


def plotly_stuff(data, y_cols):
    scatters = [
            go.Scatter(
                visible = False,
                x = data[i]["mp"],
                y = np.sum(np.array([data[i][col] for col in y_cols]), axis=0),
                mode = 'markers',
            ) for i in range(len(data))
    ]
    scatters[0]["visible"] = True

    steps = []
    for i in range(len(data)):
        steps.append({"method": "restyle", "args": ["visible", [i == j for j in range(len(data))]]})
    sliders = [{"active": 0, "steps": steps}]

    layout = go.Layout(
        xaxis = dict(
            type = 'log',
            range = [11.9, 16]
        ),
        yaxis = dict(
            type = 'log',
            range = [9, 14]
        ),
        sliders = sliders,
    )
    return go.Figure(data = scatters, layout = layout)
