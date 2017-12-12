import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def dm_vs_insitu(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(catalog["mp"], catalog["sm"], s=0.01)
    ax.set(
            xscale="log",
            yscale="log",
            xlabel="Peak DM mass",
            ylabel="Insitu Stellar Mass",
            title="Peak halo mass vs insitu stellar mass",
    )
    return ax

def dm_vs_exsitu(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(catalog["mp"], catalog["icl"], s=0.1)
    ax.set(
            xscale="log",
            yscale="log",
            xlabel="Peak DM mass",
            ylabel="Exsitu Stellar Mass",
            title="Peak halo mass vs exsitu stellar mass",
    )
    return ax

def dm_vs_all_sm(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(catalog["mp"], catalog["icl"] + catalog["sm"], s=0.1)
    ax.set(
            xscale="log",
            yscale="log",
            xlabel="Peak DM mass",
            ylabel="Stellar Mass",
            title="Peak halo mass vs total stellar mass",
    )
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
        xaxis=dict(
            type='log',
            range=[11.9, 16]
        ),
        yaxis=dict(
            type='log',
            range=[9, 14]
        ),
        sliders=sliders,
    )
    return go.Figure(data=scatters, layout=layout)
