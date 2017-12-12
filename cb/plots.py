import matplotlib.pyplot as plt
def dm_vs_insitu(catalog, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(catalog["mp"], catalog["sm"], s=0.1)
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
