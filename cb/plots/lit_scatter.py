# Literature values of the scatter in SM at fixed HM
# Only care about HM > 13
import numpy as np

hm0, hm1, n = 13, 15, 50

def plot_lit(ax):
    lit_lines = []
    for _, v in lit2.items():
        line = ax.plot(v["x"], v["y"], label=v["label"], linestyle=v["ls"], color=v["color"])[0]
        ax.fill_between(v["x"], v["y"]-v["error"], v["y"]+v["error"], alpha=0.2, facecolor=v["color"])
        lit_lines.append(line)
    ax.add_artist(ax.legend(handles=lit_lines, loc="upper left", fontsize="xx-small"))
    return ax


def plot_lit_old(ax):
    lit_lines = []
    for _, v in lit.items():
        lit_lines.append(ax.plot(v["x"], v["y"], label=v["label"])[0])
    ax.add_artist(ax.legend(handles=lit_lines, loc="upper left", fontsize="xx-small"))
    return ax

lit2 = {
        "kravtsov2018a_cen": {
            "x": np.linspace(np.log10(5e13), np.log10(1.5e15), num=n),
            "y": np.array([0.17] * n),
            "error": 0.03,
            "label": r"Kravtsov2018 $M_{\ast,cen}$",
            "color": "orange",
            "ls": ":",
        },
        "kravtsov2018a_halo": {
            "x": np.linspace(np.log10(5e13), np.log10(1.5e15), num=n),
            "y": np.array([0.11] * n),
            "error": 0.03,
            "label": r"Kravtsov2018 $M_{\ast,halo}$",
            "color": "green",
            "ls": ":",
        },
        "lin2012": {
            "x": np.linspace(np.log10(8e13), np.log10(2e15), num=n),
            "y": np.array([0.12] * n),
            "error": 0.00, # No error on the scatter quoted
            "label": r"Lin2012 $M_{\ast,halo}$",
            "color": "green",
            "ls": "--",
        },
        "leauthaud2012": {
            "x": np.linspace(13, 15, num=n), # Check this!
            "y": np.array([0.192] * n),
            "error": 0.031,
            "label": r"Leauthaud2012 $M_{\ast,cen}$",
            "color": "orange",
            "ls": "--",
        },
        "tinker2013": { # I didn't understand this - just copied from gu
            "x": np.linspace(13, 15, num=n), # Check this!
            "y": np.array([0.28] * n),
            "error": 0.03,
            "label": r"Tinker2013 $M_{\ast,cen}$",
            "color": "orange",
            "ls": "-.",
        },
        # "gu2016": {
        #     "x": np.linspace(12, 15, num=n),
        #     "y": np.array([0.2] * n),
        #     "error": 0,
        #     "label": r"Gu2013 $M_{\ast,cen}$",
        #     "color": "b",
        #     "ls": "--",
        # },
}

lit = {
        # "gu2016": {
        #     "x": np.linspace(hm0, hm1, num=n),
        #     "y": np.array([0.2] * n),
        #     "label": "Gu2016",
        # },
        "kravtsov2018b": {
            "x": np.linspace(hm0, hm1, num=n),
            "y": np.array([0.1] * n),
            "label": r"Kravtsov2018 $M_{\ast,halo}$",
        },
        # Note that this is at slightly higher Z (0 - 0.6) than others
        # But they say there is no redshift evolution so that is OK
        "lin2012": {
            "x": np.linspace(14, np.log10(2e15), num=n),
            "y": np.array([0.12] * n),
            "label": "Lin2012",
        },
        "leauthaud2012": {
            "x": np.linspace(hm0, hm1, num=n),
            "y": np.array([0.23] * n),
            "label": "Leauthaud2012",
        },
}
