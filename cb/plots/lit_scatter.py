# Literature values of the scatter in SM at fixed HM
# Only care about HM > 13
import numpy as np

hm0, hm1, n = 13, 15, 50

gu_vir = np.array([12.875, 13.125, 13.375, 13.625, 13.875, 14.125, 14.375, 14.625])
gu_z0 = np.array([0.2618758278, 0.2495733992, 0.2417748481, 0.2246581009, 0.2081620779, 0.2060492773, 0.1914180345, 0.1747871659])
gu_z0_err = np.array([0.005115550329, 0.006947900755, 0.009507300102, 0.0109303983, 0.01258421478, 0.01910521258, 0.02469421233, 0.02965986181])


# Think we will use z = 0
# gu_z1 = [0.2826113807, 0.2778448604, 0.2669389574, 0.2596907156, 0.23765541, 0.2268705727]
# gu_z1_err = [0.008023125012, 0.0108067776, 0.01538881487, 0.02270312572, 0.03694499255, 0.05172502909]

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
        # "leauthaud2012": {
        #     "x": np.linspace(13, 15, num=n), # Check this!
        #     "y": np.array([0.192] * n),
        #     "error": 0.031,
        #     "label": r"Leauthaud2012 $M_{\ast,cen}$",
        #     "color": "orange",
        #     "ls": "-",
        # },
        "gu2016": {
            "x": gu_vir,
            "y": gu_z0,
            "error": gu_z0_err,
            "label": r"Gu2016 $M_{\ast,cen}$",
            "color": "orange",
            "ls": "--",
        },
        "tinker2013": { # I didn't understand this - just copied from gu
            "x": np.linspace(13, 15, num=n), # Check this!
            "y": np.array([0.21] * n),
            "error": 0.03,
            "label": r"Tinker2013 $M_{\ast,cen}$",
            "color": "orange",
            "ls": "-.",
        },
        "zu2015": {
            "x": np.array([12, 14]),
            "y": np.array([0.22, 0.18]),
            "error": 0.01,
            "label": r"Zu2015 $M_{\ast,cen}$",
            "color": "orange",
            "ls": "-",
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
