import numpy as np

def get_mag_gap(centrals, satellites):
    mag_gap = np.zeros(len(centrals), float)
    for i, central in enumerate(centrals):
        sats = satellites[np.searchsorted(
            satellites["upid"], central["id"], side="left"):np.searchsorted(
                satellites["upid"], central["id"], side="right")]

        if len(sats) == 0:
            mag_gap[i] = -1
        else:
            mag_gap[i] = central["sm"] + central["icl"] - (sats[-1]["sm"] + sats[-1]["icl"])

    return mag_gap
