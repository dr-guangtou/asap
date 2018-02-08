import numpy as np


def centrals_with_satellites(centrals, satellites, n):
    """
    Given an array of centrals and satellites where the satellites are sorted by
    "upid" and then "mp" (or relevant mass key), adds the masses of the n
    largest satellites (or n% largest satellites if n < 1) to the centrals.
    """
    new_centrals = np.copy(centrals)  # no mutation!
    if n == 0:
        return new_centrals

    for i, central in enumerate(new_centrals):
        # Remember sats are sorted by "upid" then the sum of stellar masses, ascending
        sats = satellites[np.searchsorted(
            satellites["upid"], central["id"], side="left"):np.searchsorted(
                satellites["upid"], central["id"], side="right")]
        sats = sats[::-1][:get_n(n, len(sats))]  # Reversed to get the largest
        # Now add satellite stellar mass/formation to new_centrals
        for col in ['sm', 'icl', 'sfr']:
            new_centrals[i][col] += np.sum(sats[col])
    return new_centrals


def get_n(n, total_sats):
    # Error if less than 0
    if n < 0:
        raise Exception("n must be > 0, got {}".format(n))
    # Return ceiled fraction if is a fraction
    elif n < 1:
        return int(np.ceil(n * total_sats))
    # Return n (at most the total_sats) if it is a number
    return min(n, total_sats)


def get_richness(halos, satellites, min_mass):
    richness = np.zeros(len(halos), int)
    for i, central in enumerate(halos):
        sats = satellites[np.searchsorted(
            satellites["upid"], central["id"], side="left"):np.searchsorted(
                satellites["upid"], central["id"], side="right")]
        sats = sats[::-1]

        for j in range(len(sats)):
            if sats[j]["icl"] + sats[j]["sm"] < min_mass:
                richness[i] = j
                break
        else:
            richness[i] = len(sats)

    return richness
