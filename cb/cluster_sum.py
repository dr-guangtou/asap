import numpy as np

def centrals_with_satellites(centrals, satellites, n):
    """
    Given an array of centrals and satellites where the satellites are sorted by
    "uparent_ID" and then "mp" (or relevant mass key), adds the masses of the n
    largest satellites (or n% largest satellites if n < 1) to the centrals.
    """
    new_centrals = np.copy(centrals) # no mutation!
    for i, central in enumerate(new_centrals):
        # Remember sats are sorted by "uparent_ID" then "mp"
        sats = satellites[
            np.searchsorted(satellites["uparent_ID"], central["ID"], side="left"):
            np.searchsorted(satellites["uparent_ID"], central["ID"], side="right")
            ]
        sats = sats[:get_n(n, len(sats))]
        # Now add sats to new_centrals
        for col in ['m', 'mp', 'sm', 'icl', 'sfr']:
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
