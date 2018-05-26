import numpy as np
from halo_info import preprocess_data
from numpy.lib.recfunctions import append_fields
import halotools.mock_observables

box_size = 400

def get_specz_mass(centrals, satellites, min_mass, max_ssfr, n):
    new_centrals = np.copy(centrals)  # no mutation!
    if n == 0:
        return new_centrals

    z_err = 15
    centrals_ht, big_enough_gals_ht, big_enough_gals = preprocess_data(
            centrals, satellites, min_mass, max_ssfr, box_size, True,
    )
    indexes = halotools.mock_observables.idx_in_cylinders(
            centrals_ht,
            big_enough_gals_ht,
            centrals["rvir"]/1000,
            z_err,
            period=box_size,
    )
    assert np.all(centrals_ht[:,0] == centrals["x"])

    masses = big_enough_gals[indexes["i2"]]["sm"] + big_enough_gals[indexes["i2"]]["icl"]
    upid = big_enough_gals[indexes["i2"]]["upid"]
    our_id = big_enough_gals[indexes["i2"]]["id"]

    indexes = append_fields(indexes, ("mass", "upid", "id"), (masses, upid, our_id))
    indexes = np.sort(indexes, order=["i1", "mass"])

    num_doubled = 0
    found = np.ones(len(centrals), dtype=bool)
    # import pdb; pdb.set_trace()
    # Note that we will probably double count the central here - it is in both centrals and big_enough
    for i, central in enumerate(new_centrals):
        central_mass = central["sm"] + central["icl"]

        start_idx = np.searchsorted(indexes["i1"], i, side="left")
        end_idx = np.searchsorted(indexes["i1"], i, side="right")
        to_incl = get_n(n, end_idx - start_idx)

        for idx in range(start_idx, end_idx):
            extra = indexes[idx]
            if to_incl == 0:
                break

            # This is the central again. Ignore it
            if extra["id"] == central["id"]:
                num_doubled += 1
                continue
            # We are polluted with a larger central - we wouldn't find this halo
            if extra["mass"] > central_mass and extra["upid"] == -1:
                found[i] = False
                break

            new_centrals[i]["icl"] += extra["mass"]
            to_incl -= 1

    return new_centrals[found]


# cbx add a ssfr cut here
def centrals_with_satellites(centrals, satellites, n, rich_cut, min_mass):
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
        sats = sats[::-1][:get_n(n, len(sats))] # Reversed to get the largest

        if rich_cut:
            # Make sure that we are being fair - if we d
            sats = sats[
                    sats["sm"] + sats["icl"] > min_mass
            ]
        # Now add satellite stellar mass/formation to new_centrals
        for col in ["sm", "icl", "sfr"]: # I don't think we need the Acc_rate* - those are HM
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
