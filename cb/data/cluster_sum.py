import numpy as np
from halo_info import preprocess_data
from numpy.lib.recfunctions import append_fields
import halotools.mock_observables


# pre pyxing: 2'7"
# post pyxing: 2'7"

box_size = 400

# Cut down to big enough/non star forming gals.
# Add effects from RSD
def cut_and_rsd(centrals, satellites, min_mass, max_ssfr):
    centrals_ht, big_enough_gals_ht, big_enough_gals = preprocess_data(
            centrals, satellites, min_mass, max_ssfr, box_size,
    )
    map_be_to_cen = _map_big_enough_index_to_central(big_enough_gals, centrals)
    return centrals_ht, big_enough_gals_ht, big_enough_gals, map_be_to_cen

def add_uncertainty_to_sats(big_enough_gals_ht, big_enough_gals, photoz_error):
    is_sat = (big_enough_gals["upid"] != -1).astype(int)
    big_enough_gals_ht[:,2] += np.random.normal(
            loc=0, scale=photoz_error, size=len(is_sat)) * is_sat

    big_enough_gals_ht[:,2] %= box_size
    return big_enough_gals_ht


def get_cylinder_mass_and_richness2(
        centrals_ht, big_enough_gals_ht, centrals, big_enough_gals, map_be_to_cen, n, cylinder_depth
):
    new_centrals = np.copy(centrals)  # no mutation!

    indexes = halotools.mock_observables.idx_in_cylinders(
            centrals_ht,
            big_enough_gals_ht,
            centrals["rvir"]/1000,
            cylinder_depth,
            period=box_size,
    )
    assert np.all(centrals_ht[:,0] == centrals["x"])

    masses = big_enough_gals[indexes["i2"]]["sm"] + big_enough_gals[indexes["i2"]]["icl"]
    indexes = append_fields(indexes, "mass", masses)
    # Mass will be increasing. But we go through everything (atm) so this isn't a bit deal atm
    indexes = np.sort(indexes, order=["i1", "mass"])

    # True if the galaxy is not covered by a larger central
    found = np.ones(len(centrals), dtype=bool)
    richness = np.zeros(len(centrals), dtype=np.int)

    # For each central, add in the mass of the N largest sats within in the cylinder
    for i in range(len(new_centrals)):
        central_mass = new_centrals[i]["sm"] + new_centrals[i]["icl"]

        start_idx = np.searchsorted(indexes["i1"], i, side="left")
        end_idx = np.searchsorted(indexes["i1"], i, side="right")
        to_incl = get_n(n, end_idx - start_idx)

        # Find the n largest sats in the cylinder
        # Ignore this central if/when we refind it
        # If we find a central:
        #   Either - say we can't find this galaxy (that central was higher mass)
        #   Or - say we can't find that central (this central is higher mass)
        for idx in range(end_idx-1, start_idx-1, -1): # because we want to look at the highest mass things first
            assert indexes[idx]["i1"] == i
            be_idx = indexes[idx]["i2"]
            assert indexes[idx]["mass"] == big_enough_gals[be_idx]["sm"] + big_enough_gals[be_idx]["icl"]

            # This is the central again. Ignore it
            if new_centrals[i]["id"] == big_enough_gals[be_idx]["id"]:
                richness[i] += 1
                continue

            # We are polluted with a central
            if big_enough_gals[be_idx]["upid"] == -1:
                # If it is larger we wouldn't find this central
                if indexes[idx]["mass"] > central_mass:
                    found[i] = False
                # If we are larger we wouldn't find that central
                else:
                    # Need to map collision["i2"] (which is an index in big_enough_gals) to the index in
                    # the centrals. As we already know that it is a central, this should never keyerror
                    found[map_be_to_cen[be_idx]] = False

            if to_incl > 0:
                new_centrals[i]["icl"] += indexes[idx]["mass"]
                richness[i] += 1
                to_incl -= 1

    return new_centrals[found], richness[found]

def get_cylinder_mass_and_richness(centrals, satellites, min_mass, max_ssfr, n, cylinder_depth):
    centrals_ht, big_enough_gals_ht, big_enough_gals = preprocess_data(
            centrals, satellites, min_mass, max_ssfr, box_size,
    )
    map_be_to_cen = _map_big_enough_index_to_central(big_enough_gals, centrals)

    return get_cylinder_mass_and_richness2(
            centrals_ht, big_enough_gals_ht, centrals, big_enough_gals, map_be_to_cen, n, cylinder_depth)


def centrals_with_satellites(centrals, satellites, n, rich_cut, min_mass, max_ssfr):
    """
    Given an array of centrals and satellites where the satellites are sorted by
    "upid" and then "mp" (or relevant mass key), adds the masses of the n
    largest satellites (or n% largest satellites if n < 1) to the centrals.
    """
    assert np.all(
            (satellites[1:]["upid"] > satellites[:-1]["upid"]) | # Either the upid has increased
            (
                (satellites[1:]["upid"] == satellites[:-1]["upid"]) & # Or it has stayed the same and the mass has decreased
                (satellites[1:]["sm"] + satellites[1:]["icl"] >= satellites[:-1]["sm"] + satellites[:-1]["icl"])
            )
    )

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
                    (sats["sm"] + sats["icl"] > min_mass) &
                    (sats["ssfr"] < max_ssfr)
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

def _map_big_enough_index_to_central(big_enough_gals, centrals):
    centrals_id_to_index = dict(zip(centrals["id"], np.arange(len(centrals))))

    big_enough_index_to_central = {}
    for i, gal in enumerate(big_enough_gals):
        try:
            big_enough_index_to_central[i] = centrals_id_to_index[gal["id"]]
        except KeyError:
            # This can KeyError when the big_enough_gal is a sat. Centrals_id_to_index won't contain that gal["id"]
            continue
    return big_enough_index_to_central
