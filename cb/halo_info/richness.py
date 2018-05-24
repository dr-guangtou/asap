import numpy as np
import halotools.mock_observables

def get_richness(centrals, satellites, min_mass, max_ssfr):
    richness = np.zeros(len(centrals), int)
    for i, central in enumerate(centrals):
        sats = satellites[np.searchsorted(
            satellites["upid"], central["id"], side="left"):np.searchsorted(
                satellites["upid"], central["id"], side="right")]
        sats = sats[::-1]


        if (central["sm"] + central["icl"] > min_mass) & (central["ssfr"] < max_ssfr):
            richness[i] += 1

        richness[i] += np.count_nonzero(
                (sats["ssfr"] < max_ssfr) &
                (sats["sm"] + sats["icl"] > min_mass)
        )

    return richness

def get_specz_richness(centrals, satellites, min_mass, max_ssfr):
    z_err = 10 # Check this
    box_size = 400

    centrals_ht, big_enough_gals_ht = _preprocess_data(centrals, satellites, min_mass, max_ssfr, box_size)

def get_photoz_richness(centrals, satellites, min_mass, max_ssfr):
    z_err = 52 # My paper
    box_size = 400

    centrals_ht, big_enough_gals_ht = _preprocess_data(centrals, satellites, min_mass, max_ssfr, box_size)
    counts = halotools.mock_observables.counts_in_cylinders(
            centrals_ht,
            big_enough_gals_ht,
            centrals["rvir"]/1000,
            z_err,
            period=box_size,
    )
    return counts

def _preprocess_data(centrals, satellites, min_mass, max_ssfr, box_size, apply_z_space=False):
    # Slightly weird things "out of the box". We will mod to put them back in
    for i in "xy":
        assert np.max(centrals[i]) < box_size * 1.1 and np.min(centrals[i]) > box_size * -0.1
        assert np.max(satellites[i]) < box_size * 1.1 and np.min(satellites[i]) > box_size * -0.1
    assert np.max(centrals["z"]) < box_size and np.min(centrals["z"]) > 0
    assert np.max(satellites["z"]) < box_size and np.min(satellites["z"]) > 0

    big_enough_gals = _get_big_enough_galaxies(centrals, satellites, min_mass, max_ssfr)
    for i in "xy":
        big_enough_gals[i] %= box_size

    if apply_z_space:
        big_enough_gals_ht = halotools.mock_observables.return_xyz_formatted_array(
                big_enough_gals["x"], big_enough_gals["y"], big_enough_gals["z"],
                velocity=big_enough_gals["vz"], velocity_distortion_dimension="z", redshift=0.404
        )
        centrals_ht = halotools.mock_observables.return_xyz_formatted_array(
                centrals["x"], centrals["y"], centrals["z"],
        )
    else:
        big_enough_gals_ht = halotools.mock_observables.return_xyz_formatted_array(
                big_enough_gals["x"], big_enough_gals["y"], big_enough_gals["z"],
        )
        centrals_ht = halotools.mock_observables.return_xyz_formatted_array(
                centrals["x"], centrals["y"], centrals["z"],
        )
    return centrals_ht, big_enough_gals_ht

def _get_big_enough_galaxies(centrals, satellites, min_mass, max_ssfr):
    big_enough_centrals = np.copy(centrals[
        (centrals["sm"] + centrals["icl"] > min_mass) &
        (centrals["ssfr"] < max_ssfr)
    ])
    big_enough_sats = np.copy(satellites[
        (satellites["sm"] + satellites["icl"] > min_mass) &
        (satellites["ssfr"] < max_ssfr)
    ])
    return np.concatenate((big_enough_centrals, big_enough_sats))
