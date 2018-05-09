import numpy as np

def get_richness(centrals, satellites, min_mass):
    richness = np.zeros(len(centrals), int)
    for i, central in enumerate(centrals):
        sats = satellites[np.searchsorted(
            satellites["upid"], central["id"], side="left"):np.searchsorted(
                satellites["upid"], central["id"], side="right")]
        sats = sats[::-1]


        if central["sm"] + central["icl"] > min_mass:
            richness[i] += 1

        for j in range(len(sats)):
            if sats[j]["icl"] + sats[j]["sm"] > min_mass:
                richness[i] += 1
            else:
                break

    return richness

def get_photoz_richness(centrals, satellites, min_mass):
    z_err = 52 # My paper
    box_size = 400
    # Slightly weird things "out of the box". We will mod to put them back in
    for i in "xy":
        assert np.max(centrals[i]) < box_size * 1.1 and np.min(centrals[i]) > box_size * -0.1
    assert np.max(centrals["z"]) < box_size and np.min(centrals["z"]) > 0

    big_enough_centrals = np.copy(centrals[centrals["sm"] + centrals["icl"] > min_mass])
    big_enough_sats = np.copy(satellites[satellites["sm"] + satellites["icl"] > min_mass])
    big_enough_gals = np.concatenate((big_enough_centrals, big_enough_sats))

    big_enough_gals["x"] %= box_size
    big_enough_gals["y"] %= box_size
    for i in "xy":
        assert np.max(big_enough_gals[i]) < box_size and np.min(big_enough_gals[i]) > 0

    big_enough_gals = np.sort(big_enough_gals, order=["x"])

    photoz_richness = np.zeros(len(centrals), np.int16)
    for i, central in enumerate(centrals):
        r_err = central["rvir"]/1000 # r_vir in Mpc

        x_cut_gals = x_cut(central, big_enough_gals, box_size, r_err)
        xy_cut_gals = acceptable(central, x_cut_gals, box_size, r_err, "y")
        xyz_cut_gals = acceptable(central, xy_cut_gals, box_size, z_err, "z")

        xyzr_cut_gals = xyz_cut_gals[get_r_dists(central, xyz_cut_gals, box_size) < r_err]
        photoz_richness[i] = len(xyzr_cut_gals)
    return photoz_richness

def x_cut(central, test_galaxies, box_size, r_err):

    upper_index = np.searchsorted(test_galaxies["x"], central["x"] + r_err % box_size)
    lower_index = np.searchsorted(test_galaxies["x"], central["x"] - r_err % box_size)

    if upper_index > lower_index:
        return test_galaxies[lower_index:upper_index]
    else:
        return test_galaxies.take(range(
            upper_index - len(test_galaxies), lower_index
            ), mode="wrap")


def acceptable(central, test_galaxies, box_size, allowed_err, idx):
    # If it is central enough, we don't need to worry about wrapping
    if central[idx] > allowed_err and central[idx] < (box_size - allowed_err):
        return test_galaxies[
                no_wrap_subtract(central[idx], test_galaxies[idx]) < allowed_err
        ]

    return test_galaxies[
            wrap_subtract(central[idx], test_galaxies[idx], box_size) < allowed_err
    ]

# by the time we call this we should have such for things that needing to wrap subtract isn't really an issue
def get_r_dists(central, comp, box_size):
    return np.sqrt(
            wrap_subtract(central["x"], comp["x"], box_size)**2 +
            wrap_subtract(central["y"], comp["y"], box_size)**2)

def no_wrap_subtract(x1, x2):
    return np.abs(x1 - x2)

def wrap_subtract(x1, x2, box_size):
    diff = np.abs(x1 - x2)
    return np.minimum(diff, box_size - diff)
