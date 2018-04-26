import numpy as np

def get_richness(centrals, satellites, min_mass):
    richness = np.zeros(len(centrals), int)
    for i, central in enumerate(centrals):
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

        if central["sm"] + central["icl"] > min_mass:
            richness[i] += 1

    return richness

def get_photoz_richness(centrals, richness):
    z_err = 90 # https://arxiv.org/pdf/1702.01682.pdf
    box_size = 400
    # Slightly weird things "out of the box". We will mod to put them back in
    for i in "xy":
        assert np.max(centrals[i]) < box_size * 1.1 and np.min(centrals[i]) > box_size * -0.1
    assert np.max(centrals["z"]) < box_size and np.min(centrals["z"]) > 0

    # First, we run this for all rich centrals
    # Then, we run it for a subset of all non-rich centrals and scale those
    poor_subsample_size = 20000
    subsample_factor = len(np.nonzero(richness == 1)[0]) / poor_subsample_size
    indexes = [
            np.nonzero(richness > 1)[0], # rich subsample
            np.random.choice(np.nonzero(richness == 1)[0], poor_subsample_size), # poor subsample
    ]
    photoz_richnesses = np.zeros((2, len(centrals)), int)
    for j, idx in enumerate(indexes):
        subset_centrals = centrals[idx]
        subset_richness = richness[idx]

        cached = {}
        cached["x"] = cache_subtract(subset_centrals["x"] % box_size, box_size, 1, 2)
        cached["y"] = cache_subtract(subset_centrals["y"] % box_size, box_size, 1, 2)
        cached["z"] = cache_subtract(subset_centrals["z"], box_size, 10, 1)

        for i, central in enumerate(centrals):
            r_err = central["rvir"]/1000 # r_vir in Mpc
            photoz_richnesses[j][i] = np.sum(subset_richness[
                get_indexes(cached, central, box_size, r_err, z_err)
            ])
    return photoz_richnesses[0] + photoz_richnesses[1]*subsample_factor, photoz_richnesses[0], photoz_richnesses[1]

def get_indexes(cached, central, box_size, r_err, z_err):
    r_dist = get_r_dist(cached, central, box_size)
    indexes = (
            (
                r_dist < r_err
            ) & (
                cached["z"][int(np.floor(central["z"] / 10))] < z_err
            )
    )
    return indexes

def get_r_dist(cached, central, box_size):
    return np.sqrt(
            cached["x"][int(np.floor((central["x"] % box_size) / 1))] +
            cached["y"][int(np.floor((central["y"] % box_size) / 1))])

# cache the squared distance between every item in X and each position
def cache_subtract(inp, box_size, step, power):
    res = []
    for i in range(0, box_size, step):
        res.append(wrap_subtract(inp, i, box_size)**power)
    return res


def wrap_subtract(x1, x2, box_size):
    diff = np.abs(x1 - x2)
    return np.minimum(diff, box_size - diff)
