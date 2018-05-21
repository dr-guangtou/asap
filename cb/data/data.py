import os
import numpy as np

from data import cluster_sum
from halo_info import get_richness, get_mag_gap, get_photoz_richness
import smhm_fit

def load(f=None):
    datadir = os.getenv("dataDir") + "/universe_machine/"
    catalog_file = f or "sfr_catalog_insitu_exsitu_0.712400_final_extended_wpid.npz"

    catalog = np.load(datadir + catalog_file)
    centrals = catalog["centrals"]
    satellites = catalog["satellites"]
    return centrals, satellites

# This should be ordered in an appropriate way
# As I understand it at the moment this probably just means central first.
# (Later) What did I mean by this???
cut_config = {
        "cen": {"n_sats": 0, "mass_limit": 11.2},
        1: {"n_sats": 1, "mass_limit": 11.3},
        2: {"n_sats": 2, "mass_limit": 11.4},
        5: {"n_sats": 5, "mass_limit": 11.4},
        "halo": {"n_sats": 0.999999999, "mass_limit": 11.4},
}

min_mass_for_richness = 0.2*(10**11.34) # 0.2 * M_star

def cuts_with_sats(centrals, satellites):
    sm_res = {}
    hm_res = {}

    # Do the basic data cuts
    for (k, cfg) in cut_config.items():
        sm_res[k] = {}
        hm_res[k] = {}

        # Do things with both a richness cut (assuming we can only find sats > a certain size) and without
        # Note that our richness cut is < than the lowest cen mass so we never have to worry about missing cens
        for x in [
                [False, ""],
                [True, "_cut"], # This doesn't really make sense for centrals
        ]:
            centrals_with_n_sats = cluster_sum.centrals_with_satellites(centrals, satellites, cfg["n_sats"], x[0], min_mass_for_richness)

            # SM cut stuff
            sm_centrals_with_n_sats = centrals_with_n_sats[
                    (centrals_with_n_sats["sm"] + centrals_with_n_sats["icl"]) > 10**cfg["mass_limit"]
            ]
            sm_res[k]["data"+x[1]] = sm_centrals_with_n_sats
            sm_res[k]["fit"+x[1]] = smhm_fit.get_hm_at_fixed_sm_fit(sm_centrals_with_n_sats, restrict_to_power_law=False)

            # HM cut stuff
            hm_centrals_with_n_sats = centrals_with_n_sats[centrals_with_n_sats["m"] > 10**11.5]

            hm_res[k]["data"+x[1]] = hm_centrals_with_n_sats
            hm_res[k]["fit"+x[1]] = smhm_fit.get_sm_at_fixed_hm_fit(hm_centrals_with_n_sats, restrict_to_power_law = k in set(["halo"]))

    add_sm_insitu(sm_res, centrals)
    add_hm_insitu(hm_res, centrals)

    # Now add the extra stuff to the cen. This is stuff that doesn't make sense once you add sats in
    # mag gap doesn't make sense once you add sats in
    sm_res["cen"]["mag_gap"] = get_mag_gap(sm_res["cen"]["data"], satellites)
    hm_res["cen"]["mag_gap"] = get_mag_gap(hm_res["cen"]["data"], satellites)
    return sm_res, hm_res

def add_sm_insitu(res, centrals):
    insitu_only = np.copy(centrals)
    insitu_only["icl"] = 0
    insitu_only = insitu_only[insitu_only["sm"] > 10**11.1]
    res["insitu"] = {
            "data": insitu_only,
            "fit": smhm_fit.get_hm_at_fixed_sm_fit(insitu_only),
            "data_cut": insitu_only,
            "fit_cut": smhm_fit.get_hm_at_fixed_sm_fit(insitu_only),
    }

def add_hm_insitu(res, centrals):
    insitu_only = np.copy(centrals)
    insitu_only = insitu_only[insitu_only["m"] > 10**13]
    insitu_only["icl"] = 0

    # Drop one terrible data point
    insitu_only = insitu_only[np.where(insitu_only["sm"] > 1e7)]

    res["insitu"] = {
            "data": insitu_only,
            "fit": smhm_fit.get_sm_at_fixed_hm_fit(insitu_only),
            "data_cut": insitu_only,
            "fit_cut": smhm_fit.get_sm_at_fixed_hm_fit(insitu_only),
    }

def create_richness_data(centrals, satellites):
    res = np.zeros(
            len(centrals),
            dtype=[
                ("id", np.int), # The halo mass
                ("m", np.float64), # The halo mass
                ("richness", np.int16), # The true richness of each halo (N_gal with M*cen > sth)
                ("photoz_richness", np.int16), # Photoz richness!
            ],
    )
    res["id"] = centrals["id"]
    res["m"] = centrals["m"]
    res["richness"] = get_richness(centrals, satellites, min_mass_for_richness)
    res["photoz_richness"] = get_photoz_richness(centrals, satellites, min_mass_for_richness)
    return res
