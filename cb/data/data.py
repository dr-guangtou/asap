import os
import numpy as np

from data import cluster_sum
from halo_info import get_richness, get_mag_gap
import smhm_fit

def load(catalog_file):
    datadir = os.getenv("dataDir") + "/universe_machine/"

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
        "tot": {"n_sats": 0.999999999, "mass_limit": 11.4},
}
subset_cut_config = {
        "cen": {"n_sats": 0, "mass_limit": 11.2},
        2: {"n_sats": 2, "mass_limit": 11.4},
        "tot": {"n_sats": 0.999999999, "mass_limit": 11.4},
}

min_mass_for_richness = 0.2*(10**11.34) # 0.2 * M_star
max_ssfr = 1e-11

def explore_cylinder_length(centrals, satellites, photoz_error, n):
    mass_cut = 10**11.4

    centrals_ht, big_enough_gals_ht, big_enough_gals, map_be_to_cen = cluster_sum.cut_and_rsd(
            centrals, satellites, min_mass_for_richness, max_ssfr)
    big_enough_gals_ht = cluster_sum.add_uncertainty_to_sats(
            big_enough_gals_ht, big_enough_gals, photoz_error)


    sm_res = {}
    for cylinder_depth in [1]:#, 3, 5, 8, 11, 15, 20, 25, 30, 43]:
        sm_res[cylinder_depth] = {}
        centrals_obs, _ = cluster_sum.get_cylinder_mass_and_richness2(
                centrals_ht, big_enough_gals_ht, centrals, big_enough_gals, map_be_to_cen, n, cylinder_depth)

        # SM cut stuff
        sm_centrals_obs = centrals_obs[
                (centrals_obs["sm"] + centrals_obs["icl"]) > mass_cut
        ]
        sm_res[cylinder_depth]["data_cut"] = sm_centrals_obs
        try:
            sm_res[cylinder_depth]["fit_cut"] = smhm_fit.get_hm_at_fixed_sm_fit(sm_centrals_obs, restrict_to_power_law=False)
        except RuntimeError:
            print("Failure for sm,", cylinder_depth)
    return sm_res


def cylinder_sm_and_richness(centrals, satellites, cylinder_depth, photoz_error):
    centrals_ht, big_enough_gals_ht, big_enough_gals, map_be_to_cen = cluster_sum.cut_and_rsd(
            centrals, satellites, min_mass_for_richness, max_ssfr)
    big_enough_gals_ht = cluster_sum.add_uncertainty_to_sats(
            big_enough_gals_ht, big_enough_gals, photoz_error)

    sm_res, hm_res = {}, {}
    richnesses, richness = [], None
    for (k, cfg) in subset_cut_config.items():
        sm_res[k], hm_res[k] = {}, {}
        centrals_obs, counts = cluster_sum.get_cylinder_mass_and_richness2(
                centrals_ht, big_enough_gals_ht, centrals, big_enough_gals, map_be_to_cen, cfg["n_sats"], cylinder_depth)
        # centrals_obs, counts = cluster_sum.get_cylinder_mass_and_richness(
        #         centrals, satellites, min_mass_for_richness, max_ssfr, cfg["n_sats"], cylinder_depth,
        # )
        richnesses.append(counts)

        if k == "tot":
            richness = _build_richness(centrals_obs, counts)

        # SM cut stuff
        sm_centrals_obs = centrals_obs[
                (centrals_obs["sm"] + centrals_obs["icl"]) > 10**cfg["mass_limit"]
        ]
        sm_res[k]["data_cut"] = sm_centrals_obs
        try:
            sm_res[k]["fit_cut"] = smhm_fit.get_hm_at_fixed_sm_fit(sm_centrals_obs, restrict_to_power_law=False)
        except RuntimeError:
            print("Failure for sm,", k)

        # HM cut stuff
        hm_centrals_obs = centrals_obs[centrals_obs["m"] > 10**11.5]

        hm_res[k]["data_cut"] = hm_centrals_obs
        try:
            hm_res[k]["fit_cut"] = smhm_fit.get_sm_at_fixed_hm_fit(hm_centrals_obs, restrict_to_power_law = k in set(["tot"]))
        except RuntimeError:
            print("Failure for hm,", k)

    return sm_res, hm_res, richnesses, richness

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
            centrals_with_n_sats = cluster_sum.centrals_with_satellites(centrals, satellites, cfg["n_sats"], x[0], min_mass_for_richness, max_ssfr)

            # SM cut stuff
            sm_centrals_with_n_sats = centrals_with_n_sats[
                    (centrals_with_n_sats["sm"] + centrals_with_n_sats["icl"]) > 10**cfg["mass_limit"]
            ]
            sm_res[k]["data"+x[1]] = sm_centrals_with_n_sats
            sm_res[k]["fit"+x[1]] = smhm_fit.get_hm_at_fixed_sm_fit(sm_centrals_with_n_sats, restrict_to_power_law=False)

            # HM cut stuff
            hm_centrals_with_n_sats = centrals_with_n_sats[centrals_with_n_sats["m"] > 10**11.5]

            hm_res[k]["data"+x[1]] = hm_centrals_with_n_sats
            hm_res[k]["fit"+x[1]] = smhm_fit.get_sm_at_fixed_hm_fit(hm_centrals_with_n_sats, restrict_to_power_law = k in set(["tot"]))

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
    res["in"] = {
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

    res["in"] = {
            "data": insitu_only,
            "fit": smhm_fit.get_sm_at_fixed_hm_fit(insitu_only),
            "data_cut": insitu_only,
            "fit_cut": smhm_fit.get_sm_at_fixed_hm_fit(insitu_only),
    }

def create_richness_data(centrals, satellites):
    richness = get_richness(centrals, satellites, min_mass_for_richness, max_ssfr)
    return _build_richness(centrals, richness)

def _build_richness(centrals, richness):
    res = np.zeros(
            len(centrals),
            dtype=[
                ("id", np.int), # The halo mass
                ("m", np.float64), # The halo mass
                ("richness", np.int16), # The true richness of each halo (N_gal with M*cen > sth)
            ],
    )
    res["id"] = centrals["id"]
    res["m"] = centrals["m"]
    res["richness"] = richness
    return res
