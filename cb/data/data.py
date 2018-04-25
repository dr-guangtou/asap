import os
import numpy as np

from data import cluster_sum
from halo_info import get_richness, get_mag_gap

def load(f=None):
    datadir = os.getenv("dataDir") + "/universe_machine/"
    catalog_file = f or "sfr_catalog_insitu_exsitu_0.712400_final_extended.npz"

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

min_mass_for_richness = 10**10.3

def sm_cuts_with_sats(centrals, satellites, f):
    res = {}
    for (k, cfg) in cut_config.items():
        centrals_with_n_sats = cluster_sum.centrals_with_satellites(centrals, satellites, cfg["n_sats"])
        centrals_with_n_sats = centrals_with_n_sats[
                (centrals_with_n_sats["sm"] + centrals_with_n_sats["icl"]) > 10**cfg["mass_limit"]
        ]


        res[k] = {
            "data": centrals_with_n_sats,
            "fit": f(centrals_with_n_sats, restrict_to_power_law=False),
        }

        if k == "cen":
            richness = get_richness(centrals_with_n_sats, satellites, min_mass_for_richness)
            out = np.zeros(len(richness), dtype=[("m", "float64"), ("richness", "float64")])
            out["m"], out["richness"] = centrals_with_n_sats["m"], richness
            res[k]["richness"] = out
            res[k]["mag_gap"] = get_mag_gap(centrals_with_n_sats, satellites)

    # Add insitu
    insitu_only = np.copy(centrals)
    insitu_only["icl"] = 0
    insitu_only = insitu_only[
            insitu_only["sm"] > 10**11.1
    ]
    res["insitu"] = {
            "data": insitu_only,
            "fit": f(insitu_only),
    }

    return res

def hm_cuts_with_sats(centrals, satellites, f):
    res = {}

    for (k, cfg) in cut_config.items():
        centrals_with_n_sats = cluster_sum.centrals_with_satellites(centrals, satellites, cfg["n_sats"])
        centrals_with_n_sats = centrals_with_n_sats[centrals_with_n_sats["m"] > 10**13]

        res[k] = {
            "data": centrals_with_n_sats,
            "fit": f(centrals_with_n_sats, restrict_to_power_law = k in set(["halo"])),
        }
        if k == "cen":
            richness = get_richness(centrals_with_n_sats, satellites, min_mass_for_richness)
            out = np.zeros(len(richness), dtype=[("m", "float64"), ("richness", "float64")])
            out["m"], out["richness"] = centrals_with_n_sats["m"], richness
            res[k]["richness"] = out
            res[k]["mag_gap"] = get_mag_gap(centrals_with_n_sats, satellites)

    # Add insitu
    insitu_only = np.copy(centrals)
    insitu_only["icl"] = 0
    insitu_only = insitu_only[insitu_only["m"] > 10**13]

    # Drop one terrible data point
    insitu_only = insitu_only[np.where(insitu_only["sm"] > 1e7)]

    res["insitu"] = {
            "data": insitu_only,
            "fit": f(insitu_only),
    }

    return res
