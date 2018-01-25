import os
import numpy as np
import cluster_sum

def load():
    datadir = os.getenv("dataDir") + "/universe_machine/"
    catalog_file = "sfr_catalog_insitu_exsitu_0.712400_final_extended.npz"

    catalog = np.load(datadir + catalog_file)
    centrals = catalog["centrals"]
    satellites = catalog["satellites"]
    return centrals, satellites

def cut_centrals(centrals):
    centrals_stellar_cut = centrals[centrals["sm"] + centrals["icl"] > 1e11]
    centrals_halo_cut = centrals[centrals["m"] > 1e13]
    return centrals_stellar_cut, centrals_halo_cut

def get_with_sats(centrals, satellites, f):
    res = {}
    for (k, sats) in [(0, 0), (1, 1), (2, 2), (5, 5), (9, 9), ("all", 0.999999)]:
        centrals_with_n_sats = cluster_sum.centrals_with_satellites(centrals, satellites, sats)
        res[k] = {
            "data": centrals_with_n_sats,
            "fit": f(centrals_with_n_sats),
        }
    return res
