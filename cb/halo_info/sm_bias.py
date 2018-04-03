"""
Given a catalog return the SM_bias
"""
import numpy as np

import smhm_fit

def get_sm_bias(catalog, include_id=False):
    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    if include_id:
        res = np.zeros(len(delta_stellar_masses), dtype=[("id", np.int), ("sm_bias", np.float64)])
        res["id"] = cdata["id"]
        res["sm_bias"] = delta_stellar_masses
        return res


    return delta_stellar_masses
