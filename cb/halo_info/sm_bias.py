"""
Given a catalog return the SM_bias
"""
import numpy as np
import pandas as pd

import data
import smhm_fit

# sm_bias = True_SM - Mean_SM(given Mhalo)
# Note that this needs to be the data halo cut!
def get_sm_bias(catalog, include_id=False, use_cut_data=False):
    if use_cut_data:
        wdata, wfit = "data_cut", "fit_cut"
    else:
        wdata, wfit = "data", "fit"

    cdata = catalog[wdata]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog[wfit])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    if include_id:
        res = np.zeros(len(delta_stellar_masses), dtype=[("id", np.int), ("sm_bias", np.float64)])
        res["id"] = cdata["id"]
        res["sm_bias"] = delta_stellar_masses
        return res

    return delta_stellar_masses

def build_sm_sm_bias_fit(z):
    assert type(z) is pd.core.frame.DataFrame

    cen_stellar_masses = z["sm_cen"] + z["icl_cen"]
    tot_stellar_masses = z["sm_tot"] + z["icl_tot"]

    fit = smhm_fit.get_sm_cen_at_sm_halo_fit(cen_stellar_masses, tot_stellar_masses)

    res = {
            "fit": fit,
            "data": z,
    }
    return res

def get_sm_sm_bias(z, fit):
    cen_stellar_masses = np.log10(z["sm_cen"] + z["icl_cen"])
    tot_stellar_masses = np.log10(z["sm_tot"] + z["icl_tot"])

    predicted_stellar_masses = smhm_fit.f_shmr(tot_stellar_masses, *fit)
    delta_stellar_masses = cen_stellar_masses - predicted_stellar_masses

    return delta_stellar_masses
