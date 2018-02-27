"""
Tools to build and use the csmf fit
"""
import numpy as np

# Import from Song's stuff
import sys
sys.path.append("../..")
import lib.stellar_mass_function as smf

def _cumulate(x):
    cum_x = np.zeros_like(x)
    cum_x[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        cum_x[i] += cum_x[i+1] + x[i]
    return cum_x

sim_volume = 400**3
# given a list of stellar masses, return the smf
def build_csmf(catalog, key):
    sm = np.log10(catalog[key]["data"]["sm"] + catalog[key]["data"]["icl"])
    bin_centers, y, _ = smf.compute_smf(sm, sim_volume, 500, np.min(sm), np.max(sm))
    return bin_centers, _cumulate(y)


# TODO - write some helpers that find mass at a given CSMF and vice versa
