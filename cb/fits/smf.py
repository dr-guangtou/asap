"""
Tools to build and use the csmf fit
"""
import numpy as np

# Import from Song's stuff
import sys
sys.path.append("..")
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


def mass_at_density(catalog, key, density):
    densities = catalog[key]["smf"][1] # monotonically decreasing
    for i, d in enumerate(densities):
        if d < density:
            if i == 0:
                raise Exception("Given density is higher than everything in our CSMF")
            index = i
            break
    else:
        raise Exception("Given density is lower than everything in our CSMF")

    return np.mean([catalog[key]["smf"][0][index], catalog[key]["smf"][0][index-1]])

def density_at_mass(catalog, key, mass):
    masses = catalog[key]["smf"][0] # monotonically increasing
    for i, m in enumerate(masses):
        if m > mass:
            if i == 0:
                raise Exception("Given mass is lower than everything in our CSMF")
            index = i
            break
    else:
        raise Exception("Given mass is higher than everything in our CSMF")

    return np.mean([catalog[key]["smf"][1][index], catalog[key]["smf"][1][index-1]])
