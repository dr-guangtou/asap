"""
Tools to build and use the csmf fit
"""
import numpy as np
import fits.common as c

# Import from Song's stuff
import sys
sys.path.append("..")
import lib.stellar_mass_function as smf


# Consider normalising all of these - don't divide by bin width.

# given a list of stellar masses, return the smf
def build_csmf(catalog, key):
    sm = np.log10(catalog[key]["data"]["sm"] + catalog[key]["data"]["icl"])
    bin_centers, y, _ = smf.compute_smf(sm, c._sim_volume, c._number_of_bins, np.min(sm), np.max(sm))
    return bin_centers, c._cumulate(y)

def mass_at_density(catalog, key, density):
    return c._x_at_density(catalog, key, "smf", density)

def density_at_mass(catalog, key, mass):
    return c._density_at_x(catalog, key, "smf", mass)
