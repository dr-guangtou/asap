"""
Tools to build and use the csmf fit
"""
import numpy as np
import fits.common as c


# given a list of stellar masses, return the smf
def build_csmf(catalog, key):
    sm = np.log10(catalog[key]["data"]["sm"] + catalog[key]["data"]["icl"])
    bin_centers, cumulative_count = c.build_density_function(sm)
    return bin_centers, cumulative_count

def mass_at_density(catalog, key, density):
    return c.x_at_density(catalog, key, "smf", density)

def density_at_mass(catalog, key, mass):
    return c.density_at_x(catalog, key, "smf", mass)
