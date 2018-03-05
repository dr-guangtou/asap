"""
Tools to build and use the chmf fit
"""
import numpy as np
import fits.common as c

# given a catalog, return the hmf
def build_chmf(catalog, key):
    hm = np.log10(catalog[key]["data"]["m"])
    bin_centers, cumulative_count = c.build_density_function(hm)
    return bin_centers, cumulative_count

def hmass_at_density(catalog, key, density):
    return c.x_at_density(catalog, key, "hmf", density)

def density_at_hmass(catalog, key, mass):
    return c.density_at_x(catalog, key, "hmf", mass)
