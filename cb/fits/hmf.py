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

# catalog here needs to be data_halo_cut["cen"] (or the like)
def hmass_at_density(catalog, density):
    return c.x_at_density(catalog["hmf"][1], catalog["hmf"][0], density)

def density_at_hmass(catalog, mass):
    return c.density_at_x(catalog["hmf"][1], catalog["hmf"][0], mass)
