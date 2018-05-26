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


# catalog here needs to be data_halo_cut["cen"] (or the like)
def mass_at_density(catalog, density):
    return c.x_at_density(catalog["smf"][1], catalog["smf"][0], density)

def density_at_mass(catalog, mass):
    return c.density_at_x(catalog["smf"][1], catalog["smf"][0], mass)
