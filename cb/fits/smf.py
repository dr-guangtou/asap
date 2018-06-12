"""
Tools to build and use the csmf fit
"""
import numpy as np
import fits.common as c


# given a list of stellar masses, return the smf
def build_csmf(catalog, key, cut=False):
    if cut:
        sm = np.log10(catalog[key]["data_cut"]["sm"] + catalog[key]["data_cut"]["icl"])
    else:
        sm = np.log10(catalog[key]["data"]["sm"] + catalog[key]["data"]["icl"])
    bin_centers, cumulative_count = c.build_density_function(sm)
    return bin_centers, cumulative_count


# catalog here needs to be data_halo_cut["cen"] (or the like)
def mass_at_density(catalog, density, cut=False):
    if cut:
        return c.x_at_density(catalog["smf_cut"][1], catalog["smf_cut"][0], density)
    else:
        return c.x_at_density(catalog["smf"][1], catalog["smf"][0], density)

def density_at_mass(catalog, mass, cut=False):
    if cut:
        return c.density_at_x(catalog["smf_cut"][1], catalog["smf_cut"][0], mass)
    else:
        return c.density_at_x(catalog["smf"][1], catalog["smf"][0], mass)
