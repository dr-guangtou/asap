"""
Tools to build and use the richness mass function fit
"""
import numpy as np
import fits.common as c

# Import from Song's stuff
import sys
sys.path.append("..")
import lib.stellar_mass_function as smf

# The xmf functions are shifted up and down from each other depending on their width
# If you want to compare, you need to normalize them to the same width.
def build_rmf(catalog):
    r = catalog["richness"]
    bin_centers, y, _ = smf.compute_smf(r, c._sim_volume, c._number_of_bins, np.min(r), np.max(r))
    return bin_centers, c._cumulate(y)

def richness_at_density(catalog, key, density):
    return c._x_at_density(catalog, key, "rmf", density)

def density_at_richness(catalog, key, mass):
    return c._density_at_x(catalog, key, "rmf", mass)
