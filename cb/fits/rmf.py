"""
Tools to build and use the richness mass function fit
"""
import fits.common as c

# The xmf functions are shifted up and down from each other depending on their width
# If you want to compare, you need to normalize them to the same width.
def build_rmf(catalog):
    r = catalog["richness"]
    bin_centers, cumulative_count = c.build_density_function(r)
    return bin_centers, cumulative_count

def richness_at_density(catalog, key, density):
    return c.x_at_density(catalog, key, "rmf", density)

def density_at_richness(catalog, key, richness):
    return c.density_at_x(catalog, key, "rmf", richness)


def build_photoz_rmf(catalog):
    r = catalog["photoz_richness"]
    bin_centers, cumulative_count = c.build_density_function(r)
    return bin_centers, cumulative_count

def photoz_richness_at_density(catalog, key, density):
    return c.x_at_density(catalog, key, "photoz_rmf", density)

def density_at_photoz_richness(catalog, key, richness):
    return c.density_at_x(catalog, key, "photoz_rmf", richness)
