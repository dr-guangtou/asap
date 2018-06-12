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

# Nice easy to use funcs
def richness_at_density(catalog, density):
    return c.x_at_density(catalog["rmf"][1], catalog["rmf"][0], density)

def density_at_richness(catalog, richness):
    return c.density_at_x(catalog["rmf"][1], catalog["rmf"][0], richness)


# Old

# def build_photoz_rmf(catalog):
#     r = catalog["photoz_richness"]
#     bin_centers, cumulative_count = c.build_density_function(r)
#     return bin_centers, cumulative_count

# def build_specz_rmf(catalog):
#     r = catalog["specz_richness"]
#     bin_centers, cumulative_count = c.build_density_function(r)
#     return bin_centers, cumulative_count


# # catalog here needs to be the raw richness dict.
# def photoz_richness_at_density(catalog, density):
#     return c.x_at_density(catalog["photoz_rmf"][1], catalog["photoz_rmf"][0], density)

# def density_at_photoz_richness(catalog, richness):
#     return c.density_at_x(catalog["photoz_rmf"][1], catalog["photoz_rmf"][0], richness)

# def specz_richness_at_density(catalog, density):
#     return c.x_at_density(catalog["specz_rmf"][1], catalog["specz_rmf"][0], density)

# def density_at_specz_richness(catalog, richness):
#     return c.density_at_x(catalog["specz_rmf"][1], catalog["specz_rmf"][0], richness)
