import numpy as np
import collections

number_of_bins = 500
# sim_volume = 400**3
def build_density_function(vals):
    smf, bin_edges = np.histogram(vals, bins=number_of_bins)
    bin_centers = bin_edges[:-1] + ((bin_edges[1:] - bin_edges[:-1]) / 2)
    return bin_centers, cumulate(smf)

def cumulate(x):
    cum_x = np.zeros_like(x)
    cum_x[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        cum_x[i] += cum_x[i+1] + x[i]
    return cum_x

def x_at_density(catalog, key, xmf, density):
    if isinstance(density, collections.Iterable):
        return [_x_at_density(catalog, key, xmf, d) for d in density]
    return _x_at_density(catalog, key, xmf, density)

def density_at_x(catalog, key, xmf, mass):
    if isinstance(mass, collections.Iterable):
        return [_density_at_x(catalog, key, xmf, m) for m in mass]
    return _density_at_x(catalog, key, xmf, mass)

def _x_at_density(catalog, key, xmf, density):
    if key:
        catalog = catalog[key]

    densities = catalog[xmf][1] # monotonically decreasing
    for i, d in enumerate(densities):
        if d < density:
            if i == 0:
                raise Exception("Given density is higher than everything in our CSMF")
            index = i
            break
    else:
        raise Exception("Given density is lower than everything in our CSMF")

    return np.mean([catalog[xmf][0][index], catalog[xmf][0][index-1]])

def _density_at_x(catalog, key, xmf, mass):
    if key:
        catalog = catalog[key]

    masses = catalog[xmf][0] # monotonically increasing
    for i, m in enumerate(masses):
        if m > mass:
            if i == 0:
                raise Exception("Given mass is lower than everything in our CSMF")
            index = i
            break
    else:
        raise Exception("Given mass is higher than everything in our CSMF")

    return np.mean([catalog[xmf][1][index], catalog[xmf][1][index-1]])
