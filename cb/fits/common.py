import numpy as np
import collections

number_of_bins = 500

# [(X, number of things with Y > X), ()]
# e.g. for stellar mass [(13, 100), (13.2, 50), ... ,(14, 1)]
def build_density_function(vals):
    smf, bin_edges = np.histogram(vals, bins=number_of_bins)
    bin_centers = bin_edges[:-1] + ((bin_edges[1:] - bin_edges[:-1]) / 2)
    return bin_centers, _cumulate(smf)

def _cumulate(x):
    cum_x = np.zeros_like(x)
    cum_x[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        cum_x[i] = cum_x[i+1] + x[i]
    return cum_x

def x_at_density(densities, values, density):
    if isinstance(density, collections.Iterable):
        return np.array([_x_at_density(densities, values, d) for d in density])
    return _x_at_density(densities, values, density)

def density_at_x(densities, values, x):
    if isinstance(x, collections.Iterable):
        return np.array([_density_at_x(densities, values, i) for i in x])
    return _density_at_x(densities, values, x)

def _x_at_density(densities, values, density):
    for i, d in enumerate(densities):
        if d == density:
            return values[i]
        if d < density:
            if i == 0:
                raise Exception("Given density is higher than everything in our CSMF")
            return np.mean([values[i], values[i-1]])

    raise Exception("Given density is lower than everything in our CSMF")

def _density_at_x(densities, values, x):
    for i, m in enumerate(values):
        if m == x:
            return densities[i]
        if m > x:
            if i == 0:
                raise Exception("Given mass is lower than everything in our CSMF")
            return np.mean([densities[i], densities[i-1]])
    raise Exception("Given mass is higher than everything in our CSMF")
