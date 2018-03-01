import numpy as np

_sim_volume = 400**3
_number_of_bins = 500

def _cumulate(x):
    cum_x = np.zeros_like(x)
    cum_x[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        cum_x[i] += cum_x[i+1] + x[i]
    return cum_x

def _x_at_density(catalog, key, xmf, density):
    densities = catalog[key][xmf][1] # monotonically decreasing
    for i, d in enumerate(densities):
        if d < density:
            if i == 0:
                raise Exception("Given density is higher than everything in our CSMF")
            index = i
            break
    else:
        raise Exception("Given density is lower than everything in our CSMF")

    return np.mean([catalog[key][xmf][0][index], catalog[key][xmf][0][index-1]])

def _density_at_x(catalog, key, xmf, mass):
    masses = catalog[key][xmf][0] # monotonically increasing
    for i, m in enumerate(masses):
        if m > mass:
            if i == 0:
                raise Exception("Given mass is lower than everything in our CSMF")
            index = i
            break
    else:
        raise Exception("Given mass is higher than everything in our CSMF")

    return np.mean([catalog[key][xmf][1][index], catalog[key][xmf][1][index-1]])
