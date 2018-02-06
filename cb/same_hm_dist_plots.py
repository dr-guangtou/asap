import numpy as np
import matplotlib.pyplot as plt

# Given a sample, randomly selects a sample with the same halo mass distribution
# At the moment this uses "upscaling" to reduce error which I am not sure makes sense...
def get_sample(sample1, catalog):
    upscale_factor = 100
    count1, bin_edges = np.histogram(np.log10(sample1["m"]), bins=50)

    # If this is too large we won't get a good dist.
    # If it is too small I think there also might be issues...
    bin_width = bin_edges[1] - bin_edges[0]
    print("Bin width is {}".format(bin_width))

    bins = np.digitize(catalog["m"], np.power(10, bin_edges), right=True)
    sample2 = []
    for i in range(len(count1)):
        valid_indexes = np.where( bins == i+1 )[0]
        if len(valid_indexes) == 0:
            continue
        sample2.append(
            np.random.choice(catalog[valid_indexes], size=upscale_factor*int(count1[i])) # does this make sense?
        )
    sample2 = np.concatenate(sample2)
    return sample2


def f_concentration(sample):
    return sample["rvir"] / sample["rs"]

def f_age(sample):
    return sample["Halfmass_Scale"]

def f_mm(sample):
    return sample["scale_of_last_MM"]

def do_everything(catalog, f, cuts=(), bulk_set=None, ax=None):
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    if ax is None:
        _, ax = plt.subplots()
    # Step 1
    sample1 = catalog[
        (catalog["sm"] + catalog["icl"] > 10**cuts[0]) &
        (catalog["sm"] + catalog["icl"] < 10**cuts[1])
    ]
    # Step 2
    sample2 = get_sample(sample1, catalog)

    # Step 3 compare something
    _, bin_edges, _ = ax.hist(f(sample1), bins=18, alpha=0.3, density=True, label="SM")
    ax.hist(f(sample2), bins=bin_edges, alpha=0.3, density=True, label="HM")
    ax.legend()
    ax.set(**bulk_set)
    return ax
