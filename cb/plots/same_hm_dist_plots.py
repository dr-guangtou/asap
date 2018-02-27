import numpy as np
import matplotlib.pyplot as plt

def f_concentration(sample):
    return sample["rvir"] / sample["rs"]

def f_age(sample):
    return sample["Halfmass_Scale"]

def f_mm(sample):
    return sample["scale_of_last_MM"]

def do_everything(catalog, key, f, cuts=(), bins=None, ax=None):
    catalog = catalog[key]["data"]
    assert (len(cuts) == 2) and (cuts[1] > cuts[0])
    if ax is None:
        _, ax = plt.subplots()
    # Make the stellar mass cut
    sample1 = catalog[
        (catalog["sm"] + catalog["icl"] > 10**cuts[0]) &
        (catalog["sm"] + catalog["icl"] < 10**cuts[1])
    ]
    print(len(sample1))
    label1 = r"$M_{\ast}^{" + key + r"}$"
    '''
    # sanity plot HM dist
    _, ax1 = plt.subplots()
    ax1.hist(np.log10(sample1["m"]), bins=(bins or 18), density=True)
    '''
    # Get the sample with matching halo mass distribution
    sample2 = _get_samples(sample1, catalog, 100)
    label2 = "Random"
    # Plot
    _, bin_edges, _ = ax.hist(
            f(sample1), bins=(bins or 18), alpha=0.3, density=True, label=label1, color='b')
    bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])

    s2_hists = np.array([np.histogram(f(i),bins=bin_edges, density=True)[0] for i in sample2])
    ax.errorbar(bin_mid, np.mean(s2_hists, axis=0), yerr = np.std(s2_hists, axis=0), label=label2)
    ax.legend()
    return ax

# Given a sample, randomly selects a sample with the same halo mass distribution
# At the moment this uses "upscaling" to reduce error which I am not sure makes sense...
def _get_samples(sample1, catalog, n):
    # Put our original sample in 50 bins and count the number in each. We will match this
    count1, bin_edges = np.histogram(np.log10(sample1["m"]), bins=50)
    print("Bin width is {}".format(bin_edges[1] - bin_edges[0]))

    bins = np.digitize(catalog["m"], np.power(10, bin_edges), right=True)
    sample2 = []
    for _ in range(n):
        s = []
        for i in range(len(count1)):
            valid_indexes = np.where( bins == i+1 )[0]
            if count1[i] == 0:
                continue
            if len(valid_indexes) == 0:
                raise Exception("You need some indexes...")
            s.append(
                np.random.choice(catalog[valid_indexes], size=count1[i])
            )
        sample2.append(np.concatenate(s))
    return np.array(sample2)
