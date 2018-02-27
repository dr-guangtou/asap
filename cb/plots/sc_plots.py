import numpy as np
import matplotlib.pyplot as plt

def whats_up_with_insitu(data_halo_cut):
    _, ax = plt.subplots()
    cat = data_halo_cut["insitu"]["data"]
    print(len(cat))
    cat = cat[cat["sm"] > 1e7]
    print(len(cat))
    assert np.all(cat["icl"] == 0)
    hm = np.log10(cat["m"])
    sm = np.log10(cat["sm"])

    ax.scatter(hm, sm)
