import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def linear(x, y):

    xbin = np.arange(np.min(x), np.max(x), (np.max(x) - np.min(x))/15)
    xbin_mid = xbin[:-1] + np.diff(xbin) / 2


    mean_y, _, _ = scipy.stats.binned_statistic(x, y, statistic="mean", bins=xbin)
    std_y, _, _ = scipy.stats.binned_statistic(x, y, statistic="std", bins=xbin)

    _, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(xbin_mid, mean_y, c="orange")
    ax.fill_between(xbin_mid, mean_y-std_y, mean_y+std_y, alpha=0.5, facecolor="tab:orange")

    print(scipy.stats.linregress(x, y))
    print(scipy.stats.spearmanr(x, y))
    return ax
