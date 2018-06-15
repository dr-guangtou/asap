import numpy as np
import scipy.stats

# See https://arxiv.org/pdf/0810.1885.pdf
def resample_scatter(x, y, bins):
    bin_indexes = np.digitize(x, bins)
    # print(np.histogram(bin_indexes))
    stds, stdstds = np.zeros(len(bins)-1), np.zeros(len(bins)-1)

    cnts = []
    for i in range(len(bins) - 1):
        # digitize is 1 indexed
        indexes_in_bin = np.where(bin_indexes == i + 1)[0]
        count_in_bin = len(indexes_in_bin)
        cnts.append(count_in_bin)
        if count_in_bin < 5:
            print("Warning - {} items in bin {}".format(count_in_bin, i+1))

        # Calculate stats for that bin
        iterations = 1000
        this_bin_std = np.zeros(iterations)
        for j in range(iterations):
            ci = np.random.choice(indexes_in_bin, len(indexes_in_bin)) # chosen indexes
            this_bin_std[j] = np.std(y[ci], ddof=1)
        stds[i] = np.mean(this_bin_std)
        stdstds[i] = np.std(this_bin_std, ddof=1)
    print(cnts)
    return stds, stdstds

# This is simlar to ^ except it resamples everything which doesn't guarantee that
# the number of points in each bin is conserved. It *appears* to be the same.
# Trade off here is simple code, but the chance of having empty bins which I am not
# 100% sure how to deal with...
def resample_scatter_simple(x, y, bins):
    stds = []
    while len(stds) < 1000:
        si = np.random.choice(len(x), len(x))
        std, _, _ = scipy.stats.binned_statistic(x[si], y[si], statistic="std", bins=bins)
        if np.any(std == 0):
            print("warning, empty bin. Not an issue unless you see a lot (10?) of these")
            continue
        stds.append(std)
    stds = np.array(stds)
    return np.mean(stds, axis=0), np.std(stds, axis=0, ddof=1)

def _bins_mid(bins):
    return bins[:-1] + (bins[1:] - bins[:-1]) / 2
