"""
These plot secondary parameters on the SHMR plot.
Should show whether these secondary parameters are the cause (or are correlated
with the cause) of the scatter)
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import linear_model
from sklearn import preprocessing

import smhm_fit
from plots import labels as l
from stats import partial_corr

l_delta_sm = r"$\Delta SM$"

def _pretty_coef(coef, labels, xlabels):
    fig, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    lim = max(np.nanmax(coef), np.abs(np.nanmin(coef)))
    img = ax.imshow(coef, cmap="coolwarm", vmin=-lim, vmax=lim)
    # img = ax.imshow(np.expand_dims(coef, 1), cmap="coolwarm", vmin=-lim, vmax=lim)
    lim = np.floor(100 * lim) / 100
    fig.colorbar(img, ticks=[-lim, 0, lim], label="Coefficient value")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize="xx-small")
    ax.set_xlabel("log(Alpha)")
    ax.set_xticks(range(0, len(coef[0]), 10))
    xtick_labels = np.log10(xlabels[[int(j) for j in ax.get_xticks()]])
    ax.set_xticklabels(["{:2f}".format(i) for i in xtick_labels], fontsize="xx-small")

def _pretty_corr(corr, labels):
    # print(corr)
    fig, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    lim = max(np.nanmax(corr), np.abs(np.nanmin(corr)))
    img = ax.imshow(corr, cmap="coolwarm", vmin=-lim, vmax=lim)
    lim = np.floor(10 * lim) / 10
    fig.colorbar(img, ticks=[-lim, 0, lim])

    # I'm not sure why but the first and last ticks are beyond our data?
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize="xx-small", rotation=45)
    ax.set_yticklabels(labels, fontsize="xx-small")

def _build_partial_corr(matrix):
    p_corr = partial_corr(matrix.T)
    p_corr[np.triu_indices(len(p_corr), 0)] = np.nan
    return p_corr

def _build_corr(matrix):
    corr = np.corrcoef(matrix)
    corr[np.triu_indices(len(corr), 0)] = np.nan
    return corr

def _get_data_for_correlation_matrix(catalog):
    cdata = catalog["data"][catalog["data"]["m"] > 1e13]

    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    concentrations = cdata["rvir"] / cdata["rs"]
    mm = cdata["scale_of_last_MM"]
    ages = cdata["Halfmass_Scale"]
    insitu_fraction= cdata["sm"] / (cdata["icl"] + cdata["sm"])

    # These require some mutation so we copy first
    # richness = np.copy(catalog["richness"]["richness"]) # mutations!
    # richness[richness == 0] = np.exp(-0.5) # to fix log issues
    # richness = np.log10(richness)
    acc_rate_m_peak = np.copy(cdata["Acc_Rate_Mpeak"])
    acc_rate_m_peak[acc_rate_m_peak == 0] = np.min(acc_rate_m_peak[np.nonzero(acc_rate_m_peak)])
    acc_rate_m_peak = np.log10(acc_rate_m_peak)


    data = {
            l_delta_sm: delta_stellar_masses,
            # "Halo mass": halo_masses, Because I don't like it?
            "In-situ fraction": insitu_fraction,
            "Concentration": concentrations,
            # "Acc Rate at Mpeak": acc_rate_m_peak,
            # "Richness": richness,
            "Last MM Scale": mm,
            "Halfmass Scale": ages,
    }
    # indexes = (stellar_masses > 11.6) & (stellar_masses < 11.7)
    # indexes = (stellar_masses > 12.1) & (stellar_masses < 12.4)
    # for key in data:
        # data[key] = data[key][indexes]

    labels = list(data.keys())
    matrix = np.vstack(data.values())
    return data, labels, matrix

def cen_vs_halo_correlation_matrix(cen_catalog, halo_catalog):
    _, cen_labels, cen_matrix = _get_data_for_correlation_matrix(cen_catalog)
    _, _, halo_matrix = _get_data_for_correlation_matrix(halo_catalog)

    cen_corr = _build_corr(cen_matrix)
    halo_corr = _build_corr(halo_matrix)

    _pretty_corr(cen_corr - halo_corr, cen_labels)

def marginalized_heatmap(catalog):
    data, _, matrix = _get_data_for_correlation_matrix(catalog)
    corr = _build_corr(matrix)

    fig, axes = plt.subplots(len(data), len(data))
    fig.set_size_inches(36.5, 20.5)

    d_items = list(data.items())
    for i in range(len(d_items)):
        for j in range(i):
            _, _, _, _ = axes[i][j].hist2d(d_items[i][1], d_items[j][1], bins=20, cmap="OrRd", norm=mpl.colors.LogNorm())
            axes[i][j].annotate(s="{:.2f}".format(corr[i][j]), xy=(0.85,1.02), xycoords="axes fraction")
            if i == len(d_items) - 1:
                axes[i][j].set_xlabel(d_items[j][0])
            if j == 0:
                axes[i][j].set_ylabel(d_items[i][0])
        for j in range(i, len(d_items)):
            axes[i][j].axis("off")

    plt.tight_layout()

def partial_correlation_matrix(catalog):
    _, labels, matrix = _get_data_for_correlation_matrix(catalog)
    p_corr = _build_partial_corr(matrix)
    _pretty_corr(p_corr, labels)


def correlation_matrix(catalog):
    _, labels, matrix = _get_data_for_correlation_matrix(catalog)
    corr = _build_corr(matrix)
    _pretty_corr(corr, labels)

def lasso(catalog, skip_plot=False):
    _, labels, matrix = _get_data_for_correlation_matrix(catalog)
    # Make this a (samples, features) matrix with each feature scaled to 0-1
    # Could also scale with StandardScalar (but it shouldn't make any/much difference)
    x = preprocessing.MinMaxScaler().fit_transform(matrix[1:].T)
    y = matrix[0]

    # Range over various alphas to see which components are most important
    coefs = []
    alphas = np.logspace(-1, -5)
    ignore = []
    for alpha in alphas:
        clf = linear_model.Lasso(alpha=alpha)
        shuf_x = np.copy(x)
        for i in ignore:
            np.random.shuffle(shuf_x[:,i])

        clf.fit(shuf_x, y)
        coefs.append(clf.coef_)

    if not skip_plot:
        _pretty_coef(np.array(coefs).T, np.array(labels[1:]), alphas)

    # What I think is the best
    alpha = 10**-3.5
    print("Using log10(alpha) of {}".format(np.log10(alpha)))
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(x, y)
    return clf.coef_, clf.intercept_

def lassoCV(catalog):
    _, labels, matrix = _get_data_for_correlation_matrix(catalog)
    # Make this a (samples, features) matrix with each feature scaled to 0-1
    x = preprocessing.MinMaxScaler().fit_transform(matrix[1:].T)
    # x = preprocessing.StandardScaler().fit_transform(matrix[1:].T)
    y = matrix[0]

    # CV to find alpha. With one thing randomly shuffled
    _, ax = plt.subplots()
    clf = linear_model.LassoCV(cv=12, alphas=np.logspace(-1, -7))
    clf.fit(x, y)
    ax.plot(np.log10(clf.alphas_), np.mean(clf.mse_path_, axis=1), label="All")
    for feature in range(len(x[0])):
        # shuf_x = np.copy(x)
        # np.random.shuffle(shuf_x[:,feature])

        clf = linear_model.LassoCV(cv=12, alphas=np.logspace(-1, -7))
        clf.fit(x[:,[i != feature for i in range(len(x[0]))]] , y)
        # # each row of clf.mse_path is an alpha, each col is for one of the cvs
        # ax.plot(np.log10(clf.alphas_), clf.mse_path_, ':')
        ax.plot(np.log10(clf.alphas_), np.mean(clf.mse_path_, axis=1), ls=":", label=labels[feature+1])
        # ax.plot(np.log10(clf.alphas_), np.mean(clf.mse_path_, axis=1) + np.std(clf.mse_path_, axis=1), label="Mean + 1sd")
        # ax.axvline(np.log10(clf.alpha_), label="Suggested Alpha", color="r")
        # ax.axhline((np.mean(clf.mse_path_, axis=1) + np.std(clf.mse_path_, axis=1))[-1])
        # ax.set(xlabel="Log alpha", ylabel="MSE")
    ax.legend()

def margin_model(catalog):
    _, _, matrix = _get_data_for_correlation_matrix(catalog)

    x = preprocessing.MinMaxScaler().fit_transform(matrix[1:].T)
    y = matrix[0]
    weights, intercept = lasso(catalog, skip_plot=True)

    model = np.dot(x, weights) + intercept
    model_error = y - model

    print(np.std([y, model_error, model], axis=1))
    _, ax = plt.subplots()
    _, bins, _ = ax.hist(model_error, bins=50, alpha=0.3, label="model error")
    ax.hist(y, bins=bins, alpha=0.3, label="SM bias")
    ax.hist(model, bins=bins, alpha=0.3, label="Model")
    ax.legend()

def best_model(catalog):
    data, _, _ = _get_data_for_correlation_matrix(catalog)

    conc = data["Concentration"]
    ages = data["Halfmass Scale"]
    sm_bias = data[l_delta_sm]

    def imshow(ax, binned_stats, **kwargs):
        return ax.imshow(
                binned_stats.statistic.T,
                origin="lower",
                extent=[binned_stats.x_edge[0], binned_stats.x_edge[-1], binned_stats.y_edge[0], binned_stats.y_edge[-1]],
                aspect="auto",
                **kwargs,
        )

    fig = plt.figure()
    big_ax = fig.add_subplot(111, frame_on=False)
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.grid(False)
    ax = [fig.add_subplot(231 + i) for i in range(6)]

    big_ax.set(xlabel="Concentration", ylabel="Halfmass Scale")
    plt.tight_layout()

    # Plot the counts
    count_stats = scipy.stats.binned_statistic_2d(conc, ages, sm_bias, statistic="count", bins=13)
    img = imshow(ax[0], count_stats, norm=mpl.colors.LogNorm(), cmap="OrRd")
    ax[0].set_title("N in bin")
    fig.colorbar(img, ax=ax[0])

    # Plots the means
    means = scipy.stats.binned_statistic_2d(conc, ages, sm_bias, statistic="mean", bins=13)
    means.statistic.T[count_stats.statistic.T < 8] = np.nan
    img = imshow(ax[1], means, vmin=-0.4, vmax=0.4, cmap="coolwarm")
    ax[1].set_title("SM bias")
    fig.colorbar(img, ax=ax[1])

    # Plots the SD in each of these bins
    std= scipy.stats.binned_statistic_2d(conc, ages, sm_bias, statistic="std", bins=13)
    std.statistic.T[count_stats.statistic.T < 8] = np.nan
    img = imshow(ax[2], std, vmin=0, vmax=0.5, cmap="OrRd")
    ax[2].set_title("SD in each bin")
    fig.colorbar(img, ax=ax[2])
    print("""
    This is the issue - even though when binned you see a nice trend,
    there is a massive amount of variance in each bin still. Increasing the number of
    features we build out linear model on doesn't appear to significantly improve this.
    Or maybe I haven't yet tried the right features.
    """)

    # Plot our model
    conc_w, ages_w, intercept_w = 0.1418, -0.3994, 0.1718 # on [0,1]
    # x is conc
    x_midpoints = means.x_edge[:-1] + np.diff(means.x_edge)/2
    y_midpoints = means.y_edge[:-1] + np.diff(means.y_edge)/2

    x_midpoints = preprocessing.MinMaxScaler().fit(
            np.expand_dims(conc, 1)).transform(
            np.expand_dims(x_midpoints, 1)).flatten()
    y_midpoints = preprocessing.MinMaxScaler().fit(
            np.expand_dims(ages, 1)).transform(
            np.expand_dims(y_midpoints, 1)).flatten()
    model = (
        x_midpoints * conc_w +
        np.expand_dims(y_midpoints, 1) * ages_w +
        intercept_w)
    img = ax[3].imshow(model, origin="lower",
            extent=[means.x_edge[0], means.x_edge[-1], means.y_edge[0], means.y_edge[-1]],
            aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
    ax[3].set_title("Linear Model")
    fig.colorbar(img, ax=ax[3])

    # Plot how wrong we are
    sigmas = (model - means.statistic.T)
    lim = np.nanmax(np.abs(sigmas))
    img = ax[4].imshow(sigmas, origin="lower",
            extent=[means.x_edge[0], means.x_edge[-1], means.y_edge[0], means.y_edge[-1]],
            aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[4].set_title("Residual")
    fig.colorbar(img, ax=ax[4])



# Given a list of Xes, find the linear model that predicts Y
def linear_fit(x, y, method):
    assert len(y) == len(x), "Arrays need to be the same length. You might need to transpose X"
    assert method in ["lasso", "least_squares"]


    # assert type(y) is np.ndarray and type(x) is np.ndarray
    # x = preprocessing.MinMaxScaler().fit_transform(x)
    print(x.shape)

    if method == "lasso":
        clf = linear_model.LassoCV(cv=12, alphas=np.logspace(-1, -7))
        clf.fit(x, y)
    elif method == "least_squares":
        clf = linear_model.LinearRegression(normalize=True)
        clf.fit(x, y)
    # _, ax = plt.subplots()
    # ax.plot(np.log10(clf.alphas_), np.mean(clf.mse_path_, axis=1), label="All")
    return clf.coef_, clf.intercept_


#### Non correlation matrix stuff

def sm_at_fixed_hm_mm_split(catalog, key):
    _, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    mm_scale = cdata["scale_of_last_MM"]

    mm_scale_bins = [
            np.min(mm_scale),
            np.percentile(mm_scale, 20),
            np.percentile(mm_scale, 80),
            np.max(mm_scale),
    ]
    print(mm_scale_bins)
    ax = sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, mm_scale, mm_scale_bins, ["Old halos", "Young halos"], key)
    return ax

def sm_at_fixed_hm_age_split(catalog, key, ax=None, fit=None):
    if ax is None:
        _, ax = plt.subplots()

    cdata = catalog[key]["data"]
    cdata = cdata[cdata["m"] > 1e13]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    hm_scale = cdata["Halfmass_Scale"]

    hm_scale_bins = [
            np.min(hm_scale),
            np.percentile(hm_scale, 20),

            np.percentile(hm_scale, 80),
            np.max(hm_scale),
    ]
    print(hm_scale_bins)
    ax = sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, hm_scale, hm_scale_bins, ["Oldest halos", "Youngest halos"], key, fit)

    return ax

def sm_at_fixed_hm_conc_split(catalog, key, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    cdata = catalog[key]["data"]
    cdata = cdata[cdata["m"] > 1e13]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    concentrations = cdata["rvir"] / cdata["rs"]

    concentration_bins = [
            np.floor(np.min(concentrations)),
            np.floor(np.percentile(concentrations, 20)),
            np.floor(np.percentile(concentrations, 80)),
            np.ceil(np.max(concentrations)),
    ]
    print(concentration_bins)
    ax = sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, concentrations, concentration_bins, ["Low concentration", "High concentration"], key)
    return ax

def sm_at_fixed_hm_split(ax, stellar_masses, halo_masses, split_params, split_bins, legend, star_key, fit=None):
    hm_bin_edges = np.arange(np.min(halo_masses), np.max(halo_masses), 0.1)
    hm_bin_midpoints = hm_bin_edges[:-1] + np.diff(hm_bin_edges) / 2

    # Find various stats on our data
    mean_sm, _, _, _ = scipy.stats.binned_statistic_2d(split_params, halo_masses, stellar_masses, statistic="mean", bins=[split_bins, hm_bin_edges])
    std_sm, _, _, _ = scipy.stats.binned_statistic_2d(split_params, halo_masses, stellar_masses, statistic="std", bins=[split_bins, hm_bin_edges])

    mean_sm, std_sm, hm_bin_midpoints = _cleanup_nans(mean_sm, std_sm, hm_bin_midpoints)

    ax.plot(hm_bin_midpoints[0], mean_sm[0], label=legend[0])
    ax.plot(hm_bin_midpoints[-1], mean_sm[-1], label=legend[1])
    ax.fill_between(hm_bin_midpoints[0], mean_sm[0]-std_sm[0], mean_sm[0]+std_sm[0], alpha=0.5, facecolor="tab:blue")
    ax.fill_between(hm_bin_midpoints[-1], mean_sm[-1]-std_sm[-1], mean_sm[-1]+std_sm[-1], alpha=0.5, facecolor="tab:orange")
    ax.set(
        xlabel=l.m_vir_x_axis,
        ylabel=l.m_star_x_axis(star_key),
    )


    from importlib import reload
    reload(l)
    if fit is not None:
        ax.plot(hm_bin_midpoints[-1], smhm_fit.f_shmr(hm_bin_midpoints[-1], *fit), label=l.mstar_mhalo_fit(star_key), c="black", ls=":")
        print(hm_bin_midpoints[-1])
        print(smhm_fit.f_shmr(hm_bin_midpoints[-1], *fit))

    ax.legend(fontsize="xx-small")
    return ax

def conc_sm_heatmap_at_fixed_hm(catalog, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    cdata = catalog["data"]
    stellar_masses = np.log10(cdata["icl"] + cdata["sm"])
    halo_masses = np.log10(cdata["m"])
    predicted_stellar_masses = smhm_fit.f_shmr(halo_masses, *catalog["fit"])
    delta_stellar_masses = stellar_masses - predicted_stellar_masses

    concentrations = cdata["rvir"] / cdata["rs"]
    # ax.hist(concentrations)
    matrix = np.vstack((delta_stellar_masses, concentrations))
    print(matrix.shape)
    print(np.cov(matrix))

    _, _, _, img = ax.hist2d(concentrations, delta_stellar_masses, bins=20)
    fig.colorbar(img)

def _cleanup_nans(y, yerr, x):
    good_y, good_yerr, good_x = [], [], []
    for i in range(len(y)):
        good_idxs = ((np.isfinite(y[i])) & (yerr[i] != 0))
        good_y.append(y[i][good_idxs])
        good_yerr.append(yerr[i][good_idxs])
        good_x.append(x[good_idxs])
    return np.array(good_y), np.array(good_yerr), np.array(good_x)
