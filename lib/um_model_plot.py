"""QA plots for UM model."""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

plt.rcParams['figure.dpi'] = 100.0
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12.0
plt.rc('text', usetex=True)


def plot_logmh_sig_logms_tot(logmh_cen, sig_logms_tot,
                             sigms_a, sigms_b):
    """Log Mh v.s. sig(Log Ms_tot)."""
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0.19, right=0.995,
                        bottom=0.13, top=0.995,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle='--', linewidth=2, alpha=0.4, zorder=0)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.plot(logmh_cen, sigms_a * np.asarray(logmh_cen) + sigms_b,
             linewidth=3.0, linestyle='--', alpha=0.5)
    ax1.scatter(logmh_cen, sig_logms_tot, s=70, alpha=0.8,
                edgecolor='k')

    ax1.text(0.25, 0.09, r"$a=%5.2f\ b=%5.2f$" % (sigms_a, sigms_b),
             verticalalignment='bottom',
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlabel(r'$\log M_{\mathrm{vir}}$', fontsize=25)
    ax1.set_ylabel(r'$\sigma_{\log M_{\star, \rm Total}}$',
                   fontsize=28)

    return fig


def plot_logmh_logms_tot(logmh, logms_tot,
                         shmr_a, shmr_b):
    """Log Mh v.s. Log Ms_tot."""
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0.19, right=0.995,
                        bottom=0.13, top=0.995,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle='--', linewidth=2, alpha=0.4, zorder=0)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.hexbin(logmh, logms_tot, gridsize=(50, 60), alpha=0.7,
               mincnt=10, edgecolor='none', cmap='Oranges')

    logmh_cen = np.linspace(np.nanmin(logmh), np.nanmax(logmh), 50)
    ax1.plot(logmh_cen, shmr_a * logmh_cen + shmr_b,
             linewidth=3.0, linestyle='--', alpha=0.5)

    ax1.text(0.75, 0.09, r"$a=%5.2f\ b=%5.2f$" % (shmr_a, shmr_b),
             verticalalignment='bottom',
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlabel(r'$\log M_{\mathrm{vir}}$', fontsize=25)
    ax1.set_ylabel(r'$\log M_{\star, \rm Total}$', fontsize=25)

    return fig
