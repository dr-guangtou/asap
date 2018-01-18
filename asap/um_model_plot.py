"""QA plots for UM model."""

import numpy as np

from astroML.stats import binned_statistic_2d

import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
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

    hexbin = ax1.hexbin(logmh, logms_tot, gridsize=(45, 30),
                        alpha=0.7, bins='log',
                        mincnt=10, edgecolor='none', cmap='Oranges')
    cbar_ax = fig.add_axes([0.22, 0.92, 0.5, 0.05])
    cbar = fig.colorbar(hexbin, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r'$\log \mathrm{N}$')
    cbar.solids.set_edgecolor("face")

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


def display_obs_smf(obs_smf_mtot, obs_smf_minn,
                    obs_smf_full=None,
                    label_mtot=r'$M_{\star,\ 100\mathrm{kpc}}$',
                    label_mfull=r'$M_{\star,\ \mathrm{S82}}$',
                    label_minn=r'$M_{\star,\ 10\mathrm{kpc}}$'):
    """Display observed stellar mass functions."""
    if obs_smf_full is None:
        smf_list = [obs_smf_mtot, obs_smf_minn]
        label_list = [label_mtot, label_minn]
    else:
        obs_smf_full['logm_mean'] += 0.1
        smf_list = [obs_smf_mtot, obs_smf_minn, obs_smf_full]
        label_list = [label_mtot, label_minn, label_mfull]

    return show_smf(smf_list, label_list,
                    text=r'$\mathrm{HSC}$')


def show_smf(smf_list, label_list=None, text=None, loc=1,
             legend_fontsize=20):
    """Plot stellar mass functions."""
    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.17, right=0.994,
                        bottom=0.12, top=0.994,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)

    ax1.grid(linestyle='--', linewidth=2, alpha=0.5, zorder=0)

    m_list = ['o', '+', 's', 'h', 'x', 'H', '8', 'v', '<', '>']
    s_list = [15, 30, 20, 20, 30, 15, 15, 20, 20, 20]
    a_list = [0.4, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    if label_list is not None:
        assert len(smf_list) == len(label_list)

    for ii, smf in enumerate(smf_list):
        if label_list is not None:
            label_use = label_list[ii]
        else:
            label_use = '__no_lable__'
        ax1.fill_between(smf['logm_mean'],
                         np.log10(smf['smf_low']),
                         np.log10(smf['smf_upp']),
                         alpha=a_list[ii],
                         facecolor=c_list[ii],
                         label=label_use)
        ax1.scatter(smf['logm_mean'],
                    np.log10(smf['smf']),
                    marker=m_list[ii], c=c_list[ii],
                    s=s_list[ii], label='__no_label')

    ax1.set_xlim(11.19, 12.35)
    ax1.set_ylim(-7.9, -2.4)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.legend(fontsize=legend_fontsize, loc=loc)

    ax1.set_xlabel(r'$\log (M_{\star}/M_{\odot})$',
                   fontsize=25)
    ax1.set_ylabel((r'$\mathrm{d}N/\mathrm{d}\log M_{\star}\ $'
                    r'$[{\mathrm{Mpc}^{-3}}{\mathrm{dex}^{-1}}]$'),
                   size=25)

    if text is not None:
        ax1.text(0.15, 0.06, text,
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 fontsize=25,
                 transform=ax1.transAxes)

    return fig


def plot_mtot_minn_smf(obs_smf_tot, obs_smf_inn,
                       obs_logms_tot, obs_logms_inn,
                       um_smf_tot, um_smf_inn,
                       logms_mod_tot, logms_mod_inn,
                       obs_smf_full=None,
                       shmr_a=None, shmr_b=None,
                       sigms_a=None, sigms_b=None,
                       um_smf_tot_all=None, **kwargs):
    """Plot the UM predicted M100-M10 plane and their SMFs."""
    fig, axes = plt.subplots(2, figsize=(7, 9))
    ax1 = axes[0]
    ax2 = axes[1]

    # Scatter plot
    if len(logms_mod_tot) > len(obs_logms_tot):
        ax1.scatter(logms_mod_tot, logms_mod_inn,
                    label=r'$\mathrm{Model}$',
                    s=10, alpha=0.6, marker='o',
                    c='royalblue')

        ax1.scatter(obs_logms_tot, obs_logms_inn,
                    label=r'$\mathrm{Data}$',
                    s=15, alpha=0.5, marker='+',
                    c='lightsalmon')
    else:
        ax1.scatter(obs_logms_tot, obs_logms_inn,
                    label=r'$\mathrm{Data}$',
                    s=10, alpha=0.6, marker='o',
                    c='lightsalmon')

        ax1.scatter(logms_mod_tot, logms_mod_inn,
                    label=r'$\mathrm{Model}$',
                    s=15, alpha=0.5, marker='+',
                    c='royalblue')

    ax1.legend(fontsize=15, loc='lower right')
    ax1.grid(linestyle='--', linewidth=2, alpha=0.3, zorder=0)

    ax1.set_xlabel(r'$\log M_{\star,\ \mathrm{100,\ UM}}$', fontsize=20)
    ax1.set_ylabel(r'$\log M_{\star,\ \mathrm{10,\ UM}}$', fontsize=20)

    ax1.set_xlim(np.nanmin(logms_mod_tot) - 0.09,
                 np.nanmax(logms_mod_tot) + 0.09)
    ax1.set_ylim(np.nanmin(logms_mod_inn) - 0.02,
                 np.nanmax(logms_mod_inn) + 0.09)

    if shmr_a is not None and shmr_b is not None:
        seg1 = (r'$\log M_{\star} = %6.3f \times$' % shmr_a)
        seg2 = (r'$\log M_{\rm halo} + %6.3f$' % shmr_b)
        ax1.text(0.26, 0.91, (seg1 + seg2),
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 fontsize=12,
                 transform=ax1.transAxes)

    if sigms_a is not None and sigms_b is not None:
        seg1 = (r'$\sigma(\log M_{\star}) = %6.3f \times$' % sigms_a)
        seg2 = (r'$\log M_{\rm halo} + %6.3f$' % sigms_b)
        ax1.text(0.26, 0.83, (seg1 + seg2),
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 fontsize=12,
                 transform=ax1.transAxes)

    # Full SMF in the background if available
    #  +0.1 dex is a magic number to convert S82 SMF from BC03 to
    #  FSPS model
    ax2.grid(linestyle='--', linewidth=2, alpha=0.3, zorder=0)

    if obs_smf_full is not None:
        ax2.plot(obs_smf_full['logm_mean'] + 0.1,
                 np.log10(obs_smf_full['smf']),
                 c='mediumseagreen', alpha=0.60, zorder=0,
                 label=r'$\mathrm{Data:\ S82}$')

        ax2.scatter(obs_smf_full['logm_mean'] + 0.1,
                    np.log10(obs_smf_full['smf']),
                    c='seagreen', marker='s',
                    s=10, label='__no_label__',
                    alpha=1.0, zorder=0)

    if um_smf_tot_all is not None:
        ax2.plot(um_smf_tot_all['logm_mean'],
                 np.log10(um_smf_tot_all['smf']),
                 linewidth=1.5, linestyle='--',
                 c='royalblue',
                 label='__no_label__')

    # SMF plot
    ax2.fill_between(obs_smf_tot['logm_mean'],
                     np.log10(obs_smf_tot['smf_low']),
                     np.log10(obs_smf_tot['smf_upp']),
                     facecolor='steelblue',
                     edgecolor='none',
                     interpolate=True,
                     alpha=0.4,
                     label=r'$\mathrm{Data:\ Mtot}$')

    ax2.fill_between(obs_smf_inn['logm_mean'],
                     np.log10(obs_smf_inn['smf_low']),
                     np.log10(obs_smf_inn['smf_upp']),
                     facecolor='lightsalmon',
                     edgecolor='none',
                     interpolate=True,
                     alpha=0.4,
                     label=r'$\mathrm{Data:\ Minn}$')

    ax2.scatter(obs_smf_inn['logm_mean'],
                np.log10(obs_smf_inn['smf']),
                marker='h',
                c='lightsalmon',
                s=20, label='__no_label__',
                alpha=1.0)

    ax2.scatter(obs_smf_tot['logm_mean'],
                np.log10(obs_smf_tot['smf']),
                marker='h',
                c='steelblue',
                s=20, label='__no_label__',
                alpha=1.0)

    ax2.plot(um_smf_tot['logm_mean'],
             np.log10(um_smf_tot['smf']),
             linewidth=4, linestyle='--',
             c='royalblue',
             label=r'$\mathrm{UM:\ Mtot}$')

    ax2.plot(um_smf_inn['logm_mean'],
             np.log10(um_smf_inn['smf']),
             linewidth=4, linestyle='--',
             c='salmon',
             label=r'$\mathrm{UM:\ Minn}$')

    ax2.legend(fontsize=12, loc='upper right')

    ax2.set_xlabel(r'$\log (M_{\star}/M_{\odot})$',
                   fontsize=20)
    ax2.set_ylabel((r'$\mathrm{d}N/\mathrm{d}\log M_{\star}\ $'
                    r'$[{\mathrm{Mpc}^{-3}}{\mathrm{dex}^{-1}}]$'),
                   size=20)

    mask_inn = np.log10(obs_smf_inn['smf']) > -7.5
    mask_tot = np.log10(obs_smf_tot['smf']) > -7.5

    ax2.set_xlim(np.nanmin(obs_smf_inn[mask_inn]['logm_mean']) - 0.15,
                 np.nanmax(obs_smf_tot[mask_tot]['logm_mean']) + 0.55)

    if obs_smf_full is not None:
        ax2.set_ylim(np.nanmin(np.log10(obs_smf_inn[mask_inn]['smf']))
                     - 0.2,
                     np.nanmax(np.log10(obs_smf_full['smf']))
                     )
    else:
        ax2.set_ylim(np.nanmin(np.log10(obs_smf_inn[mask_inn]['smf']))
                     - 0.2,
                     np.nanmax(np.log10(obs_smf_tot[mask_tot]['smf']))
                     + 0.8)

    return fig


def plot_dsigma_profiles(obs_wl_dsigma, um_wl_profs,
                         obs_mhalo=None, um_wl_mhalo=None, **kwargs):
    """Plot the UM predicted weak lensing profiles."""
    obs_wl_n_bin = len(um_wl_profs)
    if obs_wl_n_bin <= 4:
        n_col = obs_wl_n_bin
        n_row = 1
    else:
        n_col = 4
        n_row = int(np.ceil(obs_wl_n_bin / 4.0))

    fig = plt.figure(figsize=(3 * n_col, 3.8 * n_row))
    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.0, hspace=0.00)

    for ii in range(obs_wl_n_bin):

        ax = plt.subplot(gs[ii])
        ax.loglog()

        if ii % n_col != 0:
            ax.yaxis.set_major_formatter(NullFormatter())
        else:
            ax.set_ylabel(r'$\Delta\Sigma$ $[M_{\odot}/{\rm pc}^2]$',
                          size=20)
        if ii % n_row != 0:
            ax.xaxis.set_major_formatter(NullFormatter())
        else:
            ax.set_xlabel(r'$r_{\rm p}$ ${\rm [Mpc]}$',
                          size=20)

        ax.grid(linestyle='--', linewidth=1, alpha=0.3, zorder=0)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        # Observed WL profile
        obs_prof = obs_wl_dsigma[ii]
        ax.errorbar(obs_prof.r, obs_prof.sig,
                    obs_prof.err_w, fmt='o',
                    color='salmon',
                    ecolor='lightsalmon',
                    alpha=0.9)
        ax.plot(obs_prof.r, obs_prof.sig,
                linewidth=0.5, color='salmon',
                alpha=0.5)

        # Label the mass range
        ax.text(0.04, 0.28,
                r'${\rm Bin: %d}$' % obs_prof.bin_id,
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=12.0,
                transform=ax.transAxes,
                color='k', alpha=0.9)

        ax.text(0.04, 0.18,
                r"$\log M_{\rm tot}:[%5.2f,%5.2f]$" % (obs_prof.low_mtot,
                                                       obs_prof.upp_mtot),
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=12.0,
                transform=ax.transAxes,
                color='k', alpha=0.9)

        ax.text(0.04, 0.08,
                r"$\log M_{\rm inn}:[%5.2f,%5.2f]$" % (obs_prof.low_minn,
                                                       obs_prof.upp_minn),
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=12.0,
                transform=ax.transAxes,
                color='k', alpha=0.9)

        # Predicted WL profile
        ax.scatter(obs_prof.r, um_wl_profs[ii],
                   marker='h', s=5, c='b', alpha=0.7)
        ax.plot(obs_prof.r, um_wl_profs[ii],
                linewidth=2.0, color='royalblue', alpha=0.7)

        if um_wl_mhalo is not None:
            ax.text(0.35, 0.90,
                    r"$[\log M_{\rm Vir, UM}]=%5.2f$" % (um_wl_mhalo[ii]),
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=15.0,
                    transform=ax.transAxes,
                    color='royalblue')

        # X, Y Limits
        ax.set_xlim(0.05, 61.0)
        ax.set_ylim(0.01, 399.0)

    return fig


def plot_best_fit_scatter_relation(sigms_a, sigms_b, min_scatter=0.02):
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

    logmh_cen = np.linspace(11.5, 15.0, 1000)
    sig_ms = sigms_a * np.asarray(logmh_cen) + sigms_b
    sig_ms = np.where(sig_ms <= min_scatter, min_scatter, sig_ms)

    ax1.plot(logmh_cen, sig_ms,
             linewidth=4.0, linestyle='--', alpha=0.8)

    ax1.text(0.25, 0.09, r"$a=%5.2f\ b=%5.2f$" % (sigms_a, sigms_b),
             verticalalignment='bottom',
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlabel(r'$\log M_{\mathrm{vir}}$', fontsize=25)
    ax1.set_ylabel(r'$\sigma_{\log M_{\star, \rm Total}}$',
                   fontsize=28)

    return fig


def plot_best_fit_shmr(shmr_a, shmr_b, sigms_a, sigms_b):
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

    logmh_cen = np.linspace(11.5, 15.0, 50)
    ax1.plot(logmh_cen, shmr_a * logmh_cen + shmr_b,
             linewidth=5.0, linestyle='--', alpha=0.8)

    ax1.text(0.75, 0.09, r"$a=%5.2f\ b=%5.2f$" % (shmr_a, shmr_b),
             verticalalignment='bottom',
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlabel(r'$\log M_{\mathrm{vir}}$', fontsize=25)
    ax1.set_ylabel(r'$\log M_{\star, \rm Total}$', fontsize=25)

    return fig


def plot_mtot_minn_trend():
    """"""
