"""QA plots for UM model."""
from __future__ import print_function, division

import numpy as np

from astroML.stats import binned_statistic_2d

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator

from scipy.ndimage.filters import gaussian_filter

import corner
from palettable.colorbrewer.sequential import OrRd_3, OrRd_8, Greys_9, PuBu_4, Purples_9
ORG = OrRd_8.mpl_colormap
ORG_2 = OrRd_3.mpl_colormap
BLU = PuBu_4.mpl_colormap
BLK = Greys_9.mpl_colormap
PUR = Purples_9.mpl_colormap

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

    ax1.plot(logmh_cen,
             sigms_a * (np.asarray(logmh_cen) - 15.3) + sigms_b,
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
                       um_smf_tot_all=None,
                       not_table=False,
                       **kwargs):
    """Plot the UM predicted M100-M10 plane and their SMFs."""
    fig, axes = plt.subplots(2, figsize=(7, 9))
    fig.subplots_adjust(left=0.145, right=0.995,
                        bottom=0.085, top=0.995,
                        wspace=0.00, hspace=0.21)
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

    ax1.legend(fontsize=20, loc='lower right')
    ax1.grid(linestyle='--', linewidth=2, alpha=0.3, zorder=0)

    ax1.set_xlabel(r'$\log M_{\star,\ \mathrm{Tot,\ UM}}$', fontsize=25)
    ax1.set_ylabel(r'$\log M_{\star,\ \mathrm{Inn,\ UM}}$', fontsize=25)

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
        ax2.errorbar(obs_smf_full['logm_mean'] + 0.15,
                     np.log10(obs_smf_full['smf']),
                     (np.log10(obs_smf_full['smf_upp']) -
                      np.log10(obs_smf_full['smf'])),
                     fmt='o', color='seagreen',
                     ecolor='seagreen',
                     alpha=0.9, marker='s',
                     label=r'$\mathrm{Data:\ PRIMUS}$',
                     zorder=0)

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

    if isinstance(um_smf_inn, (list,)):
        for ii, smf in enumerate(um_smf_inn):
            if ii == 0:
                if not_table:
                    ax2.plot(obs_smf_inn['logm_mean'],
                             np.log10(smf),
                             linewidth=1, linestyle='-',
                             c='salmon', alpha=0.7,
                             label=r'$\mathrm{UM:\ Minn}$')
                else:
                    ax2.plot(smf['logm_mean'],
                             np.log10(smf['smf']),
                             linewidth=1, linestyle='-',
                             c='salmon', alpha=0.7,
                             label=r'$\mathrm{UM:\ Minn}$')
            else:
                if not_table:
                    ax2.plot(obs_smf_inn['logm_mean'],
                             np.log10(smf),
                             linewidth=1, linestyle='-',
                             c='salmon', alpha=0.7,
                             label='__no_label__')
                else:
                    ax2.plot(smf['logm_mean'],
                             np.log10(smf['smf']),
                             linewidth=1, linestyle='-',
                             c='salmon', alpha=0.7,
                             label='__no_label__')
    else:
        if not_table:
            ax2.plot(obs_smf_inn['logm_mean'],
                     np.log10(um_smf_inn),
                     linewidth=4, linestyle='--',
                     c='salmon',
                     label=r'$\mathrm{UM:\ Minn}$')
        else:
            ax2.plot(um_smf_inn['logm_mean'],
                     np.log10(um_smf_inn['smf']),
                     linewidth=4, linestyle='--',
                     c='salmon',
                     label=r'$\mathrm{UM:\ Minn}$')

    if isinstance(um_smf_tot, (list,)):
        for ii, smf in enumerate(um_smf_tot):
            if ii == 0:
                if not_table:
                    ax2.plot(obs_smf_tot['logm_mean'],
                             np.log10(smf),
                             linewidth=1, linestyle='-',
                             c='royalblue', alpha=0.7,
                             label=r'$\mathrm{UM:\ Mtot}$')
                else:
                    ax2.plot(smf['logm_mean'],
                             np.log10(smf['smf']),
                             linewidth=1, linestyle='-',
                             c='royalblue', alpha=0.7,
                             label=r'$\mathrm{UM:\ Mtot}$')
            else:
                if not_table:
                    ax2.plot(obs_smf_tot['logm_mean'],
                             np.log10(smf),
                             linewidth=1, linestyle='-',
                             c='royalblue', alpha=0.7,
                             label='__no_label__')
                else:
                    ax2.plot(smf['logm_mean'],
                             np.log10(smf['smf']),
                             linewidth=1, linestyle='-',
                             c='royalblue', alpha=0.7,
                             label='__no_label__')
    else:
        if not_table:
            ax2.plot(obs_smf_tot['logm_mean'],
                     np.log10(um_smf_tot),
                     linewidth=4, linestyle='--',
                     c='royalblue',
                     label=r'$\mathrm{UM:\ Mtot}$')
        else:
            ax2.plot(um_smf_tot['logm_mean'],
                     np.log10(um_smf_tot['smf']),
                     linewidth=4, linestyle='--',
                     c='royalblue',
                     label=r'$\mathrm{UM:\ Mtot}$')

    ax2.legend(fontsize=15, loc='upper right')

    ax2.set_xlabel(r'$\log (M_{\star}/M_{\odot})$',
                   fontsize=25)
    ax2.set_ylabel((r'$\mathrm{d}N/\mathrm{d}\log M_{\star}\ $'
                    r'$[{\mathrm{Mpc}^{-3}}{\mathrm{dex}^{-1}}]$'),
                   size=25)

    mask_inn = np.log10(obs_smf_inn['smf']) > -7.5
    mask_tot = np.log10(obs_smf_tot['smf']) > -7.5

    ax2.set_xlim(np.nanmin(obs_smf_inn[mask_inn]['logm_mean']) - 0.15,
                 np.nanmax(obs_smf_tot[mask_tot]['logm_mean']) + 0.45)

    if obs_smf_full is not None:
        ax2.set_ylim(np.nanmin(np.log10(obs_smf_inn[mask_inn]['smf']))
                     - 0.2,
                     np.nanmax(np.log10(obs_smf_full['smf'])))
    else:
        ax2.set_ylim(np.nanmin(np.log10(obs_smf_inn[mask_inn]['smf']))
                     - 0.2,
                     np.nanmax(np.log10(obs_smf_tot[mask_tot]['smf']))
                     + 0.8)
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    
    fig.savefig('asap_mtot_minn_smf.pdf', dpi=100)

    return fig


def plot_dsigma_profiles(obs_wl_dsigma, um_wl_profs,
                         obs_mhalo=None, um_mhalo=None,
                         each_col=3, reference=None,
                         **kwargs):
    """Plot the UM predicted weak lensing profiles."""
    obs_wl_n_bin = len(obs_wl_dsigma)
    if obs_wl_n_bin <= each_col:
        n_row = obs_wl_n_bin
        n_col = 1
    else:
        n_row = each_col
        n_col = int(np.ceil(obs_wl_n_bin / each_col))

    fig = plt.figure(figsize=(4 * n_col, 3.5 * n_row))
    fig.subplots_adjust(left=0.075, right=0.995,
                        bottom=0.09, top=0.995,
                        wspace=0.00, hspace=0.00)

    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.0, hspace=0.00)

    y_min_arr = np.array(
        [np.nanmin(prof.sig) for prof in obs_wl_dsigma])
    y_min_arr = np.where(y_min_arr <= 0.0, np.nan, y_min_arr)
    y_max_arr = np.array(
        [np.nanmax(prof.sig) for prof in obs_wl_dsigma])
    y_min = np.nanmin(y_min_arr) * 0.5
    y_max = np.nanmax(y_max_arr) * 1.5

    if reference is not None:
        ref_prof = obs_wl_dsigma[reference]
    else:
        ref_prof = None

    for ii in range(obs_wl_n_bin):
        col_id = int(np.floor(ii / n_row))
        row_id = int(n_row - (ii + 1 - col_id * n_row))

        ax = plt.subplot(gs[row_id, col_id])
        ax.loglog()

        ax.grid(linestyle='--', linewidth=1, alpha=0.3, zorder=0)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)

        if ref_prof is not None:
            ax.plot(ref_prof.r, ref_prof.sig, linewidth=2.5, 
                    color=PUR(0.7), linestyle='--', alpha=0.5)

        # Observed WL profile
        obs_prof = obs_wl_dsigma[ii]
        ax.errorbar(obs_prof.r, obs_prof.sig,
                    obs_prof.err_w, fmt='o',
                    color='salmon',
                    ecolor='lightsalmon',
                    markersize=9,
                    alpha=0.8)
        ax.plot(obs_prof.r, obs_prof.sig,
                linewidth=1.0, color='salmon',
                alpha=0.5)

        if reference is not None and reference == ii:
            ax.text(0.04, 0.41, r'$\mathrm{Ref}$',
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=23.0,
                    transform=ax.transAxes,
                    color=ORG(0.7), alpha=0.9)

        # Label the mass range
        ax.text(0.04, 0.29,
                r'${\rm Bin: %d}$' % obs_prof.bin_id,
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=23.0,
                transform=ax.transAxes,
                color='k', alpha=0.9)

        ax.text(0.04, 0.18,
                r"$\log M_{\rm tot}:[%5.2f,%5.2f]$" % (obs_prof.low_mtot,
                                                       obs_prof.upp_mtot),
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=17.0,
                transform=ax.transAxes,
                color='k', alpha=0.9)

        ax.text(0.04, 0.08,
                r"$\log M_{\rm inn}:[%5.2f,%5.2f]$" % (obs_prof.low_minn,
                                                       obs_prof.upp_minn),
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=17.0,
                transform=ax.transAxes,
                color='k', alpha=0.9)

        # Predicted WL profile
        if isinstance(um_wl_profs[0], (list,)):
            for dsig in um_wl_profs:
                ax.plot(obs_prof.r, dsig[ii],
                        linewidth=2.5, color='royalblue',
                        alpha=0.7)
        else:
            ax.scatter(obs_prof.r, um_wl_profs[ii],
                       marker='h', s=7, c='b', alpha=1.0)
            ax.plot(obs_prof.r, um_wl_profs[ii],
                    linewidth=4.0, color='royalblue', alpha=0.7)

        if um_mhalo is not None:
            ax.text(0.55, 0.92,
                    r"$[%5.2f \pm %5.2f]$" % um_mhalo[ii],
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=18.0,
                    transform=ax.transAxes,
                    color='royalblue')

        # X, Y Limits
        x_min = np.min(obs_prof.r) * 0.2
        x_max = np.max(obs_prof.r) * 1.8
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if col_id != 0:
            ax.yaxis.set_major_formatter(NullFormatter())
        else:
            ax.set_ylabel(r'$\Delta\Sigma$ $[M_{\odot}/{\rm pc}^2]$',
                          fontsize=30)
        if row_id == (n_row - 1):
            ax.set_xlabel(r'$r_{\rm p}$ ${\rm [Mpc]}$',
                          fontsize=30)
        else:
            ax.xaxis.set_major_formatter(NullFormatter())

    fig.savefig('asap_dsigma_profs.pdf', dpi=120)

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

    logmh_cen = np.linspace(11.5, 15.4, 1000)
    sig_ms = sigms_a * (np.asarray(logmh_cen) - 15.3) + sigms_b
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


def plot_mcmc_trace(mcmc_chains, mcmc_labels, mcmc_best=None,
                    mcmc_burnin=None, burnin_alpha=0.2, 
                    trace_alpha=0.2):
    """Traceplot for MCMC results."""
    if mcmc_burnin is not None:
        fig = plt.figure(figsize=(12, 15))
    else:
        fig = plt.figure(figsize=(10, 15))

    fig.subplots_adjust(hspace=0.0, wspace=0.0,
                        bottom=0.027, top=0.97,
                        left=0.06, right=0.94)

    # I want the plot of individual walkers to span 2 columns
    nparam = len(mcmc_labels)
    if mcmc_burnin is not None:
        gs = gridspec.GridSpec(nparam, 5)
    else:
        gs = gridspec.GridSpec(nparam, 3)

    if mcmc_best is not None:
        assert len(mcmc_best) == len(mcmc_labels)

    for ii, param in enumerate(mcmc_labels):
        param_chain = mcmc_chains[:, :, ii]
        max_var = max(np.var(param_chain[:, :], axis=1))

        # Trace plot
        if mcmc_burnin is None:
            ax1 = plt.subplot(gs[ii, :2])
        else:
            ax1 = plt.subplot(gs[ii, 2:4])
        ax1.yaxis.grid(linewidth=1.5, linestyle='--', alpha=0.5)

        for walker in param_chain:
            ax1.plot(np.arange(len(walker)), walker,
                     drawstyle="steps",
                     color=ORG_2(1.0 - np.var(walker) / max_var),
                     alpha=trace_alpha)

            if mcmc_burnin is None:
                ax1.set_ylabel(param,
                               fontsize=28,
                               labelpad=18,
                               color='k')

            # Don't show ticks on the y-axis
            ax1.tick_params(labelleft='off')

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii != (nparam - 1):
            ax1.tick_params(labelbottom='off')
        else:
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)

        # Posterior histograms
        ax2 = plt.subplot(gs[ii, -1])
        ax2.grid(linewidth=1.5, linestyle='--', alpha=0.5)

        ax2.hist(np.ravel(param_chain[:, :]),
                 bins=np.linspace(ax1.get_ylim()[0],
                                  ax1.get_ylim()[1],
                                  100),
                 orientation='horizontal',
                 alpha=0.7,
                 facecolor=ORG_2(0.9),
                 edgecolor="none")

        ax1.set_xlim(1, len(walker))
        ax1.set_ylim(np.min(param_chain[:, 0]), np.max(param_chain[:, 0]))
        ax2.set_ylim(ax1.get_ylim())

        ax1.get_xaxis().tick_bottom()
        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        if mcmc_best is not None:
            ax1.axhline(mcmc_best[ii], linestyle='--', linewidth=2,
                        c=BLU(1.0), alpha=0.8)
            ax2.axhline(mcmc_best[ii], linestyle='--', linewidth=2,
                        c=BLU(1.0), alpha=0.8)

        # Trace plot for burnin
        if mcmc_burnin is not None:
            param_burnin = mcmc_burnin[:, :, ii]
            ax3 = plt.subplot(gs[ii, :2])
            ax3.yaxis.grid(linewidth=1.5, linestyle='--', alpha=0.5)

            for walker in param_burnin:
                ax3.plot(np.arange(len(walker)), walker,
                         drawstyle="steps",
                         color=BLU(np.var(walker) / max_var),
                         alpha=burnin_alpha)

                ax3.set_ylabel(param,
                               fontsize=25,
                               labelpad=18,
                               color='k')

                # Don't show ticks on the y-axis
                ax3.tick_params(labelleft='off')
                ax3.set_xlim(1, len(walker))
                ax3.set_ylim(np.min(param_chain[:, 0]),
                             np.max(param_chain[:, 0]))
                ax3.get_xaxis().tick_bottom()

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii != (nparam - 1):
            ax1.xaxis.set_visible(False)
            if mcmc_burnin is not None:
                ax3.xaxis.set_visible(False)
        else:
            for tick in ax3.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)

        if ii == 0:
            t = ax1.set_title("Sampling", fontsize=28, color='k')
            t.set_y(1.01)
            t = ax2.set_title("Posterior", fontsize=28, color='k')
            t.set_y(1.01)
            if mcmc_burnin is not None:
                t = ax3.set_title("Burnin", fontsize=28, color='k')
                t.set_y(1.01)

    return fig


def plot_mcmc_corner(mcmc_samples, mcmc_labels, **corner_kwargs):
    """Corner plots for MCMC samples."""
    fig = corner.corner(
        mcmc_samples,
        bins=25, color=ORG(0.7),
        smooth=1, labels=mcmc_labels,
        label_kwargs={'fontsize': 26},
        quantiles=[0.16, 0.5, 0.84],
        plot_contours=True,
        fill_contours=True,
        show_titles=True,
        title_kwargs={"fontsize": 20},
        hist_kwargs={"histtype": 'stepfilled',
                     "alpha": 0.4,
                     "edgecolor": "none"},
        use_math_text=True,
        **corner_kwargs
        )

    return fig


def plot_mtot_minn_trend(
        x_arr, y_arr, z_arr, method='count',
        x_bins=40, y_bins=30, z_min=None, z_max=None,
        contour=True, nticks=10, x_lim=None, y_lim=None,
        n_contour=6, scatter=False, gaussian=0.05,
        xlabel=r'$\log (M_{\star,\ \mathrm{Total}}/M_{\odot})$',
        ylabel=r'$\log (M_{\star,\ \mathrm{Inner}}/M_{\odot})$',
        title=r'$\log M_{\mathrm{Halo}}$',
        x_title=0.6, y_title=0.1, s_alpha=0.08, s_size=10):
    """Density plot."""
    if x_lim is None:
        x_lim = [np.nanmin(x_arr), np.nanmax(x_arr)]
    if y_lim is None:
        y_lim = [np.nanmin(y_arr), np.nanmax(y_arr)]

    x_mask = ((x_arr >= x_lim[0]) & (x_arr <= x_lim[1]))
    y_mask = ((y_arr >= y_lim[0]) & (y_arr <= y_lim[1]))
    x_arr = x_arr[x_mask & y_mask]
    y_arr = y_arr[x_mask & y_mask]
    z_arr = z_arr[x_mask & y_mask]

    z_stats, x_edges, y_edges = binned_statistic_2d(
        x_arr, y_arr, z_arr, method, bins=(x_bins, y_bins))

    if z_min is None:
        z_min = np.nanmin(z_stats)
    if z_max is None:
        z_max = np.nanmax(z_stats)

    fig = plt.figure(figsize=(8.0, 7))
    fig.subplots_adjust(left=0.14, right=0.93,
                        bottom=0.12, top=0.99,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle='--', linewidth=2, alpha=0.5, zorder=0)

    HM = ax1.imshow(z_stats.T, origin='lower',
                    extent=[x_edges[0], x_edges[-1],
                            y_edges[0], y_edges[-1]],
                    vmin=z_min, vmax=z_max,
                    aspect='auto', interpolation='nearest',
                    cmap=ORG)

    if scatter:
        ax1.scatter(x_arr, y_arr, c=BLU(0.9), alpha=s_alpha, s=s_size,
                    label='__no_label__')

    if contour:
        CT = ax1.contour(x_edges[:-1], y_edges[:-1],
                         gaussian_filter(z_stats.T, gaussian),
                         n_contour, linewidths=1.5,
                         colors=[BLK(0.6), BLK(0.7)],
                         extend='neither')
        ax1.clabel(CT, inline=1, fontsize=15)

    ax1.set_xlabel(xlabel, size=25)
    ax1.set_ylabel(ylabel, size=25)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar_ticks = MaxNLocator(nticks).tick_values(z_min, z_max)
    cbar = plt.colorbar(HM, cax=cax, ticks=cbar_ticks)
    cbar.solids.set_edgecolor("face")

    ax1.text(x_title, y_title, title, size=30, transform=ax1.transAxes)

    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    return fig


def plot_mass_scatter_fsat_trends(um_mock, logms_tot_mod, nbin=12,
                                  logmh_range=[13.0, 15.2],
                                  logms_range=[11.0, 12.4]):
    """Plot the trends among mass and scatters."""
    mask_cen = um_mock['upid'] == -1

    logms_cen = logms_tot_mod[mask_cen]
    logmh_cen = um_mock['logmh_vir'][mask_cen]

    logms_all = logms_tot_mod
    logmh_all = um_mock['logmh_vir']

    logms_bin = np.linspace(logms_range[0], logms_range[1], nbin)
    logmh_bin = np.linspace(logmh_range[0], logmh_range[1], nbin)

    idx_logms_cen = np.digitize(logms_cen, logms_bin)
    idx_logmh_cen = np.digitize(logmh_cen, logmh_bin)

    idx_logms_all = np.digitize(logms_all, logms_bin)
    idx_logmh_all = np.digitize(logmh_all, logmh_bin)

    logms_mean = [np.nanmean(logms_all[idx_logms_all == k])
                  for k in range(len(logms_bin))]
    logmh_mean = [np.nanmean(logmh_all[idx_logmh_all == k])
                  for k in range(len(logmh_bin))]

    sigmh_cen = [np.nanstd(logmh_cen[idx_logms_cen == k])
                 for k in range(len(logms_bin))]
    sigmh_all = [np.nanstd(logmh_all[idx_logms_all == k])
                 for k in range(len(logms_bin))]

    sigms_cen = [np.nanstd(logms_cen[idx_logmh_cen == k])
                 for k in range(len(logmh_bin))]
    sigms_all = [np.nanstd(logms_all[idx_logmh_all == k])
                 for k in range(len(logmh_bin))]

    frac_cen = np.array([(np.sum(mask_cen[idx_logms_all == k]) * 1.0 /
                          (len(um_mock[idx_logms_all == k])))
                         for k in range(len(logms_bin))])

    frac_sat = (1.0 - frac_cen) * 100.0

    fig, axes = plt.subplots(3, figsize=(7, 15))

    ax1 = axes[0]
    ax1.grid(linestyle='--', linewidth=2, alpha=0.3, zorder=0)

    ax1.scatter(logmh_mean, sigms_all, s=50,
                label=r'$\mathrm{All}$')
    ax1.scatter(logmh_mean, sigms_cen, s=30, alpha=0.7,
                label=r'$\mathrm{Cen}$')

    ax1.set_xlim(logmh_range)
    ax1.set_ylim(np.nanmin(sigms_cen[1:]) - 0.05,
                 np.nanmax(sigms_all[1:]) + 0.05)

    ax1.set_xlabel(r'$\log M_{\rm Halo}$', fontsize=25)
    ax1.set_ylabel(r'$\sigma_{\log M_{\star, \mathrm{Model}}}$',
                   fontsize=25)
    ax1.legend(fontsize=15, loc='upper right')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax2 = axes[1]
    ax2.grid(linestyle='--', linewidth=2, alpha=0.3, zorder=0)

    ax2.scatter(logms_mean, sigmh_all, s=50)
    ax2.scatter(logms_mean, sigmh_cen, s=30, alpha=0.7)

    ax2.set_xlabel(r'$\log M_{\star, \mathrm{Model}}$', fontsize=25)
    ax2.set_ylabel(r'$\sigma_{\log M_{\rm Halo}}$', fontsize=25)
    ax2.set_xlim(logms_range)
    ax2.set_ylim(np.nanmin(sigmh_cen[1:]) - 0.05,
                 np.nanmax(sigmh_all[1:]) + 0.05)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax3 = axes[2]
    ax3.grid(linestyle='--', linewidth=2, alpha=0.3, zorder=0)

    ax3.scatter(logms_mean, frac_sat, s=50, alpha=0.8)

    ax3.set_xlabel(r'$\log M_{\star, \mathrm{Model}}$', fontsize=25)
    ax3.set_ylabel(r'$f_{\rm Satellites}$', fontsize=25)
    ax3.set_xlim(logms_range)
    ax3.set_ylim(np.nanmin(frac_sat[1:]) - 5,
                 np.nanmax(frac_sat[1:]) + 10)

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    return fig
