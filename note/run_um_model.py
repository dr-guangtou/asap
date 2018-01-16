import os
import sys
import copy
from time import time

from astropy.table import Table, vstack

import numpy as np

import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

from astroML.stats import binned_statistic_2d

from cap_loess_2d import loess_2d
from cap_plot_velfield import plot_velfield

plt.rcParams['figure.dpi'] = 100.0
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12.0
plt.rc('text', usetex=True)

from hsc_massive import \
    s16a_path, \
    sample_selection, \
    prepare_sed, \
    catalog_summary, \
    mass_function, \
    smhm, \
    plotting

from um_model_plot import \
    plot_logmh_sig_logms_tot, \
    plot_logmh_logms_tot, \
    display_obs_smf

from um_ins_exs_model import InsituExsituModel


um_test = InsituExsituModel(obs_dir='../data/s16a_massive_wide2/',
                            model_type='frac1',
                            um_mtot_nbin=100,
                            um_min_nobj_per_bin=15,
                            mcmc_wl_weight=0.15,
                            mcmc_nsamples=60,
                            mcmc_nburnin=30,
                            mcmc_nwalkers=300)

um_test.mcmcFit()
