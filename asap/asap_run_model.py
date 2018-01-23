#!/usr/bin/env python2
"""Model using the in-situ and ex-situ mass."""

import os
import pickle
import argparse

import emcee

import numpy as np

from scipy import interpolate

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

from halotools.mock_observables import delta_sigma_from_precomputed_pairs

from stellar_mass_function import get_smf_bootstrap
from full_mass_profile_model import mass_prof_model_simple, \
    mass_prof_model_frac1
from um_model_plot import plot_mtot_minn_smf, plot_dsigma_profiles
from asap_data_io import parse_config, load_observed_data, \
    config_observed_data, config_um_data, load_um_data
from asap_utils import mcmc_save_chains, mcmc_save_results
# from convergence import convergence_check


def initial_model(cfg, verbose=True):
    """Initialize the A.S.A.P model."""
    # Configuration for HSC data
    cfg = config_observed_data(cfg, verbose=verbose)
    obs_data = load_observed_data(cfg, verbose=verbose)

    # Configuration for UniverseMachine data.
    cfg = config_um_data(cfg, verbose=verbose)
    um_data = load_um_data(cfg, verbose=verbose)

    return cfg, obs_data, um_data


def run_asap_model(args, verbose=True):
    """Run A.S.A.P model."""
    # Parse the configuration file.
    cfg = parse_config(args.config)

    # Load the data
    cfg, obs_data, um_data = initial_model(cfg, verbose=verbose)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config',
        help="Configuration file",
        default='asap_default_config.yaml')

    args = parser.parse_args()

    run_asap_model(args)
