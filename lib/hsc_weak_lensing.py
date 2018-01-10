"""
Functions for HSC weak lensing.
"""

from __future__ import \
    division, \
    print_function, \
    absolute_import

import os
import copy
import warnings
import numpy as np

# astropy
from astropy.table import Table


class HscDsigma(object):
    """Object for HSC Galaxy-Galaxy Lensing result.

    Using the deltaSigma pipeline.

    """

    def __init__(self, dsigma_output):
        """Load the result file."""
        msg = '# Can not find the Swot result file !'
        assert os.path.isfile(dsigma_output), msg

        # Read the file
        data = np.load(dsigma_output)
        dsigma = data['delta_sigma']
        cosmology = data['cosmology']
        config = data['config'].item()

        # sampling method
        self.resampling_method = 'jackknife'
        self.resampling_n = config['njackknife_fields']

        # cosmology paraeter
        self.H0 = cosmology['H0'][0]
        self.h = self.H0 / 100.0
        self.Omega_M = cosmology['omega_m'][0]
        self.Omega_L = cosmology['omega_l'][0]
        self.comoving = config['comoving']

        # Load the catalog
        self.table = Table(dsigma)

        # Organize the result
        self.r = dsigma['r_mpc']
        self.sig = dsigma['dsigma_lr']
        self.err_w = dsigma['dsigma_err_1']
        self.err_s = dsigma['dsigma_err_jk']
        self.n_sources = dsigma['lens_npairs']

        # For test
        self.r_sig = self.r * self.sig
        self.r_err_w = self.r * self.err_w
        self.r_err_s = self.r * self.err_s


class HscBoxbin(HscDsigma):
    """
    Class for HSC weak lensing profile within a box defined by
    Mtot and Minn.
    """

    def setBinId(self, bin_id):
        """Set the bin id for the box."""
        self.bin_id = bin_id

    def setMassLimits(self,
                      low_mtot, upp_mtot,
                      low_minn, upp_minn):
        """Set the lower and upper mass limits for both Mtot and Minn."""
        self.low_mtot = low_mtot
        self.upp_mtot = upp_mtot

        self.low_minn = low_minn
        self.upp_minn = upp_minn
