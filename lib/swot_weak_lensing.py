"""
Functions for weak lensing test.
"""

from __future__ import \
    division, \
    print_function, \
    absolute_import

import os
import copy
import warnings
import numpy as np

# Matplotlib related
import matplotlib as mpl
import matplotlib.pyplot as plt

# astropy
from astropy.table import Table


class SwotWL(object):
    """
    Object for Swot Galaxy-Galaxy Lensing result file.
    """

    def __init__(self, wlFile):
        """
        Load the result file.
        """
        msg = '# Can not find the Swot result file !'
        assert os.path.isfile(wlFile), msg

        # Check the first line
        wlRead = open(wlFile, 'r')
        line1 = wlRead.readline()
        msg = '# Not a Swot Galaxy-Galaxy Lensing Result File !'
        assert 'Gal-gal lensing' in line1, msg
        self.line1 = line1

        # Second line
        line2 = wlRead.readline()
        self.line2 = line2
        self.resampling_method = line2.split()[2]
        # Get the number of resampling method
        line2 = line2.replace('(', '')
        line2 = line2.replace(')', '')
        self.resampling_n = int([ss for ss in line2.split()
                                 if ss.isdigit()][0])

        # Third line
        line3 = wlRead.readline()
        self.line3 = line3
        # Get the coordinate type
        self.coord_type = line3.split()[2]

        # Fourth line
        line4 = wlRead.readline()
        self.line4 = line4
        line4 = line4.replace(':', ' ')
        line4 = line4.replace('#', ' ')
        line4 = line4.replace(',', ' ')
        line4 = line4.replace('=', ' ')
        # Get cosmology
        self.H0 = float(line4.split()[2])
        self.h = self.H0 / 100.0
        self.Omega_M = float(line4.split()[4])
        self.Omega_L = float(line4.split()[6])

        # Load the catalog
        self.table = Table.read(wlFile, format='ascii')
        self.table.rename_column('col1', 'r')
        self.table.rename_column('col2', 'sigma')
        self.table.rename_column('col3', 'err_w')
        self.table.rename_column('col4', 'err_s')
        self.table.rename_column('col5', 'n_sources')
        self.table.rename_column('col6', 'r_mean')
        self.table.rename_column('col7', 'e2')

        # Organize the result
        self.r = np.asarray(self.table['r'])
        self.sig = np.asarray(self.table['sigma'])
        self.err_w = np.asarray(self.table['err_w'])
        self.err_s = np.asarray(self.table['err_s'])
        self.n_sources = np.asarray(self.table['n_sources'])
        self.r_mean = np.asarray(self.table['r_mean'])
        self.e2 = np.asarray(self.table['e2'])

        #
        self.r_sig = self.r * self.sig
        self.r_err_w = self.r * self.err_w
        self.r_err_s = self.r * self.err_s
