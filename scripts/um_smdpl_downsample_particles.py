#!/usr/bin/env python
"""
This script will read the dark matter particle table for the SMDPL
simulation, and downsample it for our model.

We will create two versions:
1. default one will have 10 million particles.
2. another one with 50 million particles for test.
"""

import os
import pandas as pd
import numpy as np

# Input table
um_dir = '/Users/song/astro5/massive/dr16a/um2/um2_new'
um_smdpl_dir = os.path.join(um_dir, 'um_smdpl')
um_smdpl_ptbl = os.path.join(um_smdpl_dir, 'particles_0.712400.txt')

# Output files
n_ds1 = 1E7
n_ds2 = 5E7
um_smdpl_ptbl_out1 = os.path.join(um_dir, 'um_smdpl_particles_0.7124_10m.npy')
um_smdpl_ptbl_out2 = os.path.join(um_dir, 'um_smdpl_particles_0.7124_50m.npy')

# Data format for output
particle_table_dtype = [
        ("x", "float64"), ("y", "float64"), ("z", "float64")
        # halo position (comoving Mpc/h)
]

# Read the data
chunksize = 1000000
um_smdpl_pchunks = pd.read_csv(um_smdpl_ptbl, usecols=[0, 1, 2],
                               delim_whitespace=True,
                               names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'id'],
                               dtype=particle_table_dtype,
                               index_col=False,
                               chunksize=chunksize)
um_smdpl_pdframe = pd.concat(um_smdpl_pchunks)
um_smdpl_parray = um_smdpl_pdframe.values.ravel().view(
    dtype=particle_table_dtype
)

# Downsample
np.random.seed(2018)
um_smdpl_ptl_ds1 = np.random.choice(um_smdpl_parray, int(n_ds1),
                                    replace=False)
um_smdpl_ptl_ds2 = np.random.choice(um_smdpl_parray, int(n_ds2),
                                    replace=False)

np.save(um_smdpl_ptbl_out1, um_smdpl_ptl_ds1)
np.save(um_smdpl_ptbl_out2, um_smdpl_ptl_ds2)
