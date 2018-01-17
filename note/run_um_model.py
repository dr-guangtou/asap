import os
import sys
import copy
from time import time

import numpy as np
import emcee
print(emcee.__version__)

from um_ins_exs_model import InsituExsituModel

um_test = InsituExsituModel(obs_dir='../data/s16a_massive_wide2/',
                            model_type='frac1',
                            um_mtot_nbin=100,
                            um_min_nobj_per_bin=15,
                            mcmc_wl_weight=1.0,
                            mcmc_nsamples=60,
                            mcmc_nburnin=30,
                            mcmc_nwalkers=128)

um_test.mcmcFit(multi=True)
