"""Model fitting using emcee sampler."""

import numpy as np

try:
    import emcee
    EMCEE_VERSION = emcee.__version__.split('.')[0]
except(ImportError):
    pass