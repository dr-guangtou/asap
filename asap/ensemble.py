"""Model fitting using emcee sampler."""
from __future__ import print_function, division, unicode_literals

import numpy as np

try:
    import emcee
    EMCEE_VERSION = emcee.__version__.split('.')[0]
except(ImportError):
    pass

from . import io

__all__ = []