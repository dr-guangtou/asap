"""Model fitting using emcee sampler."""
from __future__ import print_function, division, unicode_literals

import numpy as np

import emcee
EMCEE_VERSION = emcee.__version__.split('.')[0]

from . import io

__all__ = []


def setup_moves(cfg_emcee, burnin=False):
    """Choose the Move object for emcee.

    Parameters
    ----------
    cfg_emcee : dict
        Configuration parameters for emcee
    burnin : bool, optional
        Whether this is for burnin

    Return
    ------
    emcee_moves : emcee.moves object
        Move object for emcee walkers.

    """
    move_col = 'moves' if not burnin else 'moves_burnin'

    if cfg_emcee[move_col] == 'snooker':
        emcee_moves = emcee.moves.DESnookerMove()
    elif cfg_emcee[move_col] == 'stretch':
        emcee_moves = emcee.moves.StretchMove(a=cfg_emcee['stretch_a'])
    elif cfg_emcee[move_col] == 'walk':
        emcee_moves = emcee.moves.WalkMove(s=cfg_emcee['walk_s'])
    elif cfg_emcee[move_col] == 'kde':
        emcee_moves = emcee.moves.KDEMove()
    elif cfg_emcee[move_col] == 'de':
        emcee_moves = emcee.moves.DEMove(cfg_emcee['de_sigma'])
    else:
        raise Exception("Wrong option: stretch, walk, kde, de, snooker")

    return emcee_moves
