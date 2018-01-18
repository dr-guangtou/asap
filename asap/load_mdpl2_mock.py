"""
"""
import numpy as np
from halotools.utils import crossmatch
from astropy.table import Table


def value_added_mdpl2_mock(fname):
    mdpl2_mock = Table(np.load(fname))

    mdpl2_mock.rename_column('m', 'mvir')
    mdpl2_mock.rename_column('mp', 'mpeak')
    mdpl2_mock.rename_column('id', 'halo_id')

    #  Apply periodic boundary conditions
    mdpl2_mock['x'] = np.where(mdpl2_mock['x'] < 0, 1000. + mdpl2_mock['x'],
                               mdpl2_mock['x'])
    mdpl2_mock['x'] = np.where(mdpl2_mock['x'] > 1000.,
                               mdpl2_mock['x'] - 1000., mdpl2_mock['x'])
    mdpl2_mock['y'] = np.where(mdpl2_mock['y'] < 0, 1000. + mdpl2_mock['y'],
                               mdpl2_mock['y'])
    mdpl2_mock['y'] = np.where(mdpl2_mock['y'] > 1000.,
                               mdpl2_mock['y'] - 1000., mdpl2_mock['y'])
    mdpl2_mock['z'] = np.where(mdpl2_mock['z'] < 0, 1000. + mdpl2_mock['z'],
                               mdpl2_mock['z'])
    mdpl2_mock['z'] = np.where(mdpl2_mock['z'] > 1000.,
                               mdpl2_mock['z'] - 1000., mdpl2_mock['z'])

    mdpl2_mock['halo_hostid'] = mdpl2_mock['halo_id']
    satmask = mdpl2_mock['upid'] != -1
    mdpl2_mock['halo_hostid'][satmask] = mdpl2_mock['upid'][satmask]

    idxA, idxB = crossmatch(mdpl2_mock['halo_hostid'], mdpl2_mock['halo_id'])
    mdpl2_mock['host_halo_mvir'] = mdpl2_mock['mvir']
    mdpl2_mock['host_halo_mvir'][idxA] = mdpl2_mock['mvir'][idxB]

    return mdpl2_mock


def value_added_smdpl_mock(smdpl_mock):
    """
    Value added the UniverseMachine model catalog for SMDPL simulation.

    This is designed for the short catalog, and the size of the SMDPL box
    is 400/h Mpc.
    """

    smdpl_mock.rename_column('m', 'mvir')
    smdpl_mock.rename_column('mp', 'mpeak')
    smdpl_mock.rename_column('id', 'halo_id')

    #  Apply periodic boundary conditions
    smdpl_mock['x'] = np.where(smdpl_mock['x'] < 0, 400. + smdpl_mock['x'],
                               smdpl_mock['x'])
    smdpl_mock['x'] = np.where(smdpl_mock['x'] > 400.,
                               smdpl_mock['x'] - 400., smdpl_mock['x'])
    smdpl_mock['y'] = np.where(smdpl_mock['y'] < 0, 400. + smdpl_mock['y'],
                               smdpl_mock['y'])
    smdpl_mock['y'] = np.where(smdpl_mock['y'] > 400.,
                               smdpl_mock['y'] - 400., smdpl_mock['y'])
    smdpl_mock['z'] = np.where(smdpl_mock['z'] < 0, 400. + smdpl_mock['z'],
                               smdpl_mock['z'])
    smdpl_mock['z'] = np.where(smdpl_mock['z'] > 400.,
                               smdpl_mock['z'] - 400., smdpl_mock['z'])

    smdpl_mock['halo_hostid'] = smdpl_mock['halo_id']
    satmask = smdpl_mock['upid'] != -1
    smdpl_mock['halo_hostid'][satmask] = smdpl_mock['upid'][satmask]

    idxA, idxB = crossmatch(smdpl_mock['halo_hostid'], smdpl_mock['halo_id'])
    smdpl_mock['host_halo_mvir'] = smdpl_mock['mvir']
    smdpl_mock['host_halo_mvir'][idxA] = smdpl_mock['mvir'][idxB]

    return smdpl_mock
