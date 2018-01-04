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
