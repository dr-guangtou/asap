import numpy as np
from data import cluster_sum

import pytest

class TestCylinderSMAndRichness():
    def test_colliding_centrals(self):
        centrals = np.zeros(2, dtype=gal_dtype)
        satellites = np.zeros(3, dtype=gal_dtype)

        # remeber rvir is in Kpc and almost everything else is in Mpc
        centrals[0] = _create_gal(**{"id": 1, "upid": -1, "sm": 1e11, "icl": 2e11, "x": 100, "y": 100, "z": 100, "rvir": 1000})
        centrals[1] = _create_gal(**{"id": 10, "upid": -1, "sm": 5e11, "icl": 2e11, "x": 100, "y": 100, "z": 100, "rvir": 1000})

        satellites[0] = _create_gal(**{"id": 2, "upid": 1, "sm": 2e10, "icl": 1e10, "x": 100, "y": 100, "z": 100, "rvir": 1000})
        satellites[1] = _create_gal(**{"id": 3, "upid": 1, "sm": 1e10, "icl": 3e10, "x": 100, "y": 100, "z": 100, "rvir": 1000})
        satellites[2] = _create_gal(**{"id": 4, "upid": 7, "sm": 2e10, "icl": 3e10, "x": 100, "y": 100, "z": 100, "rvir": 1000})

        z_err = 10

        # With tot
        new_centrals, counts = cluster_sum.get_cylinder_mass_and_richness(centrals, satellites, 0, np.inf, 0.9999, z_err)
        assert len(new_centrals) == 1 and len(counts) == 1
        assert new_centrals["id"] == 10
        assert new_centrals["sm"] == 5e11
        assert new_centrals["icl"] == 2e11 + 1e11 + 2e11 + np.sum(satellites["sm"] + satellites["icl"]) # all mass is in the cylinder
        assert counts == 5

    def test_get_spec_z_mass_and_richness(self):
        centrals = np.zeros(1, dtype=gal_dtype)
        satellites = np.zeros(3, dtype=gal_dtype)

        # remeber rvir is in Kpc and almost everything else is in Mpc
        centrals[0] = _create_gal(**{"id": 1, "upid": -1, "sm": 1e11, "icl": 2e11, "x": 100, "y": 100, "z": 100, "rvir": 1000})
        satellites[0] = _create_gal(**{"id": 2, "upid": 1, "sm": 2e10, "icl": 1e10, "x": 100, "y": 100, "z": 100, "rvir": 1000})
        satellites[1] = _create_gal(**{"id": 3, "upid": 1, "sm": 1e10, "icl": 3e10, "x": 100, "y": 100, "z": 100, "rvir": 1000})
        satellites[2] = _create_gal(**{"id": 4, "upid": 7, "sm": 2e10, "icl": 3e10, "x": 100, "y": 100, "z": 100, "rvir": 1000})

        z_err = 10

        # With tot
        new_centrals, counts = cluster_sum.get_cylinder_mass_and_richness(centrals, satellites, 0, np.inf, 0.9999, z_err)
        assert len(new_centrals) == 1 and len(counts) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11
        assert new_centrals["icl"] == 2e11 + np.sum(satellites["sm"] + satellites["icl"]) # all mass is in the cylinder
        assert counts == 4

        # With 1
        new_centrals, counts = cluster_sum.get_cylinder_mass_and_richness(centrals, satellites, 0, np.inf, 1, z_err)
        assert len(new_centrals) == 1 and len(counts) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11
        assert new_centrals["icl"] == 2e11 + 3e10 + 2e10
        assert counts == 2


        # With 1, but moving the one that we had in just out of the cylinder
        satellites[2]["z"] = 111
        new_centrals, counts = cluster_sum.get_cylinder_mass_and_richness(centrals, satellites, 0, np.inf, 1, z_err)
        assert len(new_centrals) == 1 and len(counts) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11
        assert new_centrals["icl"] == 2e11 + 3e10 + 1e10
        assert counts == 2


class TestCentralsWithSatellites():
    def test_basic(self):
        centrals = np.zeros(1, dtype=gal_dtype)
        satellites = np.zeros(3, dtype=gal_dtype)

        centrals[0] = _create_gal(**{"id": 1, "upid": -1, "sm": 1e11, "icl": 2e11})
        satellites[0] = _create_gal(**{"id": 2, "upid": 1, "sm": 2e10, "icl": 1e10})
        satellites[1] = _create_gal(**{"id": 3, "upid": 1, "sm": 1e10, "icl": 3e10})
        satellites[2] = _create_gal(**{"id": 4, "upid": 7, "sm": 1e10, "icl": 3e10})

        # With 1
        new_centrals = cluster_sum.centrals_with_satellites(centrals, satellites, 1, False, np.nan, np.nan)
        assert len(new_centrals) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11 + 1e10
        assert new_centrals["icl"] == 2e11 + 3e10

        # With 2
        new_centrals = cluster_sum.centrals_with_satellites(centrals, satellites, 2, False, np.nan, np.nan)
        assert len(new_centrals) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11 + 1e10 + 2e10
        assert new_centrals["icl"] == 2e11 + 3e10 + 1e10

        # With 0.9999
        new_centrals = cluster_sum.centrals_with_satellites(centrals, satellites, 0.999, False, np.nan, np.nan)
        assert len(new_centrals) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11 + 1e10 + 2e10
        assert new_centrals["icl"] == 2e11 + 3e10 + 1e10

    def test_with_mass_cut(self):
        centrals = np.zeros(1, dtype=gal_dtype)
        satellites = np.zeros(3, dtype=gal_dtype)

        centrals[0] = _create_gal(**{"id": 1, "upid": -1, "sm": 1e11, "icl": 2e11})
        satellites[0] = _create_gal(**{"id": 2, "upid": 1, "sm": 2e10, "icl": 1e10})
        satellites[1] = _create_gal(**{"id": 3, "upid": 1, "sm": 1e10, "icl": 3e10})
        satellites[2] = _create_gal(**{"id": 4, "upid": 7, "sm": 1e10, "icl": 3e10})

        # With 2
        new_centrals = cluster_sum.centrals_with_satellites(centrals, satellites, 2, True, 3e10 + 1, np.inf)
        assert len(new_centrals) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11 + 1e10
        assert new_centrals["icl"] == 2e11 + 3e10

    def test_with_ssfr_cut(self):
        centrals = np.zeros(1, dtype=gal_dtype)
        satellites = np.zeros(3, dtype=gal_dtype)

        centrals[0] = _create_gal(**{"id": 1, "upid": -1, "sm": 1e11, "icl": 2e11})
        satellites[0] = _create_gal(**{"id": 2, "upid": 1, "sm": 2e10, "icl": 1e10, "ssfr": 0})
        satellites[1] = _create_gal(**{"id": 3, "upid": 1, "sm": 1e10, "icl": 3e10, "ssfr": 1})
        satellites[2] = _create_gal(**{"id": 4, "upid": 7, "sm": 1e10, "icl": 3e10})

        # With 2
        new_centrals = cluster_sum.centrals_with_satellites(centrals, satellites, 2, True, 0, 0.1)
        assert len(new_centrals) == 1
        assert new_centrals["id"] == 1
        assert new_centrals["sm"] == 1e11 + 2e10
        assert new_centrals["icl"] == 2e11 + 1e10


class TestGetN():
    def test_get_n_int(self):
        assert cluster_sum.get_n(0, 100) == 0
        assert cluster_sum.get_n(100, 100) == 100
        assert cluster_sum.get_n(200, 100) == 100

    def test_get_n_fraction(self):
        assert cluster_sum.get_n(0.1, 100) == 10
        assert cluster_sum.get_n(0.9999, 100) == 100

    def test_get_n_err(self):
        with pytest.raises(Exception) as exc:
            cluster_sum.get_n(-0.1, 100)
        assert "n must be > 0" in str(exc)

class TestAddUncertaintyToSats():
    def test_add_unc_to_sats(self):
        centrals = np.zeros(1, dtype=gal_dtype)
        satellites = np.zeros(100, dtype=gal_dtype)

        centrals[0] = _create_gal(**{"id": 999, "upid": -1, "sm": 1e11, "icl": 2e11})
        for i in range(len(satellites)):
            satellites[i] = _create_gal(**{"id": i, "upid": 1, "sm": 2e10, "icl": 1e10})

        centrals_ht, big_enough_gals_ht, big_enough_gals, _ = cluster_sum.cut_and_rsd(
                centrals, satellites, 0, np.inf)

        assert len(centrals_ht) == 1
        assert len(big_enough_gals_ht) == 1 + len(satellites)

        big_enough_gals_ht = cluster_sum.add_uncertainty_to_sats(big_enough_gals_ht, big_enough_gals, 20)
        assert big_enough_gals_ht[0,2] == 0
        sats_z = big_enough_gals_ht[1:,2]
        assert np.all(sats_z != 0)
        assert np.max(sats_z) < 400 and np.min(sats_z) > 0



### Helpers

gal_dtype = [
        ('id', '<i8'),
        ('upid', '<i8'),
        ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
        ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
        ('m', '<f8'),
        ('mp', '<f8'),
        ('sm', '<f8'),
        ('icl', '<f8'),
        ('sfr', '<f8'),
        ('ssfr', '<f8'),
        ('rvir', '<f8'),
        #('pid', '<f8'), ('mvir', '<f8'), , ('rs', '<f8'), ('Halfmass_Scale', '<f8'), ('scale_of_last_MM', '<f8'), ('M200b', '<f8'), ('M200c', '<f8'), ('Acc_Rate_Inst', '<f8'), ('Acc_Rate_100Myr', '<f8'), ('Acc_Rate_1*Tdyn', '<f8'), ('Acc_Rate_2*Tdyn', '<f8'), ('Acc_Rate_Mpeak', '<f8'), ('Vmax@Mpeak', '<f8')])]
]
def _create_gal(**kwargs):
    gal = np.zeros(1, dtype=gal_dtype)
    for k, v in kwargs.items():
        gal[k] = v
    return gal
