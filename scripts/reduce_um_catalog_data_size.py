#!/usr/bin/env python3
"""
Script that takes the output of the universe machine. This output is a large text file, each
row being one object. We convert this into a numpy array and
1) throw away the columns we don't need
2) throw away central galaxies (and their satellites) smaller than a certain mass
The output is a .npz file with two keys, "centrals" and "satellites". Satellites
are sorted (ascending) by their parent ID and then by their total stellar mass

Step 1 runs at ~25M rows per hour
Step 2 is very fast

We will not re-run steps if the intermediate files already exist
"""
import numpy as np
import numpy.lib.recfunctions as np_rfn
import os
import helpers

full_catalog_dtype = [
        # Halo stuff
        ("id", "int"), # Unique halo ID
        ("descid", "int"), # ID of descendant halo (or -1 at z=0)
        ("upid", "int"), # Parent halo ID (or -1 if this is a central)
        ("flags", "int"), # Ignore
        ("updist", "float64"), # Ignore
        ("x", "float64"), ("y", "float64"), ("z", "float64"), # halo position (comoving Mpc/h)
        ("vx", "float64"), ("vy", "float64"), ("vz", "float64"), # halo velocity (physical peculiar km/s)
        ("m", "float64"), # Halo mass (Bryan & Norman 1998 virial mass, Msun)
        ("v", "float64"), # Halo vmax (physical km/s)
        ("mp", "float64"), # Halo peak historical mass (BN98 vir, Msun)
        ("vmp", "float64"), # VMP: Halo vmax at the time when peak mass was reached
        ("r", "float64"), # Halo radius (BN98 vir, comoving kpc/h)
        ("rank1", "float64"), # halo rank in Delta_vmax (see UniverseMachine paper)
        ("rank2", "float64"), # Ignore
        ("ra", "float64"), # Ignore
        ("ra_rank", "float64"), # Ignore

        # Stellar stuff
        ("sm", "float64"), # True stellar mass (Msun) (This is mass in stars)
        ("icl", "float64"), # True intracluster stellar mass (Msun) (This is mass in gas/dust)
        ("sfr", "float64"), # True star formation rate (Msun/yr)
        ("obs_sm", "float64"), # observed stellar mass, including random & systematic errors (Msun)
        ("obs_sfr", "float64"), # observed SFR, including random & systematic errors (Msun/yr)
        ("ssfr", "float64"), # observed SSFR
        ("sm_hm", "float64"), # SMHM ratio
        ("obs_uv", "float64"), # Observed UV Magnitude (M_1500 AB)
] # yapf: disable

reduced_catalog_dtype = [
        # Halo stuff
        ("id", "int"), # Unique halo ID
        ("upid", "int"), # Parent halo ID (or -1 if this is a central)
        ("x", "float64"), ("y", "float64"), ("z", "float64"), # halo position (comoving Mpc/h)
        ("m", "float64"), # Halo mass (Bryan & Norman 1998 virial mass, Msun)
        ("mp", "float64"), # Halo peak historical mass (BN98 vir, Msun)

        # Stellar stuff
        ("sm", "float64"), # True stellar mass (Msun) (This is mass in stars)
        ("icl", "float64"), # True intracluster stellar mass (Msun) (This is mass in gas/dust)
        ("sfr", "float64"), # True star formation rate (Msun/yr)
] # yapf: disable

data_dir = "/home/christopher/Data/data/universe_machine/"


def main():
    # With current data:
    # The intermediate data contains 68448380 centrals (80%)
    # The final data contains 385125 centrals (with a cut at 12) and 10809469 satellites (with no cut)
    data_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400.txt"
    # Reduce the number of colunms and save as a numpy array
    inter_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400_reduced_cols.npy"
    # Remove small parent halos and satellites associated with these small things
    final_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400_final.npz"

    # If we have already generated the inter_file, don't do it again...
    if not os.path.isfile(inter_file):
        reduced_catalog = helpers.reduce_cols(data_file, full_catalog_dtype, reduced_catalog_dtype)
        np.save(inter_file, reduced_catalog)
        del reduced_catalog
    else:
        print("Skipping reducing cols, file already exists")

    if not os.path.isfile(final_file):
        reduced_catalog = np.load(inter_file)
        # All centrals greater than a given mass
        central_catalog = np.sort(
            reduced_catalog[(reduced_catalog["upid"] == -1)
                            & (reduced_catalog["mp"] > 1e12)],
            order="id")
        central_ids = frozenset(central_catalog["id"])
        # Remove all halos not associated with one of those centrals
        satellite_catalog = reduced_catalog[np.array(
            [parent in central_ids for parent in reduced_catalog["upid"]])]
        # Add a field for the total stellar mass to the satellites and sort
        original_cols = satellite_catalog.dtype.names
        satellite_catalog = np_rfn.append_fields(
            satellite_catalog,
            "total_stellar_mass",
            satellite_catalog["sm"] + satellite_catalog["icl"],
            usemask=False)
        satellite_catalog = np.sort(satellite_catalog, order=["upid", "total_stellar_mass", "id"])
        # Remove the total stellar mass field and save the catalog
        satellite_catalog = satellite_catalog[list(original_cols)]
        np.savez(final_file, centrals=central_catalog, satellites=satellite_catalog)
        del reduced_catalog, central_catalog, central_ids, satellite_catalog
    else:
        print("Skipping removing small parent halos and their satellites, file already exists")


if __name__ == "__main__":
    main()
