#!/usr/bin/env python3

"""
Script that takes the output of the universe machine.

This output is a large text file, each
row being one object. We convert this into a numpy array and
1) throw away the columns we don't need
2) throw away central galaxies (and their satellites) smaller than a certain
    mass

The output is a .npz file with two keys, "centrals" and "satellites".
Satellites are sorted (ascending) by their parent ID and then by their total
stellar mass

Step 1 runs at ~25M rows per hour
Step 2 is very fast

We will not re-run steps if the intermediate files already exist
"""
import os
import argparse

from io import StringIO

import numpy as np

full_catalog_dtype = [
    # Halo stuff
    ("id", "int"),           # Unique halo ID
    ("descid", "int"),       # ID of descendant halo (or -1 at z=0).
    ("upid", "int"),         # Parent halo ID (or -1 if this is a central)
    ("flags", "int"),        # Ignore
    ("updist", "float64"),   # Ignore
    ("x", "float64"),
    ("y", "float64"),
    ("z", "float64"),        # halo position (comoving Mpc/h)
    ("vx", "float64"),
    ("vy", "float64"),
    ("vz", "float64"),       # halo velocity (physical peculiar km/s)
    ("m", "float64"),
    # Halo mass (Bryan & Norman 1998 virial mass, Msun)
    ("v", "float64"),        # Halo vmax (physical km/s)
    ("mp", "float64"),       # Halo peak historical mass (BN98 vir, Msun)
    ("vmp", "float64"),
    # VMP: Halo vmax at the time when peak mass was reached
    ("r", "float64"),        # Halo radius (BN98 vir, comoving kpc/h)
    ("rank1", "float64"),
    # halo rank in Delta_vmax (see UniverseMachine paper)
    ("rank2", "float64"),    # Ignore
    ("ra", "float64"),       # Ignore
    ("ra_rank", "float64"),  # Ignore
    # Stellar stuff
    ("sm", "float64"),       # True stellar mass (Msun)
    ("icl", "float64"),      # True intracluster stellar mass (Msun)
    ("sfr", "float64"),      # True star formation rate (Msun/yr)
    ("obs_sm", "float64"),
    # observed stellar mass, including random & systematic errors (Msun)
    ("obs_sfr", "float64"),
    # observed SFR, including random & systematic errors (Msun/yr)
    ("ssfr", "float64"),     # observed SSFR
    ("sm_hm", "float64"),    # SMHM ratio
    ("obs_uv", "float64"),   # Observed UV Magnitude (M_1500 AB)
]

reduced_catalog_dtype = [
    # Halo stuff
    ("id", "int"),           # Unique halo ID
    ("upid", "int"),         # Parent halo ID (or -1 if this is a central)
    ("x", "float64"),
    ("y", "float64"),
    ("z", "float64"),        # halo position (comoving Mpc/h)
    ("vx", "float64"),
    ("vy", "float64"),
    ("vz", "float64"),       # halo position (comoving Mpc/h)
    ("m", "float64"),
    # Halo mass (Bryan & Norman 1998 virial mass, Msun)
    ("mp", "float64"),       # Halo peak historical mass (BN98 vir, Msun)
    ("v", "float64"),        # Halo vmax (physical km/s)
    # Stellar stuff
    ("sm", "float64"),       # True stellar mass (Msun)
    ("icl", "float64"),      # True intracluster stellar mass (Msun)
    ("sfr", "float64"),      # True star formation rate (Msun/yr)
]


def main(data_file, mpeak_lim=11.0):
    """Update galaxy atalogs.

    With current data:
    # The intermediate data contains 68448380 centrals (80%)
    # The final data contains 385125 centrals (with a cut at 12)
      and 10809469 satellites (with no cut)
    """
    # Input catalog
    if (not os.path.isfile(data_file)) and (not os.path.islink(data_file)):
        raise IOError("# Can not find the UM results: %s" % data_file)

    um_pre, _ = os.path.splitext(data_file)

    # Reduce the number of colunms and save as a numpy array
    inter_file = um_pre + '_reduced_cols.npy'
    print("# The catalog with reduced columns will be saved as: %s" % inter_file)

    # Remove small parent halos and satellites associated with these
    # small things
    final_file = um_pre + "_logmp_%4.1f.npz" % mpeak_lim
    print("# The final catalog after the Mpeak cut will be saved as: %s" % final_file)

    # If we have already generated the inter_file, don't do it again...
    blockSize = 100000

    if not os.path.isfile(inter_file):
        reduced_catalog = np.zeros(blockSize, dtype=reduced_catalog_dtype)
        with open(data_file) as f:
            count = 0
            for line in f:
                if line.startswith("#"):
                    continue
                row = np.loadtxt(StringIO(line), dtype=full_catalog_dtype)
                reduced_catalog[count] = row[
                    list(reduced_catalog.dtype.names)].copy()
                count += 1
                if count % blockSize == 0:
                    reduced_catalog.resize(len(reduced_catalog) + blockSize)
                    print(count)
        np.save(inter_file, reduced_catalog[:count])
        del reduced_catalog
    else:
        print("# Skipping reducing cols, file already exists")

    if not os.path.isfile(final_file):
        reduced_catalog = np.load(inter_file)

        # All centrals greater than a given mass
        central_catalog = np.sort(
            reduced_catalog[
                (reduced_catalog["upid"] == -1) &
                (reduced_catalog["mp"] > (10.0 ** mpeak_lim))],
            order="id")
        central_ids = frozenset(central_catalog["id"])

        # Remove all halos not associated with one of those centrals
        satellite_catalog = reduced_catalog[
            np.array(
                [parent in central_ids for parent in reduced_catalog["upid"]]
                )]
        np.savez(final_file,
                 centrals=central_catalog,
                 satellites=satellite_catalog)

        del reduced_catalog, central_catalog, central_ids, satellite_catalog
    else:
        print("# File already exists !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'um_file', type=str,
        help=('UniverseMachine snapshot file in .txt format'))

    parser.add_argument(
        '-m', '--mpeak', dest='mpeak_lim',
        help='Lower limit of Mpeak',
        type=float, default=11.0)

    args = parser.parse_args()

    main(args.um_file, mpeak_lim=args.mpeak_lim)
