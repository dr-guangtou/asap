#!/usr/bin/env python3
"""Script that takes universe machine hlist (halo list) that is the output of
reduce_um_catalog_data_size, and merges some info from it with the info we
have from the halo catalog. We do this in two steps.
1) Convert the halo catalog to a numpy array and throw away unneeded cols.
2) Join onto the catalog data using the ID.

We do not have data in the hlist for satellites with really long IDs (>=1000008162600955). In
there cases we have nans in the data set.

TODO: Work out why we don't have the data for some of the halos.
"""
import os
import argparse

import numpy as np
import pandas as pd

import halotools.sim_manager

HLIST_COLUMNS = [
    # Serve as a reference
    ("scale", "float64"), # 1: Scale factor at this snapshot
    ("id", "int"), # 2: id of halo
    ("desc_scale", "float64"), # 3:Scale factor of descendent halo
    ("desc_id", "int"), # 4: ID of descendant halo (or -1 at z=0)
    ("num_prog", "int"), # 5: Number of progenitors that went into this halo
    ("pid", "int"), # 6: least massive parent (direct parent) halo ID (or -1 if this is a central)
    ("upid", "int"), # 7: most massive parent (ultimate parent) halo ID (or -1 if this is a central)
    ("desc_pid", "int"), # 8: pid of the descendant (or -1)
    ("phantom", "int"), # 9: nonzero if halo interpolated across timesteps
    # Physical properties of the halo
    ("sam_mvir", "float64"), # 10: Smoothed halo mass used in SAMs
    ("mvir_rock", "float64"), # 11: Msun/h
    ("rvir", "float64"), # 12: kpc/h
    ("rs", "float64"), # 13: scale radius kpc/h
    ("vrms", "float64"), # 14: v rms km/s (physical)
    ("flag_mmp", "bool"), # 15: whether the halo is the most massive progenitor
    ("scale_lmm", "float64"), # 16: scale factor at last MM
    ("vmax", "float64"), # 17 Vmax (km.s)
    ("x", "float64"), ("y", "float64"), ("z", "float64"),
    # 18, 19, 20: halo position (comoving Mpc/h)
    ("vx", "float64"), ("vy", "float64"), ("vz", "float64"),
    # 21, 22, 23: halo velocity (physical peculiar km/s)
    ("jx", "float64"), ("jy", "float64"), ("jz", "float64"),
    # 24, 25, 26: halo angular momentum
    ("spin", "float64"), # 27: cbx check if int
    # Various things for merger trees
    ("breadth_first_id", "int"), # 28
    ("depth_first_id", "int"), # 29
    ("tree_root_id", "int"), # 30
    ("orig_halo_id", "int"), # 31
    ("snap_num", "int"), # 32
    ("next_coprogenitor_depthfirst_id", "int"), # 33
    ("last_progenitor_depthfirst_id", "int"), # 34
    ("last_mainleaf_depthfirst_id", "int"), # 35
    # More physical properties but more arcane
    ("rs_klypin", "float64"), # 36: scale radius from vmax and mvir (see Rockstar)
    ("mmvir_all", "float64"), # 37: mass enclosed in specified overdensity
    ("m200b", "float64"), # 38
    ("m200c", "float64"), # 39
    ("m500c", "float64"), # 40
    ("m2500c", "float64"), # 41
    ("x_off", "float64"), # 42: offset of density peak form average particle position
    ("v_off", "float64"), # 43: offset of density peak form average velocity position
    ("spin_bullock", "float64"), # 44: Bullock spin parameter
    # Shape stuff
    ("b_to_a", "float64"), # 45: Ratio of second largest axis to largest (< 1)
    ("c_to_a", "float64"), # 46: Ratio of third largest axis to largest (< 1)
    ("a_x", "float64"), ("a_y", "float64"), ("a_z", "float64"), # 47, 48, 49
    ("b_to_a_500c", "float64"), # 50
    ("c_to_a_500c", "float64"), # 51
    ("a_x_500c", "float64"), ("a_y_500c", "float64"), ("a_z_500c", "float64"),
    # 52, 53, 54 : a_x_500c, a_y_500c, a_z_500c
    ("t_over_u", "float64"), # 55: ratio of kinetic to potential energy
    ("m_pe_behroozi", "float64"), # 56: pseudo evolution corrected masses (experimental)
    ("m_pe_diemer", "float64"), # 57: pseudo evolution corrected masses (experimental)
    ("macc", "float64"), # 58: Mass at accretion
    ("mpeak", "float64"), # 59: peak mass (cbx is this over the whole time? or up to this point)
    ("vacc", "float64"), # 60: Velocity at accretion
    ("vpeak", "float64"), # 61: peak vmax
    ("scale_half_mass", "float64"), # 62: scale factor at which we could to 0.5 * mpeak
    # Acc rate
    ("acc_inst", "float64"), # 63
    ("acc_100myr", "float64"), # 64
    ("acc_1tdyn", "float64"), # 65
    ("acc_2tdyn", "float64"), # 66
    ("acc_mpeak", "float64"), # 67
    ("scale_mpeak", "float64"), # 68: scale factor at which we reached mpeak
    ("scale_acc", "float64"), # 69: scale factor at which we last accreted a sattelite
    ("scale_first_acc", "float64"),
    # 70: Scale at which current and former satellites first passed through a larger halo.
    ("mvir_first_acc", "float64"), # 71: mvir at first_acc_scale
    ("vmax_first_acc", "float64"), # 72: vmax at first_acc_scale
    ("vmax_mpeak", "float64"), # 73: vmax at the scale where mpeak was reached
]

hlist_selected = {
    "id": (1, "int32"),
    "pid": (5, "int32"),
    "mvir_rock": (10, "float64"), # There will be a mvir on the UM side
    "rvir": (11, "float64"),
    "rs": (12, "float64"),
    "scale_half_mass": (61, "float64"),
    "scale_lmm": (15, "float64"),
    "m200b": (37, "float64"),
    "m200c": (38, "float64"),
    "m500c": (39, "float64"),
    "acc_inst": (62, "float64"),
    "acc_100myr": (63, "float64"),
    "acc_1tdyn": (64, "float64"),
    "acc_2tdyn": (65, "float64"),
    "acc_mpeak": (66, "float64"),
    "vmax": (16, "float64"),
    "vmax_first_acc": (71, "float64"),
    "vmax_mpeak": (72, "float64"),
    "b_to_a": (44, "float64"),
    "c_to_a": (45, "float64"),
    "b_to_a_500c": (49, "float64"),
    "c_to_a_500c": (50, "float64"),
}

def main(hlist_file, um_file):
    """Add additional halo properties to the Universe Machine catalog."""
    if not os.path.isfile(hlist_file):
        raise IOError("# Can not find the halo catalog : %s" % hlist_file)
    hlist_pre, _ = os.path.splitext(hlist_file)

    # Reduce the number of colunms and save as a numpy array
    inter_file = hlist_pre + "_reduced_cols.npy"

    # Current UM catalog
    if not os.path.isfile(um_file):
        raise IOError("# Can not find the UMachine catalog : %s" % um_file)
    um_pre, _ = os.path.splitext(um_file)

    # Extended UM catalog
    um_extended_file = um_pre + "_extended.npz"

    # If we have already generated the inter_file, don't do it again...
    if not os.path.isfile(inter_file):
        hlist_reader = halotools.sim_manager.TabularAsciiReader(hlist_file, hlist_selected)
        reduced_catalog = hlist_reader.read_ascii()
        np.save(inter_file, reduced_catalog)
        del reduced_catalog
    else:
        print("Skipping reducing cols, file already exists")

    if not os.path.isfile(um_extended_file):
        # Load current catalog
        current = np.load(um_file)
        current_df = pd.concat([
            pd.DataFrame(current["centrals"]),
            pd.DataFrame(current["satellites"]),
        ]).set_index("id")
        del current

        # Load the the extensions we just created and join on id
        extension_df = pd.DataFrame(np.load(inter_file)).set_index("id")
        result_df = current_df.join(extension_df, how="left")
        print(len(result_df), len(current_df))
        del current_df, extension_df

        # Save out result converting back to structured numpy array
        result = result_df.to_records()
        result = result.view(result.dtype.descr, np.ndarray)
        del result_df
        np.savez(
            um_extended_file,
            centrals=result[result["upid"] == -1],
            satellites=result[result["upid"] != -1])
    else:
        print("Skipping combining data, file already exists")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'hlist_file', type=str,
        help=('The DM only simulation halo catalog in .txt format'))

    parser.add_argument(
        'um_file', type=str,
        help=('UniverseMachine snapshot file in .txt format'))

    args = parser.parse_args()

    main(args.hlist_file, args.um_file)
