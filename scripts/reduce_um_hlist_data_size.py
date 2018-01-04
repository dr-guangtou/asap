#!/usr/bin/env python3
"""
Script that takes universe machine hlist (halo list) and merges some info from it with
the info we have from the catalog. We do this in two steps.
1) Convert to a numpy array and throw away unneeded cols.
2) Join onto the catalog data using the ID.

We do not have data in the hlist for satellites with really long IDs (>=1000008162600955). In
there cases we have nans in the data set.

TODO: Work out why we don't have the data for some of the halos.
"""
import numpy as np
import pandas as pd
import os
import helpers

reduced_hlist_dtype = [
    ("id", "int"), # id of halo
    ("mvir", "float64"), # Msun/h
    ("rvir", "float64"), # kpc/h
    ("rs", "float64"), # scale radius kpc/h
    ("Halfmass_Scale", "float64"), # scale factor at which we could to 0.5 * mpeak
    ("scale_of_last_MM", "float64"), # scale factor at last MM
    ("M200b", "float64"),
    ("M200c", "float64"),
    ("Acc_Rate_Inst", "float64"),
    ("Acc_Rate_100Myr", "float64"),
    ("Acc_Rate_1*Tdyn", "float64"),
    ("Acc_Rate_2*Tdyn", "float64"),
    ("Acc_Rate_Mpeak", "float64"),
    ("Vmax@Mpeak", "float64"), # vmax at the scale where mpeak was reached
]

full_hlist_dtype = [
    ("scale", "float64"), # Scale factor at this snapshot
    ("id", "int"), # id of halo
    ("desc_scale", "float64"), # Scale factor of descendent halo (which snapshot this halo next appears in)
    ("desc_id", "int"), # ID of descendant halo (or -1 at z=0)
    ("num_prog", "int"), # Number of progenitors that went into this halo
    ("pid", "int"), # least massive parent (direct parent) halo ID (or -1 if this is a central)
    ("upid", "int"), # most massive parent (ultimate parent) halo ID (or -1 if this is a central)
    ("desc_pid", "int"), # pid of the descendant (or -1)
    ("phantom", "int"), # nonzero if halo interpolated across timesteps (cbx what does this mean and is this an int?)

    # Physical properties of the halo
    ("sam_mvir", "float64"), # Smoothed halo mass used in SAMs (cbx always greater than sum of halo masses of contributing progenitors (Msun/h))
    ("mvir", "float64"), # Msun/h
    ("rvir", "float64"), # kpc/h
    ("rs", "float64"), # scale radius kpc/h
    ("vrms", "float64"), # v rms km/s (physical)
    ("mmp?", "bool"), # whether the halo is the most massive progenitor
    ("scale_of_last_MM", "float64"), # scale factor at last MM
    ("vmax", "float64"),
    ("x", "float64"), ("y", "float64"), ("z", "float64"), # halo position (comoving Mpc/h)
    ("vx", "float64"), ("vy", "float64"), ("vz", "float64"), # halo velocity (physical peculiar km/s)
    ("Jx", "float64"), ("Jy", "float64"), ("Jz", "float64"), # halo angular momentum
    ("Spin", "float64"), # cbx check if int

    # Various things for merger trees
    ("Breadth_first_ID", "int"),
    ("Depth_first_ID", "int"),
    ("Tree_root_ID", "int"),
    ("Orig_halo_ID", "int"),
    ("Snap_num", "int"),
    ("Next_coprogenitor_depthfirst_ID", "int"),
    ("Last_progenitor_depthfirst_ID", "int"),
    ("Last_mainleaf_depthfirst_ID", "int"),

    # More physical properties but more arcane
    ("Rs_Klypin", "float64"), # scale radius from vmax and mvir (see Rockstar)
    ("Mmvir_all", "float64"), # mass enclosed in specified overdensity (cbx what is the specified overdensity) including unbound particles
    ("M200b", "float64"),
    ("M200c", "float64"),
    ("M500c", "float64"),
    ("M2500c", "float64"),
    ("Xoff", "float64"), # offset of density peak form average particle position
    ("Voff", "float64"), # offset of density peak form average velocity position
    ("Spin_Bullock", "float64"), # Bullock spin parameter

    # Shape stuff
    ("b_to_a", "float64"), # Ratio of second largest axis to largest (< 1) (Allgood et al. (2006).)
    ("c_to_a", "float64"), # Ratio of third largest axis to largest (< 1) (Allgood et al. (2006).)
    ("a[x]", "float64"), ("a[y]", "float64"), ("a[z]", "float64"), # (cbx ???)
    ("b_to_a(500c)", "float64"),
    ("c_to_a(500c)", "float64"),
    ("a[x](500c)", "float64"), ("a[y](500c)", "float64"), ("a[z](500c)", "float64"), # (cbx ???)

    ("T/|U|", "float64"), # ratio of kinetic to potential energy
    ("M_pe_Behroozi", "float64"), # pseudo evolution corrected masses (experimental)
    ("M_pe_Diemer", "float64"), # pseudo evolution corrected masses (experimental)
    ("Macc", "float64"), # Mass at accretion
    ("Mpeak", "float64"), # peak mass (cbx is this over the whole time? or up to this point)
    ("Vacc", "float64"), # Velocity at accretion
    ("Vpeak", "float64"), # peak vmax
    ("Halfmass_Scale", "float64"), # scale factor at which we could to 0.5 * mpeak

    # Acc rate
    ("Acc_Rate_Inst", "float64"),
    ("Acc_Rate_100Myr", "float64"),
    ("Acc_Rate_1*Tdyn", "float64"),
    ("Acc_Rate_2*Tdyn", "float64"),
    ("Acc_Rate_Mpeak", "float64"),

    ("Mpeak_Scale", "float64"), # scale factor at which we reached mpeak
    ("Acc_Scale", "float64"), # scale factor at which we last accreted a sattelite
    ("First_Acc_Scale", "float64"), # Scale at which current and former satellites first passed through a larger halo.
    ("First_Acc_Mvir", "float64"), # mvir at first_acc_scale
    ("First_Acc_Vmax", "float64"), # vmax at first_acc_scale
    ("Vmax@Mpeak", "float64"), # vmax at the scale where mpeak was reached
] # yapf: disable



data_dir = "/home/christopher/Data/data/universe_machine/"
def main():
    data_file = data_dir + "hlist_0.71240.list"
    # Reduce the number of colunms and save as a numpy array
    inter_file = data_dir + "hlist_0.71240_reduced_cols.npy"
    # Current DB
    final_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400_final.npz"
    # final final DB
    final_extended_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400_final_extended.npz"


    # If we have already generated the inter_file, don't do it again...
    if not os.path.isfile(inter_file):
        reduced_catalog = helpers.reduce_cols(data_file, full_hlist_dtype, reduced_hlist_dtype)
        np.save(inter_file, reduced_catalog)
        del reduced_catalog
    else:
        print("Skipping reducing cols, file already exists")

    if not os.path.isfile(final_extended_file):
        # Load current catalog
        current = np.load(final_file)
        current_df = pd.concat([
            pd.DataFrame(current["centrals"]),
            pd.DataFrame(current["satellites"]),
        ]).set_index("id")
        del current
        # Load the the extensions we just created and join on id
        extension_df = pd.DataFrame(np.load(inter_file)).set_index("id")
        result_df = pd.concat([current_df, extension_df], axis=1, join_axes=[current_df.index])
        del current_df, extension_df
        # Save out result converting back to structured numpy array
        result = result_df.to_records()
        result = result.view(result.dtype.descr, np.ndarray)
        del result_df
        np.savez(final_extended_file,
                centrals=result[result["upid"] == -1],
                satellites=result[result["upid"] != -1]
        )
    else:
        print("Skipping combining data, file already exists")


if __name__ == "__main__":
    main()
