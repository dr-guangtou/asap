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
import numpy as np
import pandas as pd
import os
import halotools.sim_manager

hlist_cols = {
    "id": (1, "int32"), # id of halo
    "pid": (5, "int32"), # least massive parent (direct parent) halo ID
    "mvir": (10, "float64"), # Msun/h
    "rvir": (11, "float64"), # kpc/h
    "rs": (12, "float64"), # scale radius kpc/h
    "Halfmass_Scale": (61, "float64"), # scale factor at which we could to 0.5 * mpeak
    "scale_of_last_MM": (15, "float64"), # scale factor at last MM
    "M200b": (37, "float64"),
    "M200c": (38, "float64"),
    "Acc_Rate_Inst": (62, "float64"),
    "Acc_Rate_100Myr": (63, "float64"),
    "Acc_Rate_1*Tdyn": (64, "float64"),
    "Acc_Rate_2*Tdyn": (65, "float64"),
    "Acc_Rate_Mpeak": (66, "float64"),
    "Vmax@Mpeak": (72, "float64"), # vmax at the scale where mpeak was reached
}

data_dir = "/home/christopher/Data/data/universe_machine/"


def main():
    data_file = data_dir + "hlist_0.71240.list"
    # Reduce the number of colunms and save as a numpy array
    inter_file = data_dir + "hlist_0.71240_reduced_cols_wpid.npy"
    # Current DB
    final_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400_final.npz"
    # final final DB
    final_extended_file = data_dir + "sfr_catalog_insitu_exsitu_0.712400_final_extended_wpid.npz"

    # If we have already generated the inter_file, don't do it again...
    if not os.path.isfile(inter_file):
        hlist_reader = halotools.sim_manager.TabularAsciiReader(data_file, hlist_cols)
        reduced_catalog = hlist_reader.read_ascii()
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
        np.savez(
            final_extended_file,
            centrals=result[result["upid"] == -1],
            satellites=result[result["upid"] != -1])
    else:
        print("Skipping combining data, file already exists")


if __name__ == "__main__":
    main()
