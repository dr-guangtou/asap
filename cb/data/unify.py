import numpy as np
import pandas as pd

# Given two of our data sets, return the shared columns
def unify(data, key1, key2, append=None):

    d1 = pd.DataFrame(data[key1]["data"], index=data[key1]["data"]["id"])
    d2 = pd.DataFrame(data[key2]["data"], index=data[key2]["data"]["id"])
    assert np.all(d1.index == d2.index)

    d1 = d1.rename(columns={"sm": "sm_" + key1, "icl": "icl_" + key1, "sfr": "sfr_" + key1})

    d1["sm_" + key2] = d2["sm"]
    d1["icl_" + key2] = d2["icl"]
    d1["sfr_" + key2] = d2["sfr"]

    for df_to_append in (append or []):
        assert np.all(d1.index == df_to_append.index)
        for col in list(df_to_append):
            d1[col] = df_to_append[col]

    return d1

# The above relys on the IDs being the same (e.g. data_halo_cut). This doesn't
def unify_different(data, key1, key2):
    d1 = pd.DataFrame(data[key1]["data"], index=data[key1]["data"]["id"])
    d2 = pd.DataFrame(data[key2]["data"], index=data[key2]["data"]["id"])
    print(len(d1), len(d2))

    d1 = d1.join(d2, how="inner", lsuffix="_"+key1, rsuffix="_"+key2)
    print(len(d1))
    return d1
