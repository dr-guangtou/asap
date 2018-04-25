import numpy as np
import pandas as pd

# Given two of our data sets, return the shared columns
def unify(data, key1, key2):

    d1 = pd.DataFrame(data[key1]["data"], index=data[key1]["data"]["id"])
    d2 = pd.DataFrame(data[key2]["data"], index=data[key2]["data"]["id"])


    joined = d1.join(d2, how="inner", lsuffix="_{}".format(key1), rsuffix="_{}".format(key2))

    return joined
