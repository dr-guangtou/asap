import numpy as np

def rollup(sats, central_ids):
    sats = np.copy(sats) # so we can mutate.

    rolled_sats = []
    for central_id in central_ids:
        rolled_sats.append(_rollup_one(sats[sats["upid"] == central_id]))

    return np.concatenate(rolled_sats)

def _rollup_one(sats):

    is_top_level_subhalo = (sats["pid"] == sats["upid"])
    sat_directory = {
            # for each id: index, is_top_level_subhalo
            sats[i]["id"]: i for i in range(len(sats))
    }

    for i in range(len(sats)):
        if is_top_level_subhalo[i]:
            continue

        idx = i
        while True:
            try:
                idx = sat_directory[sats[idx]["pid"]]
            except KeyError: # No parent info. Orphan halo
                assert np.isnan(sats[idx]["pid"])
                idx = -1
                break


            if is_top_level_subhalo[idx]:
                break

        if idx == -1:
            continue

        sats[idx]["sm"] += sats[i]["sm"]
        sats[idx]["icl"] += sats[i]["icl"]

    return sats[is_top_level_subhalo]
