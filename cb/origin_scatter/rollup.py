import numpy as np

def rollup(sats, central_ids):
    all_rolled_sats, all_num_sats = [], []
    for central_id in central_ids:
        this_sats = sats[sats["upid"] == central_id]
        all_rolled_sats.append(_rollup_one(this_sats))
        all_num_sats.append(len(this_sats))

    return np.concatenate(all_rolled_sats), np.array(all_num_sats)

def _rollup_one(sats):
    is_top_level_subhalo = (sats["pid"] == sats["upid"]) | (np.isnan(sats["pid"]))
    sat_directory = {
            # id: index
            sats[i]["id"]: i for i in range(len(sats))
    }

    for i in range(len(sats)):
        if is_top_level_subhalo[i]:
            continue

        top_level_parent_idx = _find_top_level_subhalo(sats, i, sat_directory, is_top_level_subhalo)

        if top_level_parent_idx == -1:
            continue

        sats[top_level_parent_idx]["sm"] += sats[i]["sm"]
        sats[top_level_parent_idx]["icl"] += sats[i]["icl"]

    return sats[is_top_level_subhalo]

def _find_top_level_subhalo(sats, start_index, sat_directory, is_top_level_subhalo):
    idx = start_index
    while True:
        pid = sats[idx]["pid"]
        if np.isnan(pid): # orphan halo
            raise Exception("We shouldn't get here anymore now that orphans are classed as TLH")

        try:
            idx = sat_directory[pid]
        except KeyError: # I don't actually know what this case is
            return -1

        if is_top_level_subhalo[idx]:
            return idx


# We have id, pid and upid.
# if pid is nan, either we have lost this halo or we have lost the parent
def build_sat_tree(all_sats, central_ids):
    sums = {
            "total": 0,
            "nth": [0,0,0,0,0,0,0,0,0],
            "orphan": 0,
            "not_found": 0,
    }
    sm = {
            "total": 0,
            "nth": [0,0,0,0,0,0,0,0,0],
            "orphan": 0,
            "not_found": 0,
    }
    for idx in central_ids:
        sats = all_sats[all_sats["upid"] == idx]
        sums["total"] += len(sats)
        sm["total"] += np.sum(sats["sm"] + sats["icl"])

        found = []
        ids = [idx]
        for i in range(len(sums["nth"])):
            nth_sats = sats[np.isin(sats["pid"], ids)]
            found.append(nth_sats["id"])
            sums["nth"][i] += len(nth_sats)
            sm["nth"][i] += np.sum(nth_sats["sm"] + nth_sats["icl"])
            ids = nth_sats["id"]

        orphan = sats[np.isnan(sats["pid"])]
        found.append(orphan["id"])
        sums["orphan"] += len(orphan)
        sm["orphan"] += np.sum(orphan["sm"] + orphan["icl"])

        all_found = np.concatenate(found)
        not_found = sats[np.logical_not(np.isin(sats["id"], all_found))]
        sums["not_found"] += len(not_found)
        sm["not_found"] += np.sum(not_found["sm"] + not_found["icl"])

    sums["well_found"] = sum(sums["nth"])
    sm["well_found"] = sum(sm["nth"])

    print(sums)
    print(sums["total"] - (sum(sums["nth"]) + sums["orphan"] + sums["not_found"]))

    print({k: np.log10(v) for k, v in sm.items()})
