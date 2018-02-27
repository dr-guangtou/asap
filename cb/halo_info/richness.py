def get_richness(halos, satellites, min_mass):
    richness = np.zeros(len(halos), int)
    for i, central in enumerate(halos):
        sats = satellites[np.searchsorted(
            satellites["upid"], central["id"], side="left"):np.searchsorted(
                satellites["upid"], central["id"], side="right")]
        sats = sats[::-1]

        for j in range(len(sats)):
            if sats[j]["icl"] + sats[j]["sm"] < min_mass:
                richness[i] = j
                break
        else:
            richness[i] = len(sats)

    return richness
