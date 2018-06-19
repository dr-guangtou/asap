import numpy as np

def resample_sats(cen_sample, sat_sample, num_sats, incl_icl = True):
    assert len(cen_sample) == len(num_sats)
    full_poisson_masses, half_poisson_masses, full_poisson_masses_no_cen = [], [], []

    mean_sats = int(np.mean(num_sats))
    mean_cen_mass = np.mean(cen_sample["sm"] + cen_sample["icl"])

    for i in range(len(cen_sample)):
        base_mass = cen_sample[i]["sm"]
        if incl_icl:
            base_mass += cen_sample[i]["icl"]
        for _ in range(10):

            chosen_sats = np.random.choice(sat_sample, np.random.poisson(int(mean_sats)))
            additional_mass = np.sum(chosen_sats["sm"] + chosen_sats["icl"])
            full_poisson_masses.append(np.log10(base_mass + additional_mass))

            full_poisson_masses_no_cen.append(np.log10(mean_cen_mass + additional_mass))

            chosen_sats = np.random.choice(sat_sample, num_sats[i])
            additional_mass = np.sum(chosen_sats["sm"] + chosen_sats["icl"])
            half_poisson_masses.append(np.log10(base_mass + additional_mass))

    return full_poisson_masses, half_poisson_masses, full_poisson_masses_no_cen
