"""Predictions of the A.S.A.P. model."""

from full_mass_profile_model import mass_prof_model_simple, \
    mass_prof_model_frac1

__all__ = ['asap_predict_mass']


def asap_predict_mass(parameters, cfg, obs_data, um_data,
                      constant_bin=False):
    """M100, M10, Mtot using Mvir, M_gal, M_ICL.

    Parameters
    ----------

    parameters : array, list, or tuple
        Model parameters.

    cfg : dict
        Configurations of the data and model.

    obs_data: dict
        Dictionary for observed data.

    um_data: dict
        Dictionary for UniverseMachine data.

    constant_bin : boolen
        Whether to use constant bin size for logMs_tot or not.
    """
    if cfg['model_type'] == 'simple':
        return mass_prof_model_simple(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=obs_data['obs_smf_tot_min'],
            max_logms=obs_data['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    elif cfg['model_type'] == 'frac1':
        return mass_prof_model_frac1(
            um_data['um_mock'],
            obs_data['obs_logms_tot'],
            obs_data['obs_logms_inn'],
            parameters,
            min_logms=obs_data['obs_smf_tot_min'],
            max_logms=obs_data['obs_smf_tot_max'],
            n_bins=cfg['um_mtot_nbin'],
            constant_bin=constant_bin,
            logmh_col=cfg['um_halo_col'],
            logms_col=cfg['um_star_col'],
            min_scatter=cfg['um_min_scatter'],
            min_nobj_per_bin=cfg['um_min_nobj_per_bin']
            )
    else:
        raise Exception("# Wrong model choice! ")
