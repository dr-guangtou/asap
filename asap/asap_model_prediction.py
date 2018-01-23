"""Predictions of the A.S.A.P. model."""

from stellar_mass_function import get_smf_bootstrap
from full_mass_profile_model import mass_prof_model_simple, \
    mass_prof_model_frac1

__all__ = ['asap_predict_mass', 'asap_predict_smf', 'asap_predict_model']


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


def asap_predict_smf(logms_mod_tot, logms_mod_inn, cfg):
    """Stellar mass functions of Minn and Mtot predicted by UM."""
    # SMF of the predicted Mtot (M1100)
    um_smf_tot = get_smf_bootstrap(logms_mod_tot,
                                   cfg['um_volume'],
                                   cfg['obs_smf_tot_nbin'],
                                   cfg['obs_smf_tot_min'],
                                   cfg['obs_smf_tot_max'],
                                   n_boots=1)

    # SMF of the predicted Minn (M10)
    um_smf_inn = get_smf_bootstrap(logms_mod_inn,
                                   cfg['um_volume'],
                                   cfg['obs_smf_inn_nbin'],
                                   cfg['obs_smf_inn_min'],
                                   cfg['obs_smf_inn_max'],
                                   n_boots=1)

    return um_smf_tot, um_smf_inn


def asap_predict_model(parameters, cfg, obs_data, um_data,
                       constant_bin=False, return_all=False,
                       show_smf=False, show_dsigma=False):
    """Return all model predictions.

    Parameters:
    -----------

    parameters: list, array, or tuple.
        Input model parameters.

    cfg : dict
        Configurations of the data and model.

    obs_data: dict
        Dictionary for observed data.

    um_data: dict
        Dictionary for UniverseMachine data.


    constant_bin : boolen
        Whether to use constant bin size for logMs_tot or not.

    return_all : bool, optional
        Return all model information.

    show_smf : bool, optional
        Show the comparison of SMF.

    show_dsigma : bool, optional
        Show the comparisons of WL.

    """
    # Predict stellar mass
    (logms_mod_inn, logms_mod_tot_all,
     logms_mod_halo, mask_mtot, um_mock_use) = asap_predict_mass(
         parameters, cfg, obs_data, um_data, constant_bin=constant_bin)

    # Predict the SMFs
    um_smf_tot, um_smf_inn = self.umPredictSMF(
        logms_mod_tot_all[mask_mtot],
        logms_mod_inn)

    # TODO: If one mass bin is empty, set the error to a higer value
    # mask_zero = um_smf_tot['smf'] <= 1.0E-10
    # um_smf_tot['smf'][mask_zero] = np.nan
    # um_smf_tot['smf_err'][mask_zero] = np.nan

    um_wl_profs = self.umPredictWL(logms_mod_tot_all[mask_mtot],
                                   logms_mod_inn,
                                   mask_mtot,
                                   add_stellar=self.um_wl_add_stellar)

    if plotSMF:
        um_smf_tot_all = get_smf_bootstrap(logms_mod_tot_all,
                                           self.um_volume,
                                           20, 10.5, 12.5,
                                           n_boots=1)
        logms_mod_tot = logms_mod_tot_all[mask_mtot]
        plot_mtot_minn_smf(self.obs_smf_tot, self.obs_smf_inn,
                           self.obs_mtot, self.obs_minn,
                           um_smf_tot, um_smf_inn,
                           logms_mod_tot,
                           logms_mod_inn,
                           obs_smf_full=self.obs_smf_full,
                           um_smf_tot_all=um_smf_tot_all)

    if plotWL:
        # TODO: add halo mass information
        plot_dsigma_profiles(self.obs_wl_dsigma,
                             um_wl_profs,
                             obs_mhalo=None,
                             um_wl_mhalo=None)

    return (um_smf_tot, um_smf_inn, um_wl_profs,
            logms_mod_inn, logms_mod_tot_all[mask_mtot],
            logms_mod_halo, mask_mtot,
            um_mock_use)
