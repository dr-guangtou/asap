m_vir_x_axis = r"$\mathrm{log}\ M_{\rm vir}/M_{\odot}$"
hm_scatter = r"$\sigma_{\mathrm{log}\ M_{\rm vir}/M_{\odot}}$"

sm_scatter_simple = r"$\sigma_{\mathrm{log}\ M_{\ast, \rm x}/M_{\odot}}$"
m_star_x_axis_simple = r"$\mathrm{log}\ M_{\ast}$"

def m_star_x_axis(n_sats):
    return r"$\mathrm{log}\ M_{\ast, \rm" + str(n_sats) + r"}/M_{\odot}$"
def m_star_legend(n_sats):
    return r"$M_{\ast, \rm" + str(n_sats) + "}$"
def sm_scatter(n_sats):
    return r"$\sigma_{\mathrm{log}\ M_{\ast, \rm" + str(n_sats) + r"}/M_{\odot}}$"

def sm_scatter_x_given_hm(x):
    return r"$\sigma_{M_{\ast, \rm{" + str(x) + r"}} | M_{halo}}$"

def sm_scatter_x_given_sm_y(x, y):
    return r"$\sigma_{M_{\ast, \rm{" + str(x) + r"}} | M_{\ast, \rm{" + str(y) + r"}}}$"

def sm_delta_x_given_sm_y(x, y):
    return r"$\Delta_{M_{\ast, \rm{" + str(x) + r"}} | M_{\ast, \rm{" + str(y) + r"}}}$"


def mstar_mhalo_fit(n_sats):
    return r"$M_{\ast, \rm" + str(n_sats) + "}-M_{halo}$ fit"


number_density = r"$\Phi (\frac{dn}{dlogM_{\ast}} h^{3} ^{-3} dex^{-1})$"
cum_number_density = r"Cumulative number density [$h^{3} \mathrm{Mpc^{-3}}$]"
number_density_richness = r"$\Phi (\frac{dn}{d\Lambda} h^{3} Mpc^{-3} dex^{-1})$"
richness = r"Richness"
ngals = r"$N_{\mathrm{gals}}$"
cum_count = "Cumulative Count"
log_cum_count = "log10(Cumulative Count)"

scatter_photoz = r"$\sigma_{N_{\mathrm{gals}}, \mathrm{photoz}}$"
scatter_specz = r"$\sigma_{N_{\mathrm{gals}}, \mathrm{specz}}$"
scatter_ideal = r"$\sigma_{N_{\mathrm{gals}}, \mathrm{ideal}}$"
scatter_ngals = r"$\sigma_{N_{\mathrm{gals}}}$"

### Gamma
gamma = r"$\Gamma$"
gamma2 = r"$\Gamma_{2}$"


### Colors
r2014 = "#00FF33"
r2009 = "#990099"
