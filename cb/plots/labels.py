m_vir_x_axis = r"$\mathrm{log}\ M_{\rm vir}/M_{\odot}$"
hm_scatter = r"$\sigma_{\mathrm{log}\ M_{\rm vir}/M_{\odot}}$"

sm_scatter_simple = r"$\sigma_{\mathrm{log}\ M_{\ast, \rm x}/M_{\odot}}$"
m_star_x_axis_simple = r"$\mathrm{log}\ M_{\ast}$"

def m_star_x_axis(n_sats):
    return r"$\mathrm{log}\ M_{\ast, \rm" + str(n_sats) + r"}/M_{\odot}$"
def m_star_legend(n_sats):
    return r"$M_{\ast, \rm" + str(n_sats) + "}$"
def sm_scatter(n_sats):
    return r"$\sigma_{\mathrm{log}\ M_{\ast, \rm" + str(n_sats) + "}/M_{\odot}}$"



number_density = r"$\Phi (\frac{dn}{dlogM_{\ast}} h^{3} ^{-3} dex^{-1})$"
cum_number_density = r"Cumulative number density [$h^{3} \mathrm{Mpc^{-3}}$]"
number_density_richness = r"$\Phi (\frac{dn}{d\Lambda} h^{3} Mpc^{-3} dex^{-1})$"
richness = r"Richness"
ngals = r"$N_{\mathrm{gals}}$"
cum_count = "Cumulative Count"
log_cum_count = "log10(Cumulative Count)"

scatter_observed = r"$\sigma_{N_{\mathrm{gals}}, \mathrm{obs}}$"
scatter_intrinsic = r"$\sigma_{N_{\mathrm{gals}}, \mathrm{ideal}}$"

### Gamma
gamma = r"$\Gamma$"


### Colors
r2014 = "#00FF33"
r2009 = "#990099"
