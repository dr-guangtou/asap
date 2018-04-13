solarMassUnits = "M_{\odot}"

m_vir_x_axis = r"$log\ M_{vir}/M_{\odot}$"
hm_scatter = r"$\sigma_{log\ M_{vir}}$"

sm_scatter_simple = r"$\sigma_{log\ M_{\ast}}$"
m_star_x_axis_simple = r"$log\ M_{\ast}$"

def m_star_x_axis(n_sats):
    return r"$log\ M_{\ast," + str(n_sats) + "}/M_{\odot}$"
def m_star_legend(n_sats):
    return r"$M_{\ast," + str(n_sats) + "}$"
def sm_scatter(n_sats):
    return r"$\sigma(log\ M_{\ast," + str(n_sats) + "})$"



number_density = r"$\Phi (\frac{dn}{dlogM_{\ast}} h^{3} Mpc^{-3} dex^{-1})$"
cum_number_density = r"Cumulative number density $h^{3} Mpc^{-3}$"
number_density_richness = r"$\Phi (\frac{dn}{d\Lambda} h^{3} Mpc^{-3} dex^{-1})$"
richness = r"Richness"
cum_count = "Cumulative Count"
log_cum_count = "log10(Cumulative Count)"
