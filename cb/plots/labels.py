solarMassUnits = r"($M_{\odot}$)"

m_vir_x_axis = r"$log\ M_{vir}$"
hm_scatter = r"$\sigma(log\ M_{vir})$"

sm_scatter_simple = r"$\sigma(log\ M_{\ast})$"
m_star_x_axis_simple = r"$log\ M_{\ast}$"
m_star_cen_x_axis_simple = r"$log\ M_{\ast}^{cen}$"
def m_star_x_axis(n_sats):
    return r"$log\ M_{\ast}^{" + str(n_sats) + "}$"
def sm_scatter(n_sats):
    return r"$\sigma(log\ M_{\ast}^{" + str(n_sats) + "})$"
