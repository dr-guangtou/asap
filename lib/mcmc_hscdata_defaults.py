# A default set of priors
default_priors = {}
default_priors['alpha'] = [0.0, 2.0]
default_priors['logM1'] = [10.7, 15.0]
default_priors['sigma_logM'] = [0.02, 1.5]
default_priors['logM0'] = [9.0, 14.0]
default_priors['logMmin'] = [9.0, 14.0]
default_priors['mean_occupation_centrals_assembias_param1'] = [-1.0, 1.0]
default_priors['mean_occupation_satellites_assembias_param1'] = [-1.0, 1.0]


# Some default fit parameters
default_ndim = 7
default_nwalkers = 28

# Default HOD Model Features

# Default starting parameters
default_start = [1.00, 13.08, 0.26, 12.35, 11.83, 0.0, 0.0]

# Default rp_bins to match SDSS DR7 analysis
# -- bin choice if running out to rp=26.8
# default_log10_rpmin=-0.869443
# default_log10_rpmax=1.5280296
# default_nrpbins=13
# -- bin choice of running out to rp=16.9
default_rpcut = 17.0
default_log10_rpmin = -0.869423
default_log10_rpmax = 1.32776
default_nrpbins = 12

# Default pi_max, to match SDSS DR7 analysis
default_pi_max = 60.0

# Default simulation information
default_simname = 'bolplanck'
default_simredshift = 0.0
default_halofinder = 'rockstar'
default_version_name = 'halotools_v0p4'

# Default data file
default_wp_datafile = 'sdss_wp20.0.dat'
default_binfile = 'sdss_binfile.dat'
default_wp_covarfile = 'sdss_wp_covar_20.0.dat'
