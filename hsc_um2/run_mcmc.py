#!/usr/bin/python

from optparse import OptionParser
import sys
import numpy as np
import abhodfits


def main():
    print ''
    print ' Running MCMC '
    print ''
    print (sys.version)
    print ''
    input_data_file = ''
    output_chain_file = ''

    # List of initial parameters
    start_parameters = np.array(
        [1.258, 13.339, 0.345, 11.962, 11.918, 0.01, -0.01])

    parser = OptionParser()

    parser.add_option('--ifile', type='string',
                      help='input data file containing wp(rp)')
    parser.add_option('--covarfile', type='string',
                      help='covariance matrix file for wp(rp)')
    parser.add_option('--ofile', type='string',
                      help='output file to write chain samples to')

    parser.add_option('--start', nargs=7, type='float',
                      help='alpha log(M1) sigma_log(M) log(M0) log(Mmin)')

    parser.add_option('--walkers', default=10, type='int',
                      help='integer number of mcmc walkers')

    parser.add_option('--samples', default=10000, type='int',
                      help='integer number of mcmc samples per walker')

    (options, args) = parser.parse_args()

    if options.start != None:
        start_parameters = np.array(
            [options.start[0], options.start[1], options.start[2], options.start[3], options.start[4], options.start[5], options.start[6]])

    if options.ifile == None:
        print ' Input file must be specified with --ifile option'
        sys.exit()
    else:
        input_data_file = options.ifile

    if options.ofile == None:
        print 'Output chain file must be specified with --ofile option'
        sys.exit()
    else:
        output_chain_file = options.ofile

    if options.covarfile == None:
        print 'Covariance matrix file must be specified with --covarfile.'
        sys.exit()
    else:
        covar_file = options.covarfile

    num_walkers = options.walkers
    num_samples = options.samples

    print ' Input data file    : ', input_data_file
    print ' Covariance file    : ', covar_file
    print ' Output chain file  : ', output_chain_file
    print ' '
    print ' start_parameters   : ', start_parameters
    print ' '
    print ' number of walkers  : ', num_walkers
    print ' samples per walker : ', num_samples
    print ' total samples      : ', num_walkers * num_samples

    # Now we are ready to run an MCMC, so let's do it
    print ' beginning mcmc'

    hmfit = abhodfits.ABHodFitModel(
        datafile=input_data_file, covarfile=covar_file)
    hmfit.mcmcfit(start_parameters, nwalkers=num_walkers, samples=num_samples)
    hmfit.save_chains(filename=output_chain_file)

    print ' sampling complete '
    print ' '
    print ' alpha : ', hmfit.alpha_mcmc
    print ' logM1 : ', hmfit.logM1_mcmc
    print ' slogM : ', hmfit.sigma_logM_mcmc
    print ' logM0 : ', hmfit.logM0_mcmc
    print ' lMmin : ', hmfit.logMmin_mcmc
    print ' Acen  : ', hmfit.Acen_mcmc
    print ' Asat  : ', hmfit.Asat_mcmc
    print ' '


if __name__ == "__main__":
    main()
