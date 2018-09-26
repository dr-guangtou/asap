# ###############################################################################
#
# Copyright (C) 2012-2017, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# Updated versions of the software are available from my web page
# http://purl.org/cappellari/software
#
# If you have found this software useful for your research, I would
# appreciate an acknowledgement to the use of the "LTS_PLANEFIT program
# described in Cappellari et al. (2013, MNRAS, 432, 1709), which
# combines the Least Trimmed Squares robust technique of Rousseeuw &
# van Driessen (2006) into a least-squares fitting algorithm which
# allows for errors in all variables and intrinsic scatter."
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################
#+
# NAME:
#       LTS_PLANEFIT
#
# PURPOSE:
#       Best plane *robust* fit to data with errors in all
#       three coordinates and fitting for the intrinsic scatter.
#       See Sec.3.2 here http://adsabs.harvard.edu/abs/2013MNRAS.432.1709C
#
# EXPLANATION:
#       Linear Least-squares approximation in two-dimension (z = a + b*x + c*y),
#       when x, y and z data have errors, and allowing for intrinsic
#       scatter in the relation.
#
#       Outliers are iteratively clipped using the extremely robust
#       FAST-LTS technique by Rousseeuw & van Driessen (2006)
#       http://dx.doi.org/10.1007/s10618-005-0024-4
#       See also http://books.google.co.uk/books?id=woaH_73s-MwC&pg=PA15
#
# CALLING SEQUENCE:
#        p = lts_planefit(x, y, z, sigx, sigy, sigz, clip=2.6, epsz=True,
#                        frac=None, pivotx=None, pivoty=None, plot=True, text=True)
#
#       The output values are stored as attributes of "p".
#       See usage example at the bottom of this file.
#
# INPUT PARAMETERS:
#       x, y, z: vectors of size N with the measured values.
#       sigx, sigy, sigz: vectors of size N with the 1sigma errors in x , y and z.
#       clip: values deviating more than clip*sigma from the best fit are
#           considered outliers and are excluded from the plane fit.
#       epsz: if True, the intrinsic scatter is printed on the plot.
#       frac: fractions of values to include in the LTS stage.
#           Up to a fraction "frac" of the values can be outliers.
#           One must have 0.5 < frac < 1  (default frac=0.5).
#         - Set frac=1, to turn off outliers detection.
#       pivotx, pivoty: if these are not None, then lts_planefit fits the plane
#               z = a + b*(x - pivotx) + c*(y - pivoty)
#           pivotx, pivoty are called x_0, y_0 in eq.(7) of Cappellari et al. (2013)
#           Use of these keywords is *strongly* recommended, and suggested
#           values are pivotx ~ np.mean(x), pivoty ~ np.mean(y).
#           This keyword is important to reduce the covariance between a, b and c.
#       plot: if True a plot of the fit is produced.
#       text: if True, the best fitting parameters are printed on the plot.
#
# OUTPUT PARAMETERS:
#       The output values are stored as attributed of the lts_linefit class.
#
#       p.abc: best fitting parameters [a, b, c]
#       p.abc_err: 1*sigma formal errors [a_err, b_err, c_err] on a, b and c.
#       p.mask: boolean vector with the same size of x, y and z.
#           It contains True for the elements of (x, y, z) which were included in
#           the fit and False for the outliers which were automatically clipped.
#       p.sig_int: intrinsic scatter in the z direction around the plane.
#           sig_int is called epsilon_z in eq.(7) of Cappellari et al. (2013).
#       p.sig_int_err: 1*sigma formal error on sig_int.
#
# MODIFICATION HISTORY:
#       V1.0.0: Michele Cappellari, Oxford, 21 March 2011
#       V2.0.0: Converted from lts_linefit. MC, Oxford, 06 April 2011
#       V2.0.1: Added PIVOT keyword, MC, Oxford, 1 August 2011
#       V2.0.2: Fixed program stop affecting earlier IDL versions.
#           Thanks to Xue-Guang Zhang for reporting the problem
#           and the solution. MC, Turku, 10 July 2013
#       V2.0.3: Scale line spacing with character size in text output.
#           MC, Oxford, 19 September 2013
#       V2.0.4: Check that all input vectors have the same size.
#           MC, Baltimore, 8 June 2014
#       V2.0.5: Text plotting changes. MC, Oxford, 26 June 2014
#       V3.0.0: Converted from IDL into Python. MC, Oxford, 5 November 2014
#       V3.0.1: Updated documentation. MC, Baltimore, 9 June 2015
#       V3.0.2: Fixed potential program stop without outliers.
#           Thanks to Masato Onodera for a clear report of the problem.
#         - Output boolean mask instead of good/bad indices.
#         - Removed lts_planefit_example from this file.
#           MC, Oxford, 6 July 2015
#       V3.0.3: Fixed potential program stop without outliers.
#           MC, Oxford, 1 October 2015
#       V3.0.4: Fixed potential program stop without outliers in Matplotlib 1.5.
#           MC, Oxford, 9 December 2015
#       V3.0.5: Use LimeGreen for outliers. MC, Oxford, 8 January 2016
#       V3.0.6: Check for non finite values in input.
#           MC, Oxford, 23 January 2016
#       V3.0.7: Added capsize=0 in plt.errorbar to prevent error bar caps
#           from showing up in PDF. MC, Oxford, 4 July 2016
#       V3.0.8: Fixed: store ab errors in p.ab_err as documented.
#           Thanks to Alison Crocker for the correction.
#           MC, Oxford, 5 September 2016
#       V3.0.9: Fixed typo causing full C-step to be skipped.
#           Thanks to Francesco D'Eugenio for reporting this problem.
#           Increased upper limit of intrinsic scatter accounting for
#           uncertainty of standard deviation with small samples.
#           Michele Cappellari, Oxford, 26 July 2017
#-
#------------------------------------------------------------------------------

from __future__ import print_function

from time import clock
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#----------------------------------------------------------------------------

def _planefit(x, y, z, sigz=None, weights=None):
    """
    Fit a plane z = a + b*x + c*y to a set of points (x, y, z)
    by minimizing chi2 = np.sum(((z - zfit)/sigz)**2)

    """
    v1 = np.ones_like(x)
    if weights is None:
        if sigz is None:
            sw = v1
        else:
            sw = v1/sigz
    else:
        sw = np.sqrt(weights)

    a = np.column_stack([v1, x, y])
    abc = np.linalg.lstsq(a*sw[:, None], z*sw)[0]

    return abc

#----------------------------------------------------------------------------

def _display_errors(par, sig_par, epsz):
    """
    Print parameters rounded according to their errors

    """
    prec = np.zeros_like(par)
    w = (sig_par != 0) & (par != 0)
    prec[w] = np.ceil(np.log10(np.abs(par[w]))) - np.floor(np.log10(sig_par[w])) + 1
    prec = prec.clip(0)  # negative precisions not allowed
    dg = list(map(str, prec.astype(int)))

    # print on the terminal and save as string

    txt = ['intercept: ', 'slopeX: ', 'slopeY: ', 'scatter: ']
    for t, d, p, s in zip(txt, dg, par, sig_par):
        print('%12s' % t, ('%.' + d + 'g') % p, '+/- %.2g' % s)

    txt = ['a=', 'b=', 'c=', '\\varepsilon_z=']
    if not epsz:
        txt = txt[:-1]
    string = ''
    for t, d, p, s in zip(txt, dg, par, sig_par):
        string += '$' + t + ('%.' + d + 'g') % p + '\\pm%.2g' % s + '$\n'

    return string

#------------------------------------------------------------------------------

def _residuals(abc, x, y, z, sigx, sigy, sigz):
    """
    See equation (7) of Cappellari et al. (2013, MNRAS, 432, 1709)

    """
    res = (abc[0] + abc[1]*x + abc[2]*y - z) \
        / np.sqrt((abc[1]*sigx)**2 + (abc[2]*sigy)**2 + sigz**2)

    return res

#----------------------------------------------------------------------------

def _fitting(x, y, z, sigx, sigy, sigz, abc):

    abc, pcov, infodict, errmsg, success = optimize.leastsq(
        _residuals, abc, args=(x, y, z, sigx, sigy, sigz), full_output=1)

    if pcov is None or np.any(np.diag(pcov) < 0):
        sig_ABC = np.full(3, np.inf)
        chi2 = np.inf
    else:
        chi2 = np.sum(infodict['fvec']**2)
        sig_ABC = np.sqrt(np.diag(pcov))  # ignore covariance

    return abc, sig_ABC, chi2

#----------------------------------------------------------------------------

def _fast_algorithm(x, y, z, sigx, sigy, sigz, h):

    # Robust least trimmed squares regression.
    # Pg. 38 of Rousseeuw & van Driessen (2006)
    # http://dx.doi.org/10.1007/s10618-005-0024-4
    #
    m = 500 # Number of random starting points
    abcv = np.empty((m, 3))
    chi2v = np.empty(m)
    for j in range(m):  # Draw m random starting points
        w = np.random.choice(x.size, 3, replace=False)
        abc = _planefit(x[w], y[w], z[w])  # Find a plane going trough three random points
        for k in range(3):  # Run C-steps up to H_3
            res = _residuals(abc, x, y, z, sigx, sigy, sigz)
            good = np.argsort(np.abs(res))[:h]  # Fit the h points with smallest errors
            abc, sig_abc, chi_sq = _fitting(x[good], y[good], z[good], sigx[good], sigy[good], sigz[good], abc)
        abcv[j, :] = abc
        chi2v[j] = chi_sq

    # Perform full C-steps only for the 10 best results
    #
    w = np.argsort(chi2v)
    nbest = 10
    chi_sq = np.inf
    for j in range(nbest):
        abc1 = abcv[w[j], :]
        while True:  # Run C-steps to convergence
            abcOld = abc1
            res = _residuals(abc1, x, y, z, sigx, sigy, sigz)
            good1 = np.argsort(np.abs(res))[:h]  # Fit the h points with smallest errors
            abc1, sig_ab1, chi1_sq = _fitting(x[good1], y[good1], z[good1], sigx[good1], sigy[good1], sigz[good1], abc1)
            if np.allclose(abcOld, abc1):
                break
        if chi_sq > chi1_sq:
            abc = abc1  # Save best solution
            good = good1
            chi_sq = chi1_sq

    mask = np.zeros_like(x, dtype=bool)
    mask[good] = True

    return abc, mask

#------------------------------------------------------------------------------

class lts_planefit(object):

    def _find_outliers(self, sig_int, x, y, z, sigx, sigy, sigz1, h, offs, clip):

        sigz = np.sqrt(sigz1**2 + sig_int**2) # Gaussian intrinsic scatter

        if h == x.size: # No outliers detection

            abc = _planefit(x, y, z, sigz=sigz)  # quick initial guess
            abc, sig_abc, chi_sq = _fitting(x, y, z, sigx, sigy, sigz, abc)
            mask = np.ones_like(x, dtype=bool)  # No outliers

        else: # Robust fit and outliers detection

            # Initial estimate using the maximum breakdown of
            # the method of 50% but minimum efficiency
            #
            abc, mask = _fast_algorithm(x, y, z, sigx, sigy, sigz, h)

            # inside-out outliers removal
            #
            while True:
                res = _residuals(abc, x, y, z, sigx, sigy, sigz)
                sig = np.std(res[mask], ddof=3)
                maskOld = mask
                mask = np.abs(res) < clip*sig
                abc, sig_abc, chi_sq = _fitting(x[mask], y[mask], z[mask], sigx[mask], sigy[mask], sigz[mask], abc)
                if np.array_equal(mask, maskOld):
                    break

        # To determine 1sigma error on the intrinsic scatter the chi2
        # is decreased by 1sigma=sqrt(2(h-3)) while optimizing (a,b,c)
        #
        h = mask.sum()
        dchi = np.sqrt(2*(h - 3)) if offs else 0.

        self.abc = abc
        self.abc_err = sig_abc
        self.mask = mask

        err = (chi_sq + dchi)/(h - 3.) - 1
        print('sig_int: %10.4f  %10.4f' % (sig_int, err))

        return err

#------------------------------------------------------------------------------

    def _single_fit(self, x, y, z, sigx, sigy, sigz, h, clip):

        if self._find_outliers(0, x, y, z, sigx, sigy, sigz, h, 0, clip) < 0:
            print('No intrinsic scatter or errors overestimated')
            sig_int = 0.
            sig_int_err = 0.
        else:
            sig1 = 0.
            res = self.abc[0] + self.abc[1]*x + self.abc[2]*y - z  # Total residuals ignoring measurement errors
            std = np.std(res[self.mask], ddof=3)
            sig2 = std*(1 + 3/np.sqrt(2*self.mask.sum()))  # Observed scatter + 3sigma error
            print('Computing sig_int')
            sig_int = optimize.brentq(self._find_outliers, sig1, sig2,
                                      args=(x, y, z, sigx, sigy, sigz, h, 0, clip), rtol=1e-3)
            print('Computing sig_int error') # chi2 can always decrease
            sigMax_int = optimize.brentq(self._find_outliers, sig_int, sig2,
                                         args=(x, y, z, sigx, sigy, sigz, h, 1, clip), rtol=1e-3)
            sig_int_err = sigMax_int - sig_int

        self.sig_int = sig_int
        self.sig_int_err = sig_int_err

        print('Repeat at best fitting solution')
        self._find_outliers(sig_int, x, y, z, sigx, sigy, sigz, h, 0, clip)

#------------------------------------------------------------------------------

    def __init__(self, x0, y0, z, sigx, sigy, sigz, clip=2.6, epsz=True,
                  frac=None, pivotx=0, pivoty=0, plot=True, text=True):

        assert x0.size == y0.size == z.size == sigx.size == sigy.size == sigz.size, '[X, Y, Z, SIGX, SIGY, SIGZ] must have the same size'

        if not np.all(np.isfinite(np.hstack([x0, y0, z, sigx, sigy, sigz]))):
            raise ValueError('Input contains non finite values')

        t = clock()

        x = x0 - pivotx
        y = y0 - pivoty

        p = 3  # three dimensions
        n = x.size
        h = int((n + p + 1)/2) if frac is None else int(max(round(frac*n), (n + p + 1)/2))

        self._single_fit(x, y, z, sigx, sigy, sigz, h, clip)
        rms = np.std(self.abc[0] + self.abc[1]*x[self.mask] + self.abc[2]*y[self.mask] - z[self.mask], ddof=3)

        par = np.append(self.abc, self.sig_int)
        sig_par = np.append(self.abc_err, self.sig_int_err)
        print('################# Values and formal errors ################')
        string = _display_errors(par, sig_par, epsz)
        print('Observed rms scatter: %.3g ' % rms)
        if pivotx or pivoty:
            print('z = a + b*(x - pivotx) + c*(y - pivoty)')
            print('with pivotx = %.4g & pivoty = %.4g' % (pivotx, pivoty))
        print('##########################################################')

        print('seconds %.2f' % (clock() - t))

        if plot:

            z1 = par[0] + x*par[1] + y*par[2]
            sigz1 = np.sqrt((sigx*par[1])**2 + (sigy*par[2])**2)

            plt.errorbar(z[self.mask], z1[self.mask], xerr=sigz[self.mask], yerr=sigz1[self.mask],
                         fmt='ob', capthick=0, capsize=0)
            if not np.all(self.mask):
                plt.errorbar(z[~self.mask], z1[~self.mask], xerr=sigz[~self.mask], yerr=sigz1[~self.mask],
                             fmt='d', color='LimeGreen', capthick=0, capsize=0)
            plt.autoscale(False)
            plt.title('Best fit, 1$\sigma$ (68%) and 2.6$\sigma$ (99%)')

            # Extends lines well beyond plot to allow for rescaling.
            # Lines are plotted below the symbols using z-buffer

            xmin, xmax = np.min(z), np.max(z)
            dx = xmax - xmin
            xlim = np.array([xmin-dx, xmax+dx])
            plt.plot(xlim, xlim, '-k',
                     xlim, xlim + rms, '--r',
                     xlim, xlim - rms, '--r',
                     xlim, xlim + 2.6*rms, ':r',
                     xlim, xlim - 2.6*rms, ':r', linewidth=2, zorder=1)

            ax = plt.gca()
            if text:
                string += '$\Delta=%.2g$\n' % rms
                if pivotx:
                    string += '$(x_0=%.4g)$\n' % pivotx
                if pivoty:
                    string += '$(y_0=%.4g)$' % pivoty
                ax.text(0.05, 0.95, string, horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

            ax.minorticks_on()
            ax.tick_params(length=10, width=1, which='major')
            ax.tick_params(length=5, width=1, which='minor')

#------------------------------------------------------------------------------
