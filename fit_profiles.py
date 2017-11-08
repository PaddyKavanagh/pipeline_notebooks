# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:54:22 2016

@author: Patrick Kavanagh (DIAS)

Dependencies:
PyAbel - pip install PyAbel
"""
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import *
from scipy.stats import chisquare
import optparse
from abel.direct import direct_transform
from astropy.convolution import convolve, Gaussian1DKernel


class ProcessInput():
    """
    Class to process the input for profile fitting.  See self.load for the list of inputs
    """
    def __init__(self):
        print("Setting up input...")

    def _check_input(self):
        """
        Do some checks on the input
        """
        # check input files exist
        assert os.path.isfile(self.profile_filename), 'Input file not found'
        if self.psf_filename is not None:
            assert os.path.isfile(self.psf_filename), 'PSF file not found'

        # make sure that a psf file is supplied when kernel = 'custom'
        if self.kernel == 'custom':
            assert self.psf_filename is not None, 'You must supply an input PSF file for the custom kernel'

        # check that bin_size is positive
        assert self.bin_size > 0, 'bin_size must be > 0'

    def load(self, file, psffile=None, region='my_region', kernel='gauss',
                 gauss_fwhm=3, profile='postshock', bin_size=5.):
        """
        Load the input, check the input, initial processing of input. Inputs are:

        file:       ds9 funtools file containing the surface brightness profile

        Optional:
        psffile:    ds9 funtools file containing the psf surface brightness profile

        region:     name of the region (for output naming and plotting)
                    (default = my_region)

        kernel:     name of the convolution to be applied to the profile. Options are:
                    custom: apply a custom kernel determined from psffile
                    gauss:  apply a Gaussian kernel, fwhm given by gauss_fwhm in pixels
                    (TODO: check that this is pixels and not arcsec)
                    none:   do not apply a kernel
                    (default = 'gauss')

        gauss_fwhm: fwhm of the Gaussian kernel if kernel='gauss' is set
                    (default = 3)

        profile:    type of physical profile to fit to the data. Options are:
                    postshock:  exponential decay in the postshock region
                    precursor:  exponential decay in the postshock region with precursor
                    sb:         exponential decay outward
                    patch:      exponential decay in a spherical cap type postshock region

        bin_size:   real size of the profile bins in arcsec
                    (default=5)
        """
        # load the input
        self.profile_filename = file
        self.psf_filename = psffile
        self.region_name = region
        self.kernel = kernel
        self.gauss_fwhm = gauss_fwhm
        self.profile_type = profile
        self.bin_size = bin_size

        # check input
        self._check_input()

        # load the profile data
        self.profile_data = np.loadtxt(self.profile_filename, comments="#", unpack=True, skiprows=2)
        self.profile_regnum = self.profile_data[0, :]
        self.profile_surfbri = self.profile_data[6, :]
        self.profile_surfbri_err = self.profile_data[7, :]

        # subtract a background, assume the last 10 points are background
        self.bkg_level = np.mean(self.profile_surfbri[-10:-1])
        self.profile_net_surfbri = self.profile_surfbri - self.bkg_level

        # load the psf profile data if given
        if self.psf_filename is not None:
            self.psf_profile_data = np.loadtxt(self.psf_filename, comments="#", unpack=True, skiprows=2)
            self.psf_profile_regnum = self.profile_data[0, :]
            self.psf_profile_surfbri = self.profile_data[6, :]
            self.psf_profile_surfbri_err = self.profile_data[7, :]

        # convert the bin number to real distance in arcsec using the bin size
        self.profile_radius = self.profile_regnum * self.bin_size
        if self.psf_filename is not None:
            self.psf_profile_radius = self.psf_profile_regnum * self.bin_size

    def test_plot_profile(self, plot_type='raw'):
        """
        Do a quick test plot of the loaded data

        :param plot_type:   type of plot. Options:
                            raw:        plot surface brightness against bin no
                            physical:   plot surface brightness against radial distance (in arcsec)
                            (Default = 'raw')
        """
        if plot_type == 'raw':
            x = self.profile_regnum
        elif plot_type == 'physical':
            x = self.profile_radius

        y = self.profile_surfbri
        yerr = self.profile_surfbri_err
        net_y = self.profile_net_surfbri
        bkg = self.bkg_level

        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        plt.tight_layout(pad=3.0)

        axs.errorbar(x, y, yerr, c='b', marker='o', markersize=2, linestyle='-',
                     linewidth=0.5, label='measured')
        axs.errorbar(x, net_y, yerr, c='r', marker='o', markersize=2, linestyle='-',
                     linewidth=0.5, label='bkg subtracted')
        axs.plot([min(x), max(x)], [bkg, bkg], c='g', marker='o', markersize=0, linestyle='--',
                 linewidth=1, label='bkg level')
        axs.plot([min(x), max(x)], [0, 0], c='k', marker='o', markersize=0, linestyle=':',
                     linewidth=1, label='zero level')

        axs.set_ylabel('surface brightness')
        if plot_type == 'raw':
            axs.set_xlabel('Bin number')
        elif plot_type == 'physical':
            axs.set_xlabel('Radius (arcsec)')
        axs.legend(prop={'size':10}, loc=0)
        plt.show()

    def test_plot_psf(self, plot_type='raw'):
        """
        Do a quick test plot of the loaded psf

        :param plot_type:   type of plot. Options:
                            raw:        plot psf against bin no
                            physical:   plot psf radial distance (in arcsec)
                            (Default = 'raw')
        """
        try:
            if plot_type == 'raw':
                x = self.psf_profile_regnum
            elif plot_type == 'physical':
                x = self.psf_profile_radius
        except AttributeError as e:
            print("AttributeError: {0}".format(e))
            print("No psf file was provided!\n")

        y = self.psf_profile_surfbri
        yerr = self.psf_profile_surfbri_err

        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        plt.tight_layout(pad=3.0)

        axs.errorbar(x, y, yerr, c='b', marker='o', markersize=2, linestyle='-',
                     linewidth=0.5, label='measured')
        axs.plot([min(x), max(x)], [0, 0], c='k', marker='o', markersize=0, linestyle=':',
                     linewidth=1, label='zero level')

        axs.set_ylabel('surface brightness')
        if plot_type == 'raw':
            axs.set_xlabel('Bin number')
        elif plot_type == 'physical':
            axs.set_xlabel('Radius (arcsec)')
        axs.legend(prop={'size':10}, loc=0)
        plt.show()


class ProfileModel():
    """
    Class to define profile to fit to data. See the individual model methods
    for descriptions.

    profile:    type of physical profile to fit to the data. Options are:
                postshock:  exponential decay in the postshock region
                precursor:  exponential decay in the postshock region with precursor
                superbubble:exponential decay outward
                patch:      exponential decay in a spherical cap type postshock region
    """
    def __init__(self):
        print("Creating model...")

        # create the x array for the profile
        self.profile_radius = np.linspace(0.1, 300., 3000)

    def postshock(self, a, l, r):
        """
        Produce a profile with a jump, then exponential decay in the
        postshock region.

        :param a:   model normalisation
        :param l:   shell width
        :param r:   shell radius

        :returns:   profile model
        """
        profile_model = np.zeros(self.profile_radius.shape)

        for n, rad in enumerate(self.profile_radius):
            if (rad > r):
                profile_model[n] = 0.0
            else:
                profile_model[n] = a * np.exp(-1 * np.absolute((rad - r) / l))

        return profile_model

    def cap(self, a, l, r, w):
        """
        Produce a profile of a projected cap of emission on the shell

        :param a:   model normalisation
        :param l:   shell width
        :param r:   shell radius
        :param w:   cap radius

        :returns:   profile model
        """
        profile_model = np.zeros(self.profile_radius.shape)

        for n, rad in enumerate(self.profile_radius):
            if (rad > r):
                profile_model[n] = 0.0
            else:
                profile_model[n] = a * np.exp(-1 * np.absolute((rad - r) / l))
                if rad <  w:
                    profile_model[n] = 0

        return profile_model

    def postshock_to_nonzero(self, a, l, r, b):
        """
        Produce a profile with a jump, then exponential decay to a non-zero level

        :param a:   model normalisation
        :param l:   shell width
        :param r:   shell radius
        :param b:   non-zero interior level as fraction of a

        :returns:   profile model
        """
        profile_model = np.zeros(self.profile_radius.shape)

        for n, rad in enumerate(self.profile_radius):
            if (rad > r):
                profile_model[n] = 0.0
            else:
                profile_model[n] = (b * a) + (a * np.exp(-1 * np.absolute((rad - r) / l)))

        return profile_model

    def precursor(self, a, l, r, f):
        """
        Produce a profile with a precursor component with peak
        a factor of f less than the postshock component

        x = list of off axis points
        f = factor of postshock peak in relation to precursor
        l = shell width
        r = shell radius
        """
        profile_model = np.zeros(self.profile_radius.shape)

        for n, rad in enumerate(self.profile_radius):
            if (rad > r):
                profile_model[n] = (a/f) * np.exp(-1*np.absolute((r - rad)/l))
                # TODO check what the l value should be for a given f (for now just l)
            else:
                profile_model[n] = a * np.exp(-1 * np.absolute((rad - r) / l))

        return profile_model

    def superbubble(self, a, l, r):
        """
        Produce a profile with a jump, then exponential decay outward
        (Not very physical, just for curiosity)

        :param a:   model normalisation
        :param l:   shell width
        :param r:   shell radius

        :returns:   profile model
        """
        profile_model = np.zeros(self.profile_radius.shape)

        for n, rad in enumerate(self.profile_radius):
            if (rad > r):
                profile_model[n] = a * np.exp(-1 * np.absolute((r - rad) / l ))
            else:
                profile_model[n] = 0

        return profile_model

    def _abel_forward(self):
        """
        Take in the profile and perform Abel transform
        (3D volumetric to 2D projected)
        """
        dr = self.profile_data[0, 3] - self.profile_data[0, 2]
        self.profile_abel_transform = direct_transform(self.profile_data, dr=dr,
                                                       direction='forward', correction=True)

    def _convolveG1D(self, fwhm):
        """
        Convolve a 1D array with a kernel
        """
        # need to convert the fwhm supplied in arcsec to data points
        sigma = fwhm / (np.diff(self.profile_radius)[0])
        gauss_kernel = Gaussian1DKernel(sigma)
        self.profile_convolved = convolve(self.profile_abel_transform[1,:],
                                          gauss_kernel)

    def create(self, profile, a=1., l=10., r=150., w=120., b=0.2, f=16.,
               kernel='gauss', fwhm=3):
        """
        load the model

        :param profile: profile name.
        :param a:       model normalisation
        :param l:       shell width
        :param r:       shell radius
        :param w:       cap radius (if cap_profile)
        :param b:       non-zero interior level as fraction of a (if postshock_to_nonzero)
        :param f:       factor of postshock peak in relation to precursor (if precursor)
        :param fwhm:    FWHM of a 1D Gaussian if kernel='gauss'
        """
        self.profile_name = profile

        if self.profile_name == 'postshock':
            self.profile_model = self.postshock(a, l, r)

        if self.profile_name == 'cap':
            self.profile_model = self.cap(a, l, r, w)

        if self.profile_name == 'postshock_to_nonzero':
            self.profile_model = self.postshock_to_nonzero(a, l, r, b)

        if self.profile_name == 'precursor':
            self.profile_model = self.precursor(a, l, r, f)

        if self.profile_name == 'superbubble':
            self.profile_model = self.superbubble(a, l, r)

        # create the profile data array
        self.profile_data = np.array([self.profile_radius, self.profile_model])

        # do the Abel transform
        self._abel_forward()

        # apply convolution
        if kernel == 'gauss':
            self._convolveG1D(fwhm=fwhm)

        return (self.profile_data, self.profile_abel_transform,
                self.profile_convolved)

    def test_plot_profile(self):
        """
        Do a quick test plot of the loaded profile

        """
        fig, axs = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
        plt.tight_layout(pad=3.0)

        axs[0].plot(self.profile_data[0,:], self.profile_data[1,:], c='b', marker='o', markersize=0, linestyle='-',
                     linewidth=1.0, label='%s profile' % self.profile_name)
        axs[0].legend(prop={'size': 10}, loc=0)
        axs[1].plot(self.profile_data[0,:], self.profile_abel_transform[1, :], c='r', marker='o', markersize=0, linestyle='-',
                 linewidth=1.0, label='%s Abel' % self.profile_name)
        axs[1].legend(prop={'size':10}, loc=0)
        axs[2].plot(self.profile_data[0,:], self.profile_convolved[:], c='g', marker='o', markersize=0, linestyle='-',
                 linewidth=1.0, label='%s convolved' % self.profile_name)
        axs[2].set_xlabel('Radius (arcsec)')
        axs[2].legend(prop={'size':10}, loc=0)
        plt.tight_layout()
        plt.show()


class FitProfiles():
    """
    Class to perform the profile fits to the data

        Load the input, check the input, initial processing of input. Inputs are:

        file:       ds9 funtools file containing the surface brightness profile

        Optional:
        psffile:    ds9 funtools file containing the psf surface brightness profile

        region:     name of the region (for output naming and plotting)
                    (default = my_region)

        kernel:     name of the convolution to be applied to the profile. Options are:
                    custom: apply a custom kernel determined from psffile
                    gauss:  apply a Gaussian kernel, fwhm given by gauss_fwhm in pixels
                    (TODO: check that this is pixels and not arcsec)
                    none:   do not apply a kernel
                    (default = 'gauss')

        gauss_fwhm: fwhm of the Gaussian kernel if kernel='gauss' is set
                    (default = 3)

        profile:    type of physical profile to fit to the data. Options are:
                    postshock:  exponential decay in the postshock region
                    precursor:  exponential decay in the postshock region with precursor
                    sb:         exponential decay outward
                    patch:      exponential decay in a spherical cap type postshock region

        bin_size:   real size of the profile bins in arcsec
                    (default=5)
        """
    def __init__(self, file, region, profile, kernel='gauss',
                 psffile=None, gauss_fwhm=3, bin_size=5.):

        # load the input
        self.profile_filename = file
        self.psf_filename = psffile
        self.region_name = region
        self.kernel = kernel
        self.gauss_fwhm = float(gauss_fwhm)
        self.profile_type = profile
        self.bin_size = bin_size

        # load the data
        self.my_data = ProcessInput()
        self.my_data.load(file, psffile=self.psf_filename, region=self.region_name, kernel=self.kernel,
                          gauss_fwhm=self.gauss_fwhm, profile=self.profile_type, bin_size=self.bin_size)

        # create model profile
        #self.my_model = ProfileModel()

    def test_profile_plots(self):
        """
        Run test plots
        """
        # show data and bkg subbed data
        self.my_data.test_plot_profile(plot_type='physical')

        # show a sample model profile, Abel transform and convolution
        my_model = ProfileModel()
        my_model.create(self.profile_type, a=10., l=20., r=150.,
                             kernel=self.kernel, fwhm=self.gauss_fwhm)
        my_model.test_plot_profile()

    def run_initial_fits(self):
        """
        Run initial fits with sparse paramter space sampling
        to narrow the fits
        """
        # parameter ranges (a, l, r = norm, width, radius)
        a_range = np.linspace(0.001, 0.02, 20)
        r_range = np.linspace(130., 180., 10)
        l_range = np.linspace(1., 10., 10)
        fit_num_total = a_range.shape[0] * r_range.shape[0] * l_range.shape[0]

        if self.profile_type == 'cap':
            # w = cap radius
            w_range = np.linspace(100, 200., 10)
            fit_num_total = fit_num_total * w_range.shape[0]

        if self.profile_type == 'postshock_to_nonzero':
            # b = background level (fraction of a)
            b_range = np.linspace(0.01, 0.1, 10)
            fit_num_total = fit_num_total * b_range.shape[0]

        if self.profile_type == 'precursor':
            # f factor of precursor compared to postshock
            f_range = np.linspace(10, 20., 10)
            fit_num_total = fit_num_total * b_range.shape[0]

        # run the fits
        if self.profile_type == 'postshock':
            # instantiate the ProfileModel class
            my_model = ProfileModel()

            # create an empty array to write statisical results
            results = np.zeros(fit_num_total, dtype={'names':['a', 'l', 'r', 'chi_sq', 'dof', 'red_chisq'],
                                                     'formats':['<f4','<f4','<f4','<f4','<f4','<f4']})

            fit_num = 0
            for a in a_range:
                for r in r_range:
                    for l in l_range:
                        if self.kernel == 'gauss':
                            # create the model
                            my_model.create(self.profile_type, a=a, l=l, r=r,
                                            kernel=self.kernel, fwhm=self.gauss_fwhm)

                            # need to extract the points at the same radius as the data
                            my_model_at_data = np.zeros(self.my_data.profile_radius.shape)

                            for n,row in enumerate(self.my_data.profile_radius):
                                for rrow in my_model.profile_radius:
                                    if abs(rrow - row) < 0.0001:
                                        my_model_at_data[n] = my_model.profile_convolved[n]

                            # determine chi-square, dof, red chi-square
                            # TODO work out the chisq taking the uncertainty into account
                            chi_sq, p = chisquare(self.my_data.profile_net_surfbri, my_model_at_data)

                            chi_sq = np.sum((self.my_data.profile_net_surfbri - my_model_at_data)**2 / self.my_data.profile_surfbri_err**2)

                            dof = my_model_at_data.shape[0] - 1
                            red_chisq = chi_sq / dof
                            results[fit_num] = (a, l, r, chi_sq, dof, red_chisq)

                        fit_num += 1
                        sys.stdout.write("Fit progress: %d/%d   \r" % (fit_num, fit_num_total))
                        sys.stdout.flush()



            # testing plot - best fit
            best_fit_index = np.argmin(results['red_chisq'])
            my_best_fit_model = ProfileModel()
            my_best_fit_model.create(self.profile_type, a=results['a'][best_fit_index],
                                     l=results['l'][best_fit_index], r=results['r'][best_fit_index],
                                     kernel=self.kernel, fwhm=self.gauss_fwhm)

            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            plt.tight_layout(pad=3.0)

            axs.errorbar(self.my_data.profile_radius, self.my_data.profile_net_surfbri,
                         self.my_data.profile_surfbri_err, c='b', marker='o', markersize=2, linestyle='-',
                         linewidth=0.5, label='data')
            axs.plot(my_best_fit_model.profile_data[0,:], my_best_fit_model.profile_convolved[:],
                     c='r', marker='x', markersize=0,
                     linestyle='-', lw=1, label='best-fit')

            axs.annotate("a = %0.2f, l = %0.2f, r = %0.2f,\n chi_sq = %0.2f, dof=%0.2f, red_chi_sq = %0.2f" % (results['a'][best_fit_index],
                         results['l'][best_fit_index], results['r'][best_fit_index], results['chi_sq'][best_fit_index],
                         results['dof'][best_fit_index], results['red_chisq'][best_fit_index]), xy=(0.01,0.95), xycoords='axes fraction',
                         fontsize=8, color='k')
            axs.set_xlabel('Radius (arcsec)')
            axs.legend(prop={'size': 10}, loc=0)
            plt.show()

            # testing plot - red_chisq vs. parameters
            fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey=True)
            plt.tight_layout(pad=3.0)
            axs = axs.ravel()

            axs[0].plot(results['a'], results['red_chisq'], c='b', marker='o', markersize=3,
                     linestyle='-', lw=0, label='a')
            axs[0].set_xlabel('a')
            axs[0].set_ylabel('red_chisq')
            axs[0].set_yscale('log')
            axs[0].set_ylim(0.1,100)
            axs[0].grid(color='k', linestyle='--', linewidth=0.5)
            axs[0].legend(prop={'size': 10}, loc=0)

            axs[1].plot(results['l'], results['red_chisq'], c='r', marker='o', markersize=3,
                        linestyle='-', lw=0, label='l')
            axs[1].set_xlabel('l')
            axs[1].set_ylim(0.1, 100)
            axs[1].grid(color='k', linestyle='--', linewidth=0.5)
            axs[1].legend(prop={'size': 10}, loc=0)

            axs[2].plot(results['r'], results['red_chisq'], c='g', marker='o', markersize=3,
                        linestyle='-', lw=0, label='r')
            axs[2].set_xlabel('r')
            axs[2].set_ylim(0.1, 100)
            axs[2].grid(color='k', linestyle='--', linewidth=0.5)
            axs[2].legend(prop={'size': 10}, loc=0)

            #plt.grid()
            plt.show()

            # testing plot - red_chisq contour
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
            plt.tight_layout(pad=3.0)
            axs = axs.ravel()

            axs[0].scatter(results['r'], results['a'], c=results['red_chisq'], s=100,
                           cmap='jet')
            axs[0].set_xlabel('r')
            axs[0].set_ylabel('a')

            axs[1].scatter(results['l'], results['a'], c=results['red_chisq'], s=100,
                           cmap='jet')
            axs[1].set_xlabel('l')
            axs[1].set_ylabel('a')

            plt.show()



if __name__ == "__main__":
    """
    Execute the following if called from the command line
    """
    # parse arguments
    usage = "Usage: %prog <profile file> [options]"

    parser = optparse.OptionParser(usage)
    parser.add_option('-r', '--regname', dest='regname', action='store',
                      help="Name of the region (for output naming/plotting)", default='my_region')
    parser.add_option('-s', '--profile', dest='profile', action='store',
                      help="Profile shape. Options: postshock, precursor, sb, patch", default='postshock')
    parser.add_option('-k', '--kernel', dest='kernel', action='store',
                      help="Name of convolution kernel to be applied. Options: custom, gauss, none",
                      default='gauss')
    parser.add_option('-p', '--psffile', dest='psffile', action='store',
                      help="Name of the PSF profile file (ds9 funtools output) if kernel='custom'",
                      default=None)
    parser.add_option('-f', '--fwhm', dest='fwhm', action='store',
                      help="FWHM of the kernel if kernel='gauss'", default=3.)
    parser.add_option('-b', '--bin_size', dest='bin_size', action='store',
                      help="Size of the profile bins in arcsec", default=5.)
    parser.add_option('-t', '--test', dest='test', action='store_true',
                      help="Produce test plots", default=False)
    (options, args) = parser.parse_args()

    # TODO finish the full list of options

    # read command line args
    try:
        input_file = args[0]

        if options.test == False:
            my_fits = FitProfiles(input_file, psffile=options.psffile, region=options.regname,
                            kernel=options.kernel, gauss_fwhm=options.fwhm, profile=options.profile,
                            bin_size=options.bin_size)
            my_fits.run_initial_fits()

        elif options.test == True:
            my_fits = FitProfiles(input_file, psffile=options.psffile, region=options.regname,
                            kernel=options.kernel, gauss_fwhm=options.fwhm, profile=options.profile,
                            bin_size=options.bin_size)
            my_fits.test_profile_plots()

    except IndexError as e:
        print("IndexError: {0}".format(e))
        print("No input file provided!\n")
        print(parser.print_help())
