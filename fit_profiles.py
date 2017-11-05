# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:54:22 2016

@author: Patrick Kavanagh (DIAS)
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import savefig
from math import *
import fit_profile_functions
import scipy.optimize as opt
import optparse


class ProcessInput():
    """
    Class to process the input for profile fitting.  See self.load for the list of inputs
    """
    def __init__(self):
        print("setting up input...")

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
    (options, args) = parser.parse_args()

    # read command line args
    try:
        input_file = args[0]
        my_profile = ProcessInput()
        my_profile.load(input_file, psffile=options.psffile, region=options.regname,
                        kernel=options.kernel, gauss_fwhm=options.fwhm, profile=options.profile,
                        bin_size=options.bin_size)

    except IndexError as e:
        print("IndexError: {0}".format(e))
        print("No input file provided!\n")
        print(parser.print_help())

    # create a test plot of the loaded input
    my_profile.test_plot_profile(plot_type='physical')
