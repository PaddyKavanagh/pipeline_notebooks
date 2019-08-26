#!/usr/bin/env python

"""

Setup file for installing the MiriTeam software

:History:
11 May 2017: Major new setup.py script based on ez_setup.py, using
             the miriteam.im setup.py as a template and copying code
             from the package defsetup.py scripts.
12 May 2017: MIRI software divided into 4 parts: MiriTools, MiriPipeline,
             MiriSimulators and MiriCalibration.
14 Jul 2017: Unpack the 470 micron version of the cosmic ray libraries.
27 Jul 2017: Require Python 2.7
30 Nov 2017: Removed MiriTools and MiriSimulators and changed namespace
             from miri to miriteam.
15 Jan 2018: Removed MiriCalibration and MiriPipeline levels from package.
25 Jan 2018: Added missing url metadata.
27 Apr 2018: Require Python 3.5.
22 May 2018: Added README and LICENCE.
20 Jun 2018: Added jwst_pipeline packages (PK, DIAS)
14 Sep 2018: Added jwst_pipeline cfg_files (PK, DIAS)

@author: Steven Beard (UKATC)

"""

import io
import os
import re
import sys
import zipfile
import numpy

try:
    from setuptools import Extension
#    from distutils.core import Extension
    from Cython.Distutils import build_ext
except ImportError:
    build_ext = None

try:
    from setuptools import setup
except ImportError:
    from .ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup



def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Test the command arguments given with this script.
# Only unzip data files when building or installing, not when 
# cleaning. If a "clean" is requested on its own, the previously 
# unzipped files are deleted.
if len(sys.argv[0]) > 0:
    argv = sys.argv
if ("build" in argv) or ("install" in argv):
    zipflag = True
    cleanflag = False
else:
    zipflag = False
    cleanflag = ("clean" in argv)
if "--quiet" in argv:
    verbose = False
else:
    verbose = True


# ------------------------------------------------------------------
# Unzip the data files contained in the spectroscopy data directories.
#
# The example data files are found relative to the 
# directory containing this Python script.
(this_dir, this_file) = os.path.split(__file__)
mrs_data_path = os.path.join(this_dir, "spectroscopy/mrs/data")
if not os.path.isdir(mrs_data_path):
    strg = "MRS spectroscopy data directory %s not found" % mrs_data_path
    raise EnvironmentError(strg)

fziplist = [zipfile.ZipFile(os.path.join(mrs_data_path,'testCUBE_CH1_A.zip'),'r')]

if zipflag:
    # Unpack the data files
    for fzip in fziplist:
        for name in fzip.namelist():
            fullname = os.path.join(mrs_data_path, name)
            if not os.path.isfile(fullname):
                if verbose:
                    print( "Unzipping \'%s\'" % fullname )
                data = fzip.read(name)
                temp = open(fullname, "wb")
                temp.write(data)
                temp.close()
            else:
                if verbose:
                    print( "\'%s\' already exists" % fullname )

        fzip.close()

elif cleanflag:
    # Clean up the data files
    for fzip in fziplist:
        for name in fzip.namelist():
            fullname = os.path.join(mrs_data_path, name)
            if os.path.isfile(fullname):
                if verbose:
                    print( "Deleting \'%s\'" % fullname )
                try:
                    os.remove(fullname)
                except Exception:
                    pass

# ------------------------------------------------------------------
#
# Build the cython - C interface in lrs/ipipe/spectralExtraction
# using Cython utilities (if available).
#
if build_ext is not None:
#    setup(
   cmdclass = {'build_ext': build_ext}
   ext_modules = [Extension("polyclip", 
                            sources=["spectroscopy/lrs/ipipe/spectralExtraction/polyclip.pyx",
                                     "spectroscopy/lrs/ipipe/spectralExtraction/c_polyclip.c"],
                            include_dirs=[numpy.get_include()]),
                  Extension("extractExtended",
                            sources=["spectroscopy/lrs/ipipe/spectralExtraction/extractExtended.pyx",
                                     "spectroscopy/lrs/ipipe/spectralExtraction/c_extractExtended.c"],
                            include_dirs=[numpy.get_include()])
                  ]
#          )
else:
    import warnings
    warnings.warn("Cython utilities not available. spectroscopy/lrs C functions not built.")
    cmdclass = {}
    ext_modules = []

# ------------------------------------------------------------------


setup(
    name="miriteam",
    version=find_version("__init__.py"),
    description="MIRI calibration, pipeline and data analysis software",
    url="https://github.com/JWST-MIRI/MiriTeam",
    author="MIRI European Consortium",
    author_email="F.Lahuis@sron.nl",
    license="See LICENCE file",
    platforms=["Linux", "Mac OS X", "Win"],
    python_requires='>=3.5',
    packages=['miriteam',
              'miriteam.calibration_generator', 'miriteam.calibration_generator.tests',
              'miriteam.calibration_generator.darks',
              'miriteam.calibration_generator.reset_correction',
              'miriteam.imaging', 'miriteam.imaging.tests',
              'miriteam.coronography', 'miriteam.coronography.tests',
              'miriteam.spectroscopy', 'miriteam.spectroscopy.tests',
              'miriteam.spectroscopy.lrs', 'miriteam.spectroscopy.lrs.tests',
              'miriteam.spectroscopy.lrs.ipipe', 'miriteam.spectroscopy.lrs.ipipe.tests',
              'miriteam.spectroscopy.lrs.ipipe.spectralExtraction',
              'miriteam.spectroscopy.mrs', 'miriteam.spectroscopy.mrs.tests',
              'miriteam.spectroscopy.mrsdetex', 'miriteam.spectroscopy.mrsdetex.tests',
              'miriteam.jwst_pipeline', 'miriteam.jwst_pipeline.steps', 'miriteam.jwst_pipeline.pipelines',
              'miriteam.jwst_pipeline.tests', 'miriteam.jwst_pipeline.cfg_files',
             ],
    package_dir={
                 'miriteam': '',
                 'miriteam.calibration_generator': 'calibration_generator',
                 'miriteam.calibration_generator.tests': 'calibration_generator/tests',
                 'miriteam.calibration_generator.darks': 'calibration_generator/darks',
                 'miriteam.calibration_generator.reset_correction': 'calibration_generator/reset_correction',
                 'miriteam.imaging': 'imaging/',
                 'miriteam.imaging.tests': 'imaging/tests',
                 'miriteam.coronography': 'coronography/',
                 'miriteam.coronography.tests': 'coronography/tests',
                 'miriteam.spectroscopy': 'spectroscopy/',
                 'miriteam.spectroscopy.tests': 'spectroscopy/tests',
                 'miriteam.spectroscopy.lrs': 'spectroscopy/lrs',
                 'miriteam.spectroscopy.lrs.tests': 'spectroscopy/lrs/tests',
                 'miriteam.spectroscopy.lrs.ipipe': 'spectroscopy/lrs/ipipe',
                 'miriteam.spectroscopy.lrs.ipipe.tests': 'spectroscopy/lrs/ipipe/tests',
                 'miriteam.spectroscopy.lrs.ipipe.spectralExtraction': 'spectroscopy/lrs/ipipe/spectralExtraction',
                 'miriteam.spectroscopy.mrs': 'spectroscopy/mrs',
                 'miriteam.spectroscopy.mrs.tests': 'spectroscopy/mrs/tests',
                 'miriteam.spectroscopy.mrsdetex': 'spectroscopy/mrsdetex',
                 'miriteam.spectroscopy.mrsdetex.tests': 'spectroscopy/mrsdetex/tests',
                 'miriteam.jwst_pipeline': 'jwst_pipeline/',
                 'miriteam.jwst_pipeline.steps': 'jwst_pipeline/steps',
                 'miriteam.jwst_pipeline.pipelines': 'jwst_pipeline/pipelines',
                 'miriteam.jwst_pipeline.tests': 'jwst_pipeline/tests',
                 'miriteam.jwst_pipeline.cfg_files': 'jwst_pipeline/cfg_files',
                 },
    package_data={'miriteam.imaging': ['schemas/*.yaml', 'data/*.fits',
                                     'data/*.txt', 'data/__init__.py'],
                  'miriteam.coronography': ['schemas/*.yaml', 'data/*.fits',
                                     'data/*.txt', 'data/__init__.py'],
                  'miriteam.spectroscopy': ['schemas/*.yaml', 'data/*.fits',
                                      'data/*.txt', 'data/__init__.py'],
                  'miriteam.spectroscopy.lrs': ['schemas/*.yaml', 'data/*.fits',
                                      'data/*.txt', 'data/__init__.py'],
                  'miriteam.spectroscopy.mrs': ['schemas/*.yaml', 'data/*.fits',
                                      'data/*.txt', 'data/__init__.py'],
                  'miriteam.jwst_pipeline': ['schemas/*.yaml', 'data/*.fits',
                                             'data/*.txt', 'data/__init__.py'],
                  'miriteam.jwst_pipeline.steps': ['schemas/*.yaml', 'data/*.fits',
                                                   'data/*.txt', 'data/__init__.py'],
                  'miriteam.jwst_pipeline.pipelines': ['schemas/*.yaml', 'data/*.fits',
                                                       'data/*.txt', 'data/__init__.py'],
                  'miriteam.jwst_pipeline.cfg_files': ['*.cfg'],
                 },
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    scripts=['miriteam_installation_check.py',
             'calibration_generator/darks/scripts/create_dark.py',
             'spectroscopy/lrs/scripts/lrs_srf_cal.py',
             'spectroscopy/lrs/scripts/lrs_wave_cal.py',
             'spectroscopy/mrs/scripts/mrs_extract2D.py',
             'spectroscopy/mrs/scripts/check_mrs_spextract3D.py',
            ],
    data_files=[('', ['LICENCE', 'README'])]
)
