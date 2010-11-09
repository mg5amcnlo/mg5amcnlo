#! /usr/bin/env python

################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################

"""This is a simple script to create a release for MadGraph 5, based
on the latest Bazaar commit of the present version. It performs the
following actions:

1. bzr branch the present directory to a new directory
   MadGraph5_vVERSION

4. Create the automatic documentation in the apidoc directory

5. Remove the .bzr directory

6. tar the MadGraph5_vVERSION directory.
"""

import sys

if not sys.version_info[0] == 2 or sys.version_info[1] < 6:
    sys.exit('MadGraph 5 works only with python 2.6 or later (but not python 3.X).\n\
               Please upgrate your version of python.')

import glob
import optparse
import logging
import logging.config

import os
import os.path as path
import shutil
import subprocess

# Get the parent directory (mg root) of the script real path (bin)
# and add it to the current PYTHONPATH

root_path = path.split(path.dirname(path.realpath( __file__ )))[0]
sys.path.append(root_path)

import madgraph.iolibs.misc as misc
from madgraph import MG4DIR, MG5DIR

# Write out nice usage message if called with -h or --help
usage = "usage: %prog [options] [FILE] "
parser = optparse.OptionParser(usage=usage)
parser.add_option("-l", "--logging", default='INFO',
                  help="logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL) [%default]")
(options, args) = parser.parse_args()
if len(args) == 0:
    args = ''

# Set logging level according to the logging level given by options
logging.basicConfig(level=vars(logging)[options.logging],
                    format="%(message)s")
    
# 1. bzr branch the present directory to a new directory
#    MadGraph5_vVERSION

filepath = "MadGraph5_v" + misc.get_pkg_info()['version']
if path.exists(filepath):
    logging.info("Removing existing directory " + filepath)
    shutil.rmtree(filepath)

logging.info("Branching " + MG5DIR + " to directory " + filepath)
status = subprocess.call(['bzr', 'branch', MG5DIR, filepath])
if status:
    logging.error("Script stopped")
    exit()

# 2. Create the automatic documentation in the apidoc directory

try:
    status1 = subprocess.call(['epydoc', '--html', '-o', 'apidoc',
                               'madgraph', 'aloha',
                               os.path.join('models', '*.py')], cwd = filepath)
except:
    info.error("Error while trying to run epydoc. Do you have it installed?")
    info.error("Execution cancelled.")
    sys.exit()

if status1:
    info.warning('Non-0 exit code %d from epydoc. Please check output.' % \
                 status)

# 3. Remove the .bzr directory and the create_release.py file,
#    take care of README files.

shutil.rmtree(path.join(filepath, '.bzr'))
os.remove(path.join(filepath, 'bin', 'create_release.py'))
os.remove(path.join(filepath, 'README.developer'))
shutil.move(path.join(filepath, 'README.release'), path.join(filepath, 'README'))

# 6. tar the MadGraph5_vVERSION directory.

logging.info("Create the tar file " + filepath + ".tar.gz")

status2 = subprocess.call(['tar', 'czf', filepath + ".tar.gz", filepath])

if status2:
    logging.warning('Non-0 exit code %d from tar. Please check result.' % \
                 status)

if not status1 and not status2:
    # Remove the MadGraph5_vVERSION directory
    logging.info("Removing directory " + filepath)
    shutil.rmtree(filepath)
    
logging.info("Thanks for creating a release.")
logging.info("*Please* make sure that you used the latest version")
logging.info("  of the MG/ME Template and models!")
logging.info("*Please* check the output log above to make sure that")
logging.info("  all copied directories are the ones you intended!")
logging.info("*Please* untar the release tar.gz and run the acceptance ")
logging.info("  tests before uploading the release to Launchpad!")
logging.info("  Syntax: python tests/test_manager.py -p A")
