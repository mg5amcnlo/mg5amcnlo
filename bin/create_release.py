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

filepath = "MadGraph5_v" + misc.get_pkg_info()['version'].replace(".", "_")
filename = "MadGraph5_v" + misc.get_pkg_info()['version'] + ".tar.gz"
if path.exists(filepath):
    logging.info("Removing existing directory " + filepath)
    shutil.rmtree(filepath)

logging.info("Branching " + MG5DIR + " to directory " + filepath)
status = subprocess.call(['bzr', 'branch', MG5DIR, filepath])
if status:
    logging.error("bzr branch failed. Script stopped")
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
    logging.error('Non-0 exit code %d from epydoc. Please check output.' % \
                 status)
    exit()

# 3. Remove the .bzr directory and the create_release.py file,
#    take care of README files.

shutil.rmtree(path.join(filepath, '.bzr'))
os.remove(path.join(filepath, 'bin', 'create_release.py'))
os.remove(path.join(filepath, 'README.developer'))
shutil.move(path.join(filepath, 'README.release'), path.join(filepath, 'README'))

# 6. tar the MadGraph5_vVERSION directory.

logging.info("Create the tar file " + filename)

status2 = subprocess.call(['tar', 'czf', filename, filepath])

if status2:
    logging.error('Non-0 exit code %d from tar. Please check result.' % \
                 status)
    exit()

logging.info("Running tests on directory %s", filepath)

try:
    logging.config.fileConfig(os.path.join(root_path,'tests','.mg5_logging.conf'))
    logging.root.setLevel(eval('logging.CRITICAL'))
    logging.getLogger('madgraph').setLevel(eval('logging.CRITICAL'))
    logging.getLogger('cmdprint').setLevel(eval('logging.CRITICAL'))
    logging.getLogger('tutorial').setLevel('CRITICAL')
except:
    pass

sys.path.insert(0, os.path.realpath(filepath))
import tests.test_manager as test_manager

# Need a __init__ file to run tests
if not os.path.exists(os.path.join(filepath,'__init__.py')):
    open(os.path.join(filepath,'__init__.py'), 'w').close()

# For acceptance tests, make sure to use filepath as MG4DIR

import madgraph

madgraph.MG4DIR = os.path.realpath(filepath)
madgraph.MG5DIR = os.path.realpath(filepath)

test_results = test_manager.run(package=os.path.join(filepath,'tests',
                                                     'unit_tests'))

a_test_results = test_manager.run(package=os.path.join(filepath,'tests',
                                                       'acceptance_tests'),
                                  )
# Set logging level according to the logging level given by options
logging.basicConfig(level=vars(logging)[options.logging],
                    format="%(message)s")
logging.root.setLevel(vars(logging)[options.logging])

if not test_results.wasSuccessful():
    logging.error("Failed %d unit tests, please check!" % \
                    (len(test_results.errors) + len(test_results.failures)))

if not a_test_results.wasSuccessful():
    logging.error("Failed %d acceptance tests, please check!" % \
                  (len(a_test_results.errors) + len(a_test_results.failures)))

if a_test_results.errors or test_results.errors:
    logging.error("Removing %s and quitting..." % filename)
    os.remove(filename)
    exit()

if not a_test_results.failures and not test_results.failures:
    logging.info("All good. Removing temporary %s directory." % filepath)
    shutil.rmtree(filepath)
else:
    logging.error("Some failures - please check before using release file")

logging.info("Thanks for creating a release.")
