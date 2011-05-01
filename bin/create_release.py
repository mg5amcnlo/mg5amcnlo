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
import time

import os
import os.path as path
import shutil
import subprocess

from datetime import date

# Get the parent directory (mg root) of the script real path (bin)
# and add it to the current PYTHONPATH

root_path = path.split(path.dirname(path.realpath( __file__ )))[0]
sys.path.append(root_path)

import madgraph.iolibs.misc as misc
from madgraph import MG5DIR

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

# 0. check that all modification are commited in this directory
#    and that the date is up-to-date
diff_result = subprocess.Popen(["bzr", "diff"], stdout=subprocess.PIPE).communicate()[0] 

if diff_result:
    logging.warning("Directory is not up-to-date. The release follow the last commited version.")
    answer = raw_input('Do you want to continue anyway? (y/n)')
    if answer != 'y':
        exit()
release_date = date.fromtimestamp(time.time())
for line in file(os.path.join(MG5DIR,'VERSION')):
    if 'version' in line:
        logging.info(line)
    if 'date' in line:
        if not str(release_date.year) in line or not str(release_date.month) in line or \
                                                           not str(release_date.day) in line:
            logging.warning("WARNING: The release time information is : %s" % line)
            answer = raw_input('Do you want to continue anyway? (y/n)')
            if answer != 'y':
                exit()

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

# 1. Remove the .bzr directory and clean bin directory file,
#    take care of README files.

shutil.rmtree(path.join(filepath, '.bzr'))
for data in glob.glob(path.join(filepath, 'bin', '*')):
    if not data.endswith('mg5'):
        os.remove(data)
os.remove(path.join(filepath, 'README.developer'))
shutil.move(path.join(filepath, 'README.release'), path.join(filepath, 'README'))

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

# 3. tar the MadGraph5_vVERSION directory.

logging.info("Create the tar file " + filename)

status2 = subprocess.call(['tar', 'czf', filename, filepath])

if status2:
    logging.error('Non-0 exit code %d from tar. Please check result.' % \
                 status)
    exit()

logging.info("Running tests on directory %s", filepath)


logging.config.fileConfig(os.path.join(root_path,'tests','.mg5_logging.conf'))
logging.root.setLevel(eval('logging.CRITICAL'))
logging.getLogger('madgraph').setLevel(eval('logging.CRITICAL'))
logging.getLogger('cmdprint').setLevel(eval('logging.CRITICAL'))
logging.getLogger('tutorial').setLevel(eval('logging.CRITICAL'))

# Change path to use now only the directory comming from bzr
sys.path.insert(0, os.path.realpath(filepath))
import tests.test_manager as test_manager

# reload from the bzr directory the element loaded here (otherwise it's 
#mixes the path for the tests
import madgraph
reload(madgraph)
import madgraph.iolibs
reload(madgraph.iolibs)
import madgraph.iolibs.misc
reload(madgraph.iolibs.misc)


test_results = test_manager.run(package=os.path.join('tests',
                                                     'unit_tests'))

a_test_results = test_manager.run(package=os.path.join('tests',
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

try:
    os.remove("%s.asc" % filename)
except:
    pass

try:
    status1 = subprocess.call(['gpg', '--armor', '--sign', '--detach-sig',
                               filename])
    if status1 == 0:
        logging.info("gpg signature file " + filename + ".asc created")
except:
    logging.warning("Call to gpg to create signature file failed. " +\
                    "Please install and run\n" + \
                    "gpg --armor --sign --detach-sig " + filename)


if not a_test_results.failures and not test_results.failures:
    logging.info("All good. Removing temporary %s directory." % filepath)
    shutil.rmtree(filepath)
else:
    logging.error("Some failures - please check before using release file")

logging.info("Thanks for creating a release.")
