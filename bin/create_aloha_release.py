#! /usr/bin/env python

################################################################################
#
# Copyright (c) 2011 The ALOHA Development team and Contributors
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

"""This is a simple script to create a release for ALOHA, based
on the latest Bazaar commit of the present version. It performs the
following actions:

1. bzr branch the present directory to a new directory
   ALOHA_vVERSION

3. Copy all Madgraph5 files needed by ALOHA to the present directory

4. Create the automatic documentation in the apidoc directory

5. Remove the .bzr directory

6. tar the ALOHA_vVERSION directory.
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

import madgraph.various.misc as misc
import madgraph.iolibs.files as files_routines
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
#    ALOHA_vVERSION

filepath = "ALOHA_v" + misc.get_pkg_info()['version'].replace(".", "_")
filename = "ALOHA_v" + misc.get_pkg_info()['version'] + ".tar.gz"
if path.exists(filepath):
    logging.info("Removing existing directory " + filepath)
    shutil.rmtree(filepath)

logging.info("Branching " + MG5DIR + "/aloha to directory " + filepath)
status = subprocess.call(['bzr', 'branch', MG5DIR, filepath])
if status:
    logging.error("bzr branch failed. Script stopped")
    exit()

# 2. Copy MG5 files needed by ALOHA + remove pointless direcoty
devnull = os.open(os.devnull, os.O_RDWR)
logging.info("copying files")
requested_files=['madgraph/iolibs/files.py','madgraph/iolibs/file_writers.py']

for fname in requested_files:
    files_routines.cp(MG5DIR +'/'+ fname, filepath+'/aloha')

for fname in os.listdir(filepath):
    if fname in ['aloha','VERSION']:
        continue
    os.system('rm -rf %s ' % os.path.join(filepath,fname))


# 4. Create the automatic documentation in the apidoc directory

try:
    status1 = subprocess.call(['epydoc', '--html', '-o', 'apidoc', 'aloha'], cwd = filepath)
except:
    logging.error("Call to epydoc failed. " +\
                  "Please check that it is properly installed.")
    exit()

if status1:
    logging.error('Non-0 exit code %d from epydoc. Please check output.' % \
                 status)
    exit()

# 5. Remove the .bzr directory and the create_release.py file,
#    take care of README files.

try:
    shutil.rmtree(path.join(filepath, '.bzr'))
except:
    pass
#os.remove(path.join(filepath, 'bin', 'create_release.py'))
#os.remove(path.join(filepath, 'bin', 'setup_madevent_template.py'))
#os.remove(path.join(filepath, 'README.developer'))
shutil.move(path.join(filepath, 'aloha','README'), path.join(filepath, 'README'))
shutil.move(path.join(filepath, 'aloha','bin'), path.join(filepath, 'bin'))
# 6. tar the MadGraph5_vVERSION directory.

logging.info("Create the tar file " + filename)

status2 = subprocess.call(['tar', 'czf', filename, filepath])

if status2:
    logging.error('Non-0 exit code %d from tar. Please check result.' % \
                 status)
    exit()

logging.info("Thanks for creating a release.")
