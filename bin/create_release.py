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

2. Copy the Template and HELAS directories, either from the present
   directory or from a valid MG_ME directory in the path or given by the
   -d flag, and remove the bin/newprocess file

   *WARNING* Note that it is your responsibility to make sure this
   Template is up-to-date!!


3. Copy all v4 model directories from the present directory and/or
   from an MG_ME directory as specified in point 2. to the models
   directory (with names modelname_v4)

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

# 0. check that all modification are commited in this directory
#    and that the date is up-to-date
diff_result = subprocess.Popen(["bzr", "diff"], stdout=subprocess.PIPE).communicate()[0] 

if diff_result:
    logging.warning("Directory is not up-to-date. The release follow the last commited version.")
    answer = raw_input('Do you want to continue anyway? (y/n)')
    if answer != 'y':
        exit()
release_date = date.fromtimestamp(time.time())
for line in file('VERSION'):
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

# 2. Copy the Template either from the present directory or from a valid
#    MG_ME directory in the path or given by the -d flag, and remove the
#    bin/newprocess file

devnull = os.open(os.devnull, os.O_RDWR)

logging.info("Getting Template from cvs")
status = subprocess.call(['cvs', 
                          '-d',
                          ':pserver:anonymous@cp3wks05.fynu.ucl.ac.be:/usr/local/CVS',
                          'checkout', '-d', 'Template', 'MG_ME/Template'],
                         stdout = devnull, stderr = devnull,
                         cwd = filepath)
if status:
    logging.error("CVS checkout failed, exiting")
    exit()

if path.exists(path.join(filepath, 'Template', 'bin', 'newprocess')):
    os.remove(path.join(filepath, 'Template', 'bin', 'newprocess'))
if path.exists(path.join(filepath, 'Template', 'bin', 'newprocess_sa')):
    os.remove(path.join(filepath, 'Template', 'bin', 'newprocess_sa'))
if path.exists(path.join(filepath, 'Template', 'bin', 'newprocess_dip')):
    os.remove(path.join(filepath, 'Template', 'bin', 'newprocess_dip'))
if path.exists(path.join(filepath, 'Template', 'bin', 'newprocess_dip_qed')):
    os.remove(path.join(filepath, 'Template', 'bin', 'newprocess_dip_qed'))

# Remove CVS directories
for i in range(6):
    cvs_dirs = glob.glob(path.join(filepath,
                                   path.join('Template', *(['*']*i)), 'CVS'))
    if not cvs_dirs:
        break
    for cvs_dir in cvs_dirs:
        shutil.rmtree(cvs_dir)

logging.info("Getting HELAS from cvs")
status = subprocess.call(['cvs', 
                          '-d',
                          ':pserver:anonymous@cp3wks05.fynu.ucl.ac.be:/usr/local/CVS',
                          'checkout', '-d', 'HELAS', 'MG_ME/HELAS'],
                         stdout = devnull, stderr = devnull,
                         cwd = filepath)
if status:
    logging.error("CVS checkout failed, exiting")
    exit()

# Remove CVS directories
for i in range(3):
    cvs_dirs = glob.glob(path.join(filepath,
                                   path.join('HELAS', *(['*']*i)), 'CVS'))
    if not cvs_dirs:
        break
    for cvs_dir in cvs_dirs:
        shutil.rmtree(cvs_dir)

# 3. Copy all v4 model directories from the present directory and/or
#    from an MG_ME directory as specified in point 2. to the models
#    directory (with names modelname_v4)

model_path = ""

logging.info("Getting Models from cvs")
status = subprocess.call(['cvs', 
                          '-d',
                          ':pserver:anonymous@cp3wks05.fynu.ucl.ac.be:/usr/local/CVS',
                          'checkout', '-d', 'Models_new', 'MG_ME/Models'],
                         stdout = devnull, stderr = devnull,
                         cwd = filepath)
if status:
    logging.error("CVS checkout failed, exiting")
    exit()

model_path = os.path.join(filepath, "Models_new")

logging.info("Copying v4 models from " + model_path + ":")
nmodels = 0
for mdir in [d for d in glob.glob(path.join(model_path, "*")) \
             if path.isdir(d) and \
             path.exists(path.join(d, "particles.dat"))]:
    modelname = path.split(mdir)[-1]
    new_m_path = path.join(filepath, 'models', modelname + "_v4")
    shutil.copytree(mdir, new_m_path)
    nmodels += 1
    if path.exists(path.join(new_m_path, "model.pkl")):
        os.remove(path.join(new_m_path, "model.pkl"))
    # Remove CVS directories
    for i in range(2):
        cvs_dirs = glob.glob(path.join(path.join(new_m_path, *(['*']*i)),
                                       'CVS'))
        if not cvs_dirs:
            break
        for cvs_dir in cvs_dirs:
            shutil.rmtree(cvs_dir)

shutil.rmtree(os.path.join(filepath, "Models_new"))
logging.info("Copied %d v4 models." % nmodels)

# 4. Create the automatic documentation in the apidoc directory

try:
    status1 = subprocess.call(['epydoc', '--html', '-o', 'apidoc',
                           'madgraph', 'aloha',
                           os.path.join('models', '*.py')], cwd = filepath)
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

shutil.rmtree(path.join(filepath, '.bzr'))
os.remove(path.join(filepath, 'bin', 'create_release.py'))
os.remove(path.join(filepath, 'bin', 'setup_madevent_template.py'))
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

## Need a __init__ file to run tests
##if not os.path.exists(os.path.join(filepath,'__init__.py')):
##    open(os.path.join(filepath,'__init__.py'), 'w').close()

## For acceptance tests, make sure to use filepath as MG4DIR
##madgraph.MG4DIR = os.path.realpath(filepath)
#madgraph.MG5DIR = os.path.realpath(filepath)

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

if not a_test_results.failures and not test_results.failures:
    logging.info("All good. Removing temporary %s directory." % filepath)
    shutil.rmtree(filepath)
else:
    logging.error("Some failures - please check before using release file")

logging.info("Thanks for creating a release.")
