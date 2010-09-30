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
parser.add_option("-d", "--mgme_dir", default='', dest = 'mgme_dir',
                  help="Use MG_ME directory MGME_DIR")
(options, args) = parser.parse_args()
if len(args) == 0:
    args = ''

# Set logging level according to the logging level given by options
logging.basicConfig(level=vars(logging)[options.logging],
                    format="%(message)s")

# If an MG_ME dir is given in the options, use that dir for Template
# and v4 models

mgme_dir = options.mgme_dir

# Set Template directory path

if mgme_dir and path.isdir(path.join(mgme_dir, 'Template')) \
   and path.isdir(path.join(mgme_dir, 'HELAS')):
    template_path = path.join(mgme_dir, 'Template')
    helas_path = path.join(mgme_dir, 'Helas')
elif MG4DIR and path.isdir(path.join(MG4DIR, 'Template')) \
    and path.isdir(path.join(MG4DIR, 'HELAS')):
    template_path = path.join(MG4DIR, 'Template')
    helas_path = path.join(MG4DIR, 'HELAS')
else:
    logging.error("Error: Could not find the Template directory in the path")
    logging.error("       Please supply MG_ME path using the -d option")
    exit()


    
# 1. bzr branch the present directory to a new directory
#    MadGraph5_vVERSION

filepath = "MadGraph5_v" + misc.get_pkg_info()['version']

logging.info("Branching " + MG5DIR + " to directory " + filepath)
status = subprocess.call(['bzr', 'branch', MG5DIR, filepath])
if status == 3:
    logging.error("Please run rm -rf "+filepath)
    exit()
elif status:
    logging.error("Script stopped")
    exit()

# 2. Copy the Template either from the present directory or from a valid
#    MG_ME directory in the path or given by the -d flag, and remove the
#    bin/newprocess file

logging.info("Copying " + template_path)
shutil.copytree(template_path, path.join(filepath, 'Template'), symlinks = True)
if path.exists(path.join(filepath, 'Template', 'bin', 'newprocess')):
    os.remove(path.join(filepath, 'Template', 'bin', 'newprocess'))
# Remove CVS directories
for i in range(6):
    cvs_dirs = glob.glob(path.join(filepath,
                                   path.join('Template', *(['*']*i)), 'CVS'))
    if not cvs_dirs:
        break
    for cvs_dir in cvs_dirs:
        shutil.rmtree(cvs_dir)
logging.info("Copying " + helas_path)
try:
    shutil.copytree(helas_path, path.join(filepath, 'HELAS'), symlinks = True)
except OSError as error:
    logging.error("Error while copying HELAS directory: " + error.strerror)
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
if mgme_dir:
    if path.isdir(path.join(mgme_dir, 'Models')):
        model_path = path.join(mgme_dir, 'Models')
elif MG4DIR and path.isdir(path.join(MG4DIR, 'Models')):
    model_path = path.join(MG4DIR, 'Models')

if model_path:
    logging.info("Copying v4 models from " + model_path + ":")
    for mdir in [d for d in glob.glob(path.join(model_path, "*")) \
                 if path.isdir(d) and \
                 path.exists(path.join(d, "particles.dat"))]:
        modelname = path.split(mdir)[-1]
        new_m_path = path.join(filepath, 'models', modelname + "_v4")
        logging.info(mdir + " -> " + new_m_path)
        shutil.copytree(mdir, new_m_path)
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

else:
    v4_models = [d for d in glob.glob(path.join(MG5DIR, "models", "*_v4")) \
                 if path.isdir(d) and \
                 path.exists(path.join(d, "particles.dat"))]

    logging.info("Copying v4 models from " + path.join(MG5DIR, "models") + ":")
    for mdir in v4_models:
        modelname = path.split(mdir)[-1]
        new_m_path = path.join(filepath, 'models', modelname + "_v4")
        try:
            shutil.copytree(mdir, new_m_path)
            logging.info(mdir + " -> " + new_m_path)
        except OSError:
            logging.warning("Directory " + new_m_path + \
                            " already exists, not copied.")
        if path.exists(path.join(new_m_path, "model.pkl")):
            os.remove(path.join(new_m_path, "model.pkl"))
        if path.exists(path.join(new_m_path, "model.pkl")):
            os.remove(path.join(new_m_path, "model.pkl"))
    if not v4_models:
        logging.info("No v4 models in " + path.join(MG5DIR, "models"))

# 4. Create the automatic documentation in the apidoc directory

status1 = subprocess.call(['epydoc', '--html', '-o', 'apidoc', 'madgraph',
                          os.path.join('models', '*.py')], cwd = filepath)

if status1:
    info.warning('Non-0 exit code %d from epydoc. Please check output.' % \
                 status)

# 5. Remove the .bzr directory and the create_release.py file,
#    take care of README files.

shutil.rmtree(path.join(filepath, '.bzr'))
os.remove(path.join(filepath, 'bin', 'create_release.py'))
os.remove(path.join(filepath, 'bin', 'setup_madevent_template.py'))
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
