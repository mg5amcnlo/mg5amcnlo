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

"""This is a simple script to setup a MadEvent template for use with
MadGraph 5. It performs the following actions:

1. Copy the Template and HELAS directories from a valid MG_ME
   directory in the path or given by the -d flag, and remove the
   bin/newprocess file

2. Copy all v4 model directories from the MG_ME directory as specified
   in point 2. to the models directory (with names modelname_v4)
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

print root_path

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

# Set the filepath to the MG5 directory:
filepath = MG5DIR

# If an MG_ME dir is given in the options, use that dir for Template
# and v4 models

mgme_dir = options.mgme_dir

if not mgme_dir:
    mgme_dir = MG4DIR

if not mgme_dir or not path.isdir(path.join(mgme_dir, "Template")) \
       or not path.isdir(path.join(mgme_dir, "Models")) \
       or not path.isdir(path.join(mgme_dir, "HELAS")):
    logging.error("Error: No valid MG_ME directory. Please specify using -d MG_ME_DIR")
    exit()
    
# Set Template directory path

template_path = path.join(mgme_dir, 'Template')
helas_path = path.join(mgme_dir, 'HELAS')

# 1. Copy the Template either from a valid MG_ME directory in the path
# or given by the -d flag, and remove the bin/newprocess file

logging.info("Copying " + template_path)
try:
    shutil.copytree(template_path, path.join(filepath, 'Template'), symlinks = True)
except OSError as error:
    logging.error("Error while copying Template directory: " + error.strerror)
    exit()
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

# 3. Copy all v4 model directories from the MG_ME directory as
# specified in point 2. to the models directory (with names
# modelname_v4)

model_path = path.join(mgme_dir, 'Models')

logging.info("Copying v4 models from " + model_path + ":")
for mdir in [d for d in glob.glob(path.join(model_path, "*")) \
             if path.isdir(d) and \
             path.exists(path.join(d, "particles.dat"))]:
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
        
    # Remove CVS directories
    for i in range(2):
        cvs_dirs = glob.glob(path.join(path.join(new_m_path, *(['*']*i)),
                                       'CVS'))
        if not cvs_dirs:
            break
        for cvs_dir in cvs_dirs:
            shutil.rmtree(cvs_dir)

logging.info("Done. Please enjoy MadGraph/MadEvent 5.")
