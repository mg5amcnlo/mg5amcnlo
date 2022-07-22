#! /usr/bin/env python3

################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""This is a simple script to create a release for MadGraph5_aMC@NLO, based
on the latest Bazaar commit of the present version. It performs the
following actions:

1. bzr branch the present directory to a new directory
   MadGraph5_vVERSION

4. Create the automatic documentation in the apidoc directory -> Now tar.gz

5. Remove the .bzr directory

6. tar the MadGraph5_vVERSION directory.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys
from six.moves import range
from six.moves import input

if sys.version_info[1] < 7:
    sys.exit('MadGraph5_aMC@NLO works only with python 2.7/3.7 or later.\n\
               Please upgrate your version of python.')

import glob
import optparse
import logging
import logging.config
import time

import os
import os.path as path
import re
import shutil
import subprocess
import six.moves.urllib.request, six.moves.urllib.parse, six.moves.urllib.error

from datetime import date

# Get the parent directory (mg root) of the script real path (bin)
# and add it to the current PYTHONPATH

root_path = path.split(path.dirname(path.realpath( __file__ )))[0]
sys.path.append(root_path)
pjoin =os.path.join
import madgraph.various.misc as misc
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

# 0. check that all modification are committed in this directory
#    and that the date/UpdateNote are up-to-date
diff_result = subprocess.Popen(["bzr", "diff"], stdout=subprocess.PIPE).communicate()[0] 

if diff_result:
    logging.warning("Directory is not up-to-date. The release follow the last committed version.")
    answer = input('Do you want to continue anyway? (y/n)')
    if answer != 'y':
        exit()

release_date = date.fromtimestamp(time.time())
for line in open(os.path.join(MG5DIR,'VERSION')):
    if 'version' in line:
        logging.info(line)
        version = line.rsplit('=')[1].strip()
    if 'date' in line:
        if not str(release_date.year) in line or not str(release_date.month) in line or \
                                                           not str(release_date.day) in line:
            logging.warning("WARNING: The release time information is : %s" % line)
            answer = input('Do you want to continue anyway? (y/n)')
            if answer != 'y':
                exit()

Update_note = open(os.path.join(MG5DIR,'UpdateNotes.txt')).read()
if version not in Update_note:
    logging.warning("WARNING: version number %s is not found in \'UpdateNotes.txt\'" % version)
    answer = input('Do you want to continue anyway? (y/n)')
    if answer != 'y':
        exit()

# 1. Adding the file .revision used for future auto-update.
# Provide this only if version is not beta/tmp/...
pattern = re.compile(r'''[\d.]+$''')
if pattern.match(version):
    #valid version format
    # Get current revision number:
    p = subprocess.Popen(['bzr', 'revno'], stdout=subprocess.PIPE)
    rev_nb = p.stdout.read().strip().decode()
    logging.info('find %s for the revision number -> starting point for the auto-update' % rev_nb)  
else:
    logging.warning("WARNING: version number %s is not in format A.B.C,\n" % version +\
         "in consequence the automatic update of the code will be deactivated" )
    answer = input('Do you want to continue anyway? (y/n)')
    if answer != 'y':
        exit()
    rev_nb=None

# checking that the rev_nb is in a reasonable range compare to the old one.
if rev_nb:
    rev_nb_i = int(rev_nb)
    try:
        filetext = six.moves.urllib.request.urlopen('http://madgraph.phys.ucl.ac.be/mg5amc3_build_nb')
        text = filetext.read().decode().split('\n')
        print(text)
        web_version = int(text[0].strip())
        if text[1]:
            last_message = int(text[1].strip())
        else:
            last_message = 99
    except (ValueError, IOError):
        logging.warning("WARNING: impossible to detect the version number on the web")
        answer = input('Do you want to continue anyway? (y/n)')
        if answer != 'y':
            exit()
        web_version = -1
    else:
        logging.info('version on the web is %s' % web_version)
    if web_version +1 == rev_nb_i or web_version == -1:
        pass # this is perfect
    elif rev_nb_i in [web_version+i for i in range(1,4)]:
        logging.warning("WARNING: current version on the web is %s" % web_version)
        logging.warning("Please check that this (small difference) is expected.")
        answer = input('Do you want to continue anyway? (y/n)')
        if answer != 'y':
            exit()
    elif web_version < rev_nb_i:
        logging.warning("CRITICAL: current version on the web is %s" % web_version)
        logging.warning("This is a very large difference. Indicating a wrong manipulation.")
        logging.warning("and can creates trouble for the auto-update.")
        answer = input('Do you want to continue anyway? (y/n)')
        if answer != 'y':
            exit()
    else:
        logging.warning("CRITICAL: current version on the web is %s" % web_version)
        logging.warning("This FORBIDS any auto-update for this version.")
        rev_nb=None
        answer = input('Do you want to continue anyway? (y/n)')
        if answer != 'y':
            exit()                        
# 1. bzr branch the present directory to a new directory
#    MadGraph5_vVERSION

filepath = "MG5_aMC_v" + misc.get_pkg_info()['version'].replace(".", "_")
filename = "MG5_aMC_v" + misc.get_pkg_info()['version'] + ".tar.gz"
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
    if not data.endswith('mg5') and not data.endswith('mg5_aMC'):
        if 'compile.py' not in data:
            os.remove(data)
        else:
            os.rename(data, data.replace('compile.py','.compile.py'))

os.remove(path.join(filepath, 'README.developer'))
shutil.move(path.join(filepath, 'README.release'), path.join(filepath, 'README'))


# 1. Add information for the auto-update
if rev_nb:
    fsock = open(os.path.join(filepath,'input','.autoupdate'),'w')
    fsock.write("version_nb   %s\n" % int(rev_nb))
    fsock.write("last_check   %s\n" % int(time.time()))
    fsock.write("last_message %s\n" % int(last_message))
    fsock.close()
    
# 1. Copy the .mg5_configuration_default.txt to it's default path
shutil.copy(path.join(filepath, 'input','.mg5_configuration_default.txt'), 
            path.join(filepath, 'input','mg5_configuration.txt'))
shutil.copy(path.join(filepath, 'input','proc_card_default.dat'), 
            path.join(filepath, 'proc_card.dat'))


# 1.1 Change the trapfpe.c code to an empty file
#os.remove(path.join(filepath,'Template','NLO','SubProcesses','trapfpe.c'))
#create_empty = open(path.join(filepath,'Template','NLO','SubProcesses','trapfpe.c'),'w')
#create_empty.close()

# 2. Create the automatic documentation in the apidoc directory
#try:
#    status1 = subprocess.call(['epydoc', '--html', '-o', 'apidoc',
#                               'madgraph', 'aloha',
#                               os.path.join('models', '*.py')], cwd = filepath)
#except:
#    logging.error("Error while trying to run epydoc. Do you have it installed?")
#    logging.error("Execution cancelled.")
#    sys.exit()
#
#if status1:
#    logging.error('Non-0 exit code %d from epydoc. Please check output.' % \
#                 status)
#    sys.exit()
#if status1:
#    logging.error('Non-0 exit code %d from epydoc. Please check output.' % \
#                 status)
#    sys.exit()

#3. tarring the apidoc directory
#status2 = subprocess.call(['tar', 'czf', 'doc.tgz', 'apidoc'], cwd=filepath)

#if status2:
#    logging.error('Non-0 exit code %d from tar. Please check result.' % \
#                 status)
#    sys.exit()
#else:
    # remove the apidoc file.
#    shutil.rmtree(os.path.join(filepath,'apidoc'))

# 4. Download the offline installer and other similar code
install_str = """
cd %s
rm -rf download-temp &> /dev/null;
mkdir -v download-temp;
cd download-temp;
bzr branch lp:~maddevelopers/mg5amcnlo/HEPToolsInstallers;
rm -rfv `find HEPToolsInstallers -name .bzr -type d` > /dev/null;
tar czf OfflineHEPToolsInstaller.tar.gz HEPToolsInstallers/ > /dev/null;
mv OfflineHEPToolsInstaller.tar.gz ../vendor;
cd ..;
rm -rf download-temp;
""" % filepath
os.system(install_str)

collier_link = "http://collier.hepforge.org/collier-latest.tar.gz" 
misc.wget(collier_link, os.path.join(filepath, 'vendor', 'collier.tar.gz'))
ninja_link = "https://bitbucket.org/peraro/ninja/downloads/ninja-latest.tar.gz"
misc.wget(ninja_link, os.path.join(filepath, 'vendor', 'ninja.tar.gz'))

# Add the tarball for SMWidth
swidth_link = "http://madgraph.phys.ucl.ac.be/Downloads/SMWidth.tgz"
misc.wget(ninja_link, os.path.join(filepath, 'vendor', 'SMWidth.tar.gz')) 

if not os.path.exists(os.path.join(filepath, 'vendor', 'OfflineHEPToolsInstaller.tar.gz')):
    print('Fail to create OfflineHEPToolsInstaller')
    sys.exit()

# 5. tar the MadGraph5_vVERSION directory.

logging.info("Create the tar file " + filename)
# clean all the pyc
os.system("cd %s;find . -name '*.pyc' -delete" % filepath)
status2 = subprocess.call(['tar', 'czf', filename, filepath])
if status2:
    logging.error('Non-0 exit code %d from tar. Please check result.' % \
                 status)
    sys.exit()

try:
    status1 = subprocess.call(['gpg', '--armor', '--sign', '--detach-sig',
                               filename])
    if status1 == 0:
        logging.info("gpg signature file " + filename + ".asc created")
except:
    logging.warning("Call to gpg to create signature file failed. " +\
                    "Please install and run\n" + \
                    "gpg --armor --sign --detach-sig " + filename)



logging.info("Running tests on directory %s", filepath)
print(os.listdir(filepath))
import subprocess
status = subprocess.call([pjoin('tests', 'test_manager.py'),'-t0'],cwd=filepath)
print("status:", status)
status = subprocess.call([pjoin('tests', 'test_manager.py'),'-t0', '-pA'],cwd=filepath)
print("status:", status)
status = subprocess.call([pjoin('tests', 'test_manager.py'),'-t0','-pP' ,'test_short.*'],cwd=filepath)
print("status:", status)


logging.info("Thanks for creating a release. please check that the tests were sucessfull before releasing the version")
sys.exit()
