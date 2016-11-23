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

"""Acceptance test library for the sanity of current MG5aMC version"""

import StringIO
import copy
import fractions
import os 
import sys
import tempfile
import glob
import shutil
import subprocess
import urllib

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import tests.IOTests as IOTests
import madgraph.interface.master_interface as MGCmd

import madgraph.fks.fks_common as fks_common
from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir,
                                'input_files')

pjoin = os.path.join

#===============================================================================
# IOExportFKSTest
#===============================================================================
class TestMG5aMCDistribution(unittest.TestCase):
        """Test class for the sanity of current MG5aMC distribution."""

        @staticmethod
        def get_data(link,curr_dir):
            """ Pulls up a tarball f rom the web """
            if sys.platform == "darwin":
                program = ["curl","-OL"]
            else:
                program = ["wget"]
            # Here shell=True is necessary. It is safe however since program and link are not
            
            subprocess.call(program+[link],cwd=curr_dir)
            return pjoin(curr_dir,os.path.basename(link))

        def test_short_OfflineHEPToolsInstaller(self):
            """ Test whether the current OfflineHEPToolsInstaller is up to date."""
            
            with misc.TMP_directory() as tmp_path:
                subprocess.call('bzr branch lp:~maddevelopers/mg5amcnlo/HEPToolsInstallers BZR_VERSION',
                                cwd=tmp_path, shell=True)
#                shutil.copy(pjoin(MG5DIR,'vendor','OfflineHEPToolsInstaller.tar.gz'),
#                            pjoin(tmp_path,'OfflineHEPToolsInstaller.tar.gz'))    
#                subprocess.call('tar -xzf OfflineHEPToolsInstaller.tar.gz', cwd=tmp_path, shell=True)
#                shutil.move(pjoin(tmp_path,'HEPToolsInstallers'),pjoin(tmp_path,'OFFLINE_VERSION'))
                online_path = dict(tuple(line.split()[:2]) for line in urllib.urlopen(
                      'http://madgraph.phys.ucl.ac.be/package_info.dat'))['HEPToolsInstaller']
                subprocess.call('tar -xzf %s'%TestMG5aMCDistribution.get_data(online_path,tmp_path), 
                                                                            cwd=tmp_path, shell=True)                
                shutil.move(pjoin(tmp_path,'HEPToolsInstallers'),pjoin(tmp_path,'ONLINE_VERSION_UCL'))
                online_path = dict(tuple(line.split()[:2]) for line in urllib.urlopen(
                      'http://madgraph.hep.uiuc.edu/package_info.dat'))['HEPToolsInstaller']
                subprocess.call('tar -xzf %s'%TestMG5aMCDistribution.get_data(online_path,tmp_path), 
                                                                            cwd=tmp_path, shell=True)                
                shutil.move(pjoin(tmp_path,'HEPToolsInstallers'),pjoin(tmp_path,'ONLINE_VERSION_UIUC'))                
                for path in misc.glob(pjoin('BZR_VERSION','*'),tmp_path):
                    if os.path.basename(path)=='.bzr':
                        continue
                    file_name = os.path.basename(path)
#                    for comparison in ['OFFLINE_VERSION','ONLINE_VERSION_UCL','ONLINE_VERSION_UIUC']:
                    for comparison in ['ONLINE_VERSION_UCL','ONLINE_VERSION_UIUC']:
#                        misc.sprint('Testing %s in %s vs %s.'%(file_name,'BZR_VERSION',comparison))
                        diff = subprocess.Popen('diff %s %s'%(path,
                                pjoin(tmp_path,comparison,file_name)),
                                cwd=tmp_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        diff = diff.communicate()[0]
                        self.assertEqual(diff,'',
                            'Comparison of HEPToolsInstallers | %s vs %s | %s failed.\n'%('BZR_VERSION',comparison,file_name)+
                            "Consider updating MG servers and '%s'."%pjoin(MG5DIR,'vendor','OfflineHEPToolsInstaller.tar.gz'))

        def test_short_OfflineToolsTarballs(self):
            """ Test whether the current Offline Ninja+oneloop+collier tarball is up to date."""
        
            test_tarballs = [
                    ('ninja','https://bitbucket.org/peraro/ninja/downloads/ninja-latest.tar.gz',pjoin(MG5DIR,'vendor','ninja.tar.gz')),
                    ('collier','http://collier.hepforge.org/collier-latest.tar.gz',pjoin(MG5DIR,'vendor','collier.tar.gz')),
                    ('oneloop','http://helac-phegas.web.cern.ch/helac-phegas/tar-files/OneLOop-3.6.tgz',pjoin(MG5DIR,'vendor','oneloop.tar.gz'))]
            with misc.TMP_directory() as tmp_path:
                for (name, online_path, local_path)  in test_tarballs:
                    diff = subprocess.Popen('diff %s %s'%(TestMG5aMCDistribution.get_data(online_path,tmp_path),local_path),
                                cwd=tmp_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    diff = diff.communicate()[0]
                    self.assertEqual(diff,'',"Comparison of the online and offline tarball '%s' failed. Consider updating it."%local_path)                
