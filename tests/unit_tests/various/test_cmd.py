################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
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

from __future__ import absolute_import
from __future__ import print_function
import os

from madgraph import MG5DIR
import tests.unit_tests as unittest
import madgraph.various.misc as misc

pjoin = os.path.join

class TestInstall(unittest.TestCase):
    """Check the class linked to a block of the param_card"""
    
    def test_install_update(self):
        """Check that the install update command point to the official link
        and not to the test one."""
        
        checklts1 = "http://madgraph.physics.illinois.edu/mg5amc_build_nb"
        checklts2 = "http://madgraph.physics.illinois.edu/patch/build%s.patch" 
        check_dev1 = "http://madgraph.phys.ucl.ac.be/mg5amc3_build_nb"
        check_dev2 = "http://madgraph.phys.ucl.ac.be/patch/build%s.patch" 
        
        to_search = [ checklts1 ,  checklts2, check_dev1, check_dev2]
        found = [False, False, False, False]
        for line in  open(os.path.join(MG5DIR,'madgraph','interface',
                                                      'madgraph_interface.py')):
            for i,tocheck in enumerate(to_search):
                if tocheck in line:
                    found[i] = True

        version = misc.get_pkg_info()['version']
        if version.startswith('2'): # current LTS
            self.assertTrue(found[0])
            self.assertTrue(found[1])
            self.assertFalse(found[2])
            self.assertFalse(found[3])
        else:
            self.assertFalse(found[0])
            self.assertFalse(found[1])
            self.assertTrue(found[2])
            self.assertTrue(found[3])



        
    def test_configuration_file(self):
        """Check that the configuration file is not modified, if he is present"""
        
        #perform this test only for .bzr repository
        if not os.path.exists(pjoin(MG5DIR, '.bzr')):
            return
        if not os.path.exists(pjoin(MG5DIR, 'input','mg5_configuration.txt')):
            return        
        
        text1 = open(pjoin(MG5DIR,'input','.mg5_configuration_default.txt')).read()
        text2 = open(pjoin(MG5DIR,'input','mg5_configuration.txt')).read()
        warning = """WARNING: Your file mg5_configuration.txt and .mg5_configuration_default.txt
        are different. This probably fine but please check it before any release."""
        if text1 != text2:
            print(warning)
        
