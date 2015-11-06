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

import os

from madgraph import MG5DIR
import tests.unit_tests as unittest

pjoin = os.path.join

class TestInstall(unittest.TestCase):
    """Check the class linked to a block of the param_card"""
    
    def test_install_update(self):
        """Check that the install update command point to the official link
        and not to the test one."""
        check1 = "            filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/mg5amc_build_nb')\n"
        check2 = "                    filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/patch/build%s.patch' %(i+1))\n" 
        
        has1, has2 = False, False
        for line in  open(os.path.join(MG5DIR,'madgraph','interface',
                                                      'madgraph_interface.py')):
            if line == check1:
                has1 = True
            elif line ==check2:
                has2 = True
        self.assertTrue(has1, "The install update command point through the wrong path")
        self.assertTrue(has2, "The install update command point through the wrong path")
        
        
    def test_configuration_file(self):
        """Check that the configuration file is not modified, if he is present"""
        
        #perform this test only for .bzr repository
        if not os.path.exists(pjoin(MG5DIR, '.bzr')):
            return
        if not os.path.exists(pjoin(MG5DIR, 'input','mg5_configuration.txt')):
            return        
        
        text1 = file(pjoin(MG5DIR,'input','.mg5_configuration_default.txt')).read()
        text2 = file(pjoin(MG5DIR,'input','mg5_configuration.txt')).read()
        warning = """WARNING: Your file mg5_configuration.txt and .mg5_configuration_default.txt
        are different. This probably fine but please check it before any release."""
        if text1 != text2:
            print warning
        