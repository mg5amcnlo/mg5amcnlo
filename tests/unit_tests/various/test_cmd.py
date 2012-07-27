################################################################################
#
# Copyright (c) 2012 The MadGraph Development team and Contributors
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

import os

from madgraph import MG5DIR
import tests.unit_tests as unittest

class TestInstall(unittest.TestCase):
    """Check the class linked to a block of the param_card"""
    
    def test_install_update(self):
        """Check that the install update command point to the official link
        and not to the test one."""
        check1 = "            filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/mg5_build_nb')\n"
        check2 = "                    filetext = urllib.urlopen('http://madgraph.phys.ucl.ac.be/patch/build%s.patch' %(i+1))\n" 
        
        has1, has2 = False, False
        for line in  open(os.path.join(MG5DIR,'madgraph','interface',
                                                      'madgraph_interface.py')):
            if 'http' in line:
                print [line]
            if line == check1:
                has1 = True
            elif line ==check2:
                has2 = True
        self.assertTrue(has1, "The install update command point through the wrong path")
        self.assertTrue(has2, "The install update command point through the wrong path")