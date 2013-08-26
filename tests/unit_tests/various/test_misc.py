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
"""Test the validity of the LHE parser"""

import unittest
import madgraph.various.misc as misc

class TEST_misc(unittest.TestCase):
    
    def test_equal(self):
        
        eq = misc.equal
        
        self.assertTrue(eq(1,1.0))
        self.assertTrue(eq(1,1.0, 1))
        self.assertTrue(eq(1,1.0, 7))
        self.assertTrue(eq(1,1.0 + 1e-8, 7))
        self.assertFalse(eq(1,1.0 + 1e-8, 8))
        self.assertFalse(eq(1,1.0 + 1e-8, 9))
        self.assertFalse(eq(1,-1.0))
        self.assertTrue(eq(0 ,0e-5))
        
        self.assertTrue(eq(100,1e2))
        self.assertTrue(eq(100,1e2 + 1e-6))
        self.assertFalse(eq(100,1e2 + 1e-4))
        self.assertTrue(eq(80.419, 80.419002))
        