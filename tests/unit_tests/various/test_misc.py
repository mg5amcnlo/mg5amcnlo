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
"""Test the validity of the LHE parser"""

import unittest
import madgraph.various.misc as misc

class TEST_misc(unittest.TestCase):
    
    def test_equal(self):
        
        eq = misc.equal
        
        self.assertTrue(eq(1,1.0))
        self.assertTrue(eq(1,1.0, 1))
        self.assertTrue(eq(1,1.0, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 7))
        self.assertTrue(eq(1,1.0 - 2e-8, 7))
        self.assertFalse(eq(1,1.0 + 2e-8, 8))
        self.assertFalse(eq(1,1.0 + 2e-8, 9))
        self.assertFalse(eq(1,-1.0))
        self.assertTrue(eq(0 ,0e-5))
        
        self.assertTrue(eq(100,1e2))
        self.assertTrue(eq(100,1e2 + 1e-6))
        self.assertFalse(eq(100,1e2 + 1e-4))
        self.assertTrue(eq(80.419, 80.419002))
        
        # Check negative number
        self.assertTrue(eq(-1,-1))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 7))
        self.assertFalse(eq(-1,-1.0 + 2e-8, 8))
        self.assertFalse(eq(-1,-1.0 + 2e-8, 9))
        self.assertFalse(eq(-1,1.0))
        self.assertTrue(eq(-0 ,-0e-5))
        
        self.assertTrue(eq(-100,-1e2))
        self.assertTrue(eq(-100,-1e2 + 1e-6))
        self.assertFalse(eq(-100,-1e2 + 1e-4))
        self.assertTrue(eq(-80.419, -80.419002))
        
        # check with 0
        self.assertTrue(eq(0 ,1e-11))  
        self.assertTrue(eq(0 ,1e-8))
        self.assertTrue(eq(0 ,1e-7))
        self.assertFalse(eq(0 ,1e-6))          
        self.assertFalse(eq(0 ,1e-1))

        self.assertTrue(eq(1e-11, 0))  
        self.assertTrue(eq(1e-8, 0))
        self.assertTrue(eq(1e-7, 0))
        self.assertFalse(eq(1e-6, 0))          
        self.assertFalse(eq(1e-1, 0))
        
        self.assertFalse(eq(0 ,1e-11, zero_limit=False))  
        self.assertFalse(eq(0 ,1e-8, zero_limit=False))
        self.assertFalse(eq(0 ,1e-7, zero_limit=False))
        self.assertFalse(eq(0 ,1e-6, zero_limit=False))          
        self.assertFalse(eq(0 ,1e-1, zero_limit=False))
        
        self.assertFalse(eq(1e-11, 0, zero_limit=False))  
        self.assertFalse(eq(1e-8, 0, zero_limit=False))
        self.assertFalse(eq(1e-7, 0, zero_limit=False))
        self.assertFalse(eq(1e-6, 0, zero_limit=False))          
        self.assertFalse(eq(1e-1, 0, zero_limit=False))         