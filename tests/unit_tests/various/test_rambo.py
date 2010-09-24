################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
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

import madgraph.various.rambo as rambo
import tests.unit_tests as unittest


class test_rambo(unittest.TestCase):
    """ Test the Rambo module """
    
    def setUp(self):
        self.m2_zero = rambo.FortranList(2)
        self.m1 = rambo.FortranList(1)
        self.m1[1] = 100
        self.m2 = rambo.FortranList(2)
        self.m2[1] , self.m2[2] = 100, 200
        
    def test_rambo_validity_check(self):
        """test if it raise error if wrong input"""

        # not enough energy
        self.assertRaises(rambo.RAMBOError, rambo.RAMBO, 2,150,self.m2)
        
        # not valid mass
        self.assertRaises(AssertionError, rambo.RAMBO, 2,1500,[1,2])
        
        # not valid mass
        self.assertRaises(AssertionError, rambo.RAMBO, 2,1500,self.m1)
    
        # at least 2 particles in final state
        self.assertRaises(AssertionError, rambo.RAMBO, 1, 1500, self.m1)

    def test_massless(self):
        """ Rambo can generate impulsion for massless final state """
        
        P, wgt = rambo.RAMBO(2,150, self.m2_zero)
        for i in range(1,3):
            self.assertAlmostEqual(0., P[(4,i)]**2 - P[(1,i)]**2 - P[(2,i)]**2 - P[(3,i)]**2)
         
    def test_massivecase(self):
        """ Rambo can generate impulsion for massive final state """
        
        P, wgt = rambo.RAMBO(2,500, self.m2)
        for i in range(1,3):
            self.assertAlmostEqual(self.m2[i]**2, P[(4,i)]**2 - P[(1,i)]**2 - P[(2,i)]**2 - P[(3,i)]**2)
        
        
        
        
     
