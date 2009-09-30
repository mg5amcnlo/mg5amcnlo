##############################################################################
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
##############################################################################

"""Unit test library for the various base objects of the core library"""

import unittest
import madgraph.core.base_objects as base_objects

class CoreBaseObjectsTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCreateParticleCorrect(self):
        "Test Particle object __init__, get and set"

        mydict = {'name':'t',
                  'antiname':'t',
                  'spin':2,
                  'color':3,
                  'mass':'mt',
                  'width':'wt',
                  'texname':'t',
                  'antitexname':'\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6}

        mypart1 = base_objects.Particle(mydict)
        mypart2 = base_objects.Particle()

        # First check mypart2 is None by default and fill it using set
        for prop in mydict.keys():
            self.assertEqual(mypart2.get(prop), None)
            mypart2.set(prop, mydict[prop])

        # Check equality between Particle objects
        self.assertEqual(mypart1, mypart2)

        # Check equality with initial dic using get
        for prop in mypart1._prop_list:
            self.assertEqual(mypart1.get(prop), mydict[prop])

if __name__ == "__main__":
    unittest.main()
