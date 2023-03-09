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

"""Unit test library for the Misc routine library in the I/O package"""

import tests.unit_tests as unittest

import madgraph.various.misc as misc

#===============================================================================
# IOMiscTest
#===============================================================================
class IOMiscTest(unittest.TestCase):
    """Test class for I/O misc function module."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parse_info_str_correct(self):
        "Test parse_info_str converts info strings to dictionaries"

        mystr = "param1 = value1\n param2=value 2\n \n"
        rightdict = {'param1':'value1', 'param2':'value 2'}

        self.assertEqual(rightdict, misc.get_pkg_info(mystr))

    def test_parse_info_str_error(self):
        "Test parse_info_str raises an error for strings which are not valid"

        mystr = "param1 : value1"

        self.assertRaises(IOError, misc.get_pkg_info, mystr)
        

