################################################################################
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
################################################################################

import unittest

TestLoader = unittest.TestLoader

class TestCase(unittest.TestCase):
    """Test Case with smarter self.assertraise in optimize mode"""
  
    def assertRaises(self, error, *opt):
        """ smarter self.assertraise in optimize mode"""
        if not __debug__:
            if error == AssertionError:
                return
        unittest.TestCase.assertRaises(self, error, *opt)