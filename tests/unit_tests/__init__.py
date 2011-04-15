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
import math

TestLoader = unittest.TestLoader

class TestCase(unittest.TestCase):
    """Test Case with smarter self.assertraise in optimize mode"""

    maxDiff = None
    def assertRaises(self, error, *opt):
        """ smarter self.assertraise in optimize mode"""
        if not __debug__:
            if error == AssertionError:
                return
        unittest.TestCase.assertRaises(self, error, *opt)

    def assertAlmostEqual(self, a, b, *opt, **arg):
        """Redefine the stupid unittest.assertAlmostEqual to act on
        significance instead of decimal places"""

        places = 7
        if opt:
            places = opt[0]

        if 'places' in arg:
            places = arg['places']

        if 'msg' not in arg and len(opt) < 2:
            arg['msg'] = '%s != %s within %s digits' % (a, b, places )

        magarg = a or b
        if not magarg or a - b == 0:
            # Both a and b are 0 or they are identical
            return
        if not (a and b):
            # One of a or b is 0
            unittest.TestCase.assertAlmostEqual(self, a, b, *opt, **arg)
            return
        # Compare significant digits
        magnitude = math.floor(math.log10(abs(magarg))) + 1
        unittest.TestCase.assertAlmostEqual(self, a/10**magnitude,
                                            b/10**magnitude, *opt, **arg)
        
