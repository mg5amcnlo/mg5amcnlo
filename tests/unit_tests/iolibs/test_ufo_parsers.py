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

"""Unit test library for the UFO parsing"""

import tests.unit_tests as unittest

import madgraph.iolibs.ufo_expression_parsers as parsers

#===============================================================================
# IOMiscTest
#===============================================================================
class UFOParserTest(unittest.TestCase):
    """Test class for ufo_expression_parsers.py"""

    def setUp(self):
        if not hasattr(self, 'calc'):
            self.calc = parsers.UFOExpressionParserFortran()

    def tearDown(self):
        pass

    def test_parse_fortran_IfElseStruct(self):
        "Test that structures like ( 1 if 2==3 else 4)"
        
        tests = {('(1 if a==0 else 1/a)',
                         'COND(A.EQ.0.000000d+00,1.000000d+00,1.000000d+00/A)'),
                 ('(1/a if a else 1)',
                         'COND(A.NE.0.000000d+00,1.000000d+00/A,1.000000d+00)'),
                 ('(1 if a<=0 else 1/a)',
                        'COND(A.LEQ.0.000000d+00,1.000000d+00,1.000000d+00/A)'),
                 ('(1 if a<0 else 1/a)',
                         'COND(A.LE.0.000000d+00,1.000000d+00,1.000000d+00/A)'),
                 ('((1) if (a<0) else (1/a))',
                 '(COND((A.LE.0.000000d+00),(1.000000d+00),(1.000000d+00/A)))'),
                 ('((2 if b==0 else 1/b) if a==0 else 1/a)',
 'COND(A.EQ.0,COND(B.EQ.0.000000d+00,2.000000d+00,1.000000d+00/B),1.000000d+00/A)'),
                 ('(1 if a==0 else (1/A if b==0 else 1/b))',
 'COND(A.EQ.0.000000d+00,1.000000d+00,COND(B.EQ.0.000000d+00,1.000000d+00/A,1.000000d+00/B))'),
                 ('(1 if a==0 and b==1 else 1/a)',
  'COND(A.EQ.0.000000d+00.AND.B.EQ.0.000000d+00,1.000000d+00,1.000000d+00/A)'),
                 }
        for toParse, sol in tests:
            self.assertEqual(self.calc.parse(toParse), sol)

    def test_parse_info_str_error(self):
        "Test parse_info_str raises an error for strings which are not valid"

        mystr = "param1 : value1"

        self.assertRaises(IOError, misc.get_pkg_info, mystr)
        

