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
        if not hasattr(self, 'mp_calc'):
            self.mp_calc = parsers.UFOExpressionParserMPFortran()

    def tearDown(self):
        pass

    def test_parse_fortran_IfElseStruct(self):
        "Test that structures like ( 1 if 2==3 else 4)"
        
        tests = [
                 ('(1 if a==0 else 1/a)',
         '(CONDIF(a.EQ.0.000000d+00,DCMPLX(1.000000d+00),DCMPLX(1.000000d+00/a)))'),
                 ('1/a if a else 1',
        'CONDIF(DCMPLX(a).NE.(0d0,0d0),DCMPLX(1.000000d+00/a),DCMPLX(1.000000d+00))'),
                 ('1 if a<=0 else 1/a',
        'CONDIF(a.LE.0.000000d+00,DCMPLX(1.000000d+00),DCMPLX(1.000000d+00/a))'),
                 ('1 if a<0 else 1/a',
        'CONDIF(a.LT.0.000000d+00,DCMPLX(1.000000d+00),DCMPLX(1.000000d+00/a))'),
                 ('((1) if (a<0) else (1/a))',
                 '(CONDIF((a.LT.0.000000d+00),DCMPLX((1.000000d+00)),DCMPLX((1.000000d+00/a))))'),
                 ('(2 if b==0 else 1/b) if a==0 else 1/a',
 'CONDIF(a.EQ.0.000000d+00,DCMPLX((CONDIF(b.EQ.0.000000d+00,DCMPLX(2.000000d+00),DCMPLX(1.000000d+00/b)))),DCMPLX(1.000000d+00/a))'),
                 ('1 if a==0 else (1/A if b==0 else 1/b)',
 'CONDIF(a.EQ.0.000000d+00,DCMPLX(1.000000d+00),DCMPLX((CONDIF(b.EQ.0.000000d+00,DCMPLX(1.000000d+00/a),DCMPLX(1.000000d+00/b)))))'),
                 ('1 if a==0 and b==1 else 1/a',
  'CONDIF(a.EQ.0.000000d+00.AND.b.EQ.1.000000d+00,DCMPLX(1.000000d+00),DCMPLX(1.000000d+00/a))'),
                  ('1+3*5 if a else 8*3+6',
    'CONDIF(DCMPLX(a).NE.(0d0,0d0),DCMPLX(1.000000d+00+3.000000d+00*5.000000d+00),DCMPLX(8.000000d+00*3.000000d+00+6.000000d+00))'),
                ('( (complex(0,1)*G**3)/(48.*cmath.pi**2) if MT else 0 )',
                  '(CONDIF(DCMPLX(mt).NE.(0d0,0d0),DCMPLX(((0.000000d+00,1.000000d+00)*g**3)/(4.800000d+01*pi**2)),DCMPLX(0.000000d+00)))')
#       Bah, we don't aim at supporting precedence for entangled if statements.
#                 ,('1 if a else 2 if b else 3',
#                  '')
                  ]

        for toParse, sol in tests:
            self.assertEqual(self.calc.parse(toParse), sol)

    def test_parse_fortran_IfElseStruct_MP(self):
        """Test that structures like ( 1 if 2==3 else 4) are correctly parsed
         for quadruple precision"""

        tests = [ ('(1 if a==0 else 1/a)',
         '(MP_CONDIF(mp__a.EQ.0.000000e+00_16,CMPLX(1.000000e+00_16,KIND=16),CMPLX(1.000000e+00_16/mp__a,KIND=16)))'),
                  ('1/a if a else 1',
         'MP_CONDIF(CMPLX(mp__a,KIND=16).NE.(0.0e0_16,0.0e0_16),CMPLX(1.000000e+00_16/mp__a,KIND=16),CMPLX(1.000000e+00_16,KIND=16))') ]

        for toParse, sol in tests:
            #print toParse
            self.assertEqual(self.mp_calc.parse(toParse), sol)


    def test_parse_fortran_fct(self):
        """Test that we can parse a series of expression including
        1j and .real"""
        
        tests = [('1j', 'DCOMPLX(0d0, 1.000000d+00)'),
                 ('3+3j', '3.000000d+00+DCOMPLX(0d0, 3.000000d+00)'),
                 ('re1j', 're1j'),
                 ('re(x)', 'dble(x)'),
                 ('x.real', 'dble(x)'),
                 ('(cmath.log(x)/x).real', 'dble(log(x)/x)'),
                 ('3*x.real', '3.000000d+00*dble(x)'),
                 ('x*y.real', 'x*dble(y)'),
                  ('(x*y.real)', '(x*dble(y))'),
                 ('im(x)', 'dimag(x)'),
                 ('x.imag', 'dimag(x)'),
                 ('(cmath.log(x)/x).imag', 'dimag(log(x)/x)'),
                 ('3*x.imag', '3.000000d+00*dimag(x)'),
                 ('x*y.imag', 'x*dimag(y)'),
                  ('(x*y.imag)', '(x*dimag(y))')
                 ]
        
        for toParse, sol in tests:
            self.assertEqual(self.calc.parse(toParse), sol)  

    def test_parse_fortran_fct_MP(self):
        """Test that we can parse a series of expression including
        1j and .real"""
        
        tests = [('1j', 'CMPLX(0.000000e+00_16, 1.000000e+00_16 ,KIND=16)'),
                 ('3+3j', '3.000000e+00_16+CMPLX(0.000000e+00_16, 3.000000e+00_16 ,KIND=16)'),
                 ('re1j', 'mp__re1j'),
                 ('re(x)', 'real(mp__x)'),
                 ('x.real', 'real(mp__x)'),
                 ('(cmath.log(x)/x).real', 'real(log(mp__x)/mp__x)'),
                 ('3*x.real', '3.000000e+00_16*real(mp__x)'),
                 ('x*y.real', 'mp__x*real(mp__y)'),
                  ('(x*y.real)', '(mp__x*real(mp__y))'),

                 ]
        
        for toParse, sol in tests:
            self.assertEqual(self.mp_calc.parse(toParse), sol)  
