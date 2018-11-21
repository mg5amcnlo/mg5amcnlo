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
import madgraph.interface.master_interface as Cmd
import madgraph.iolibs.ufo_expression_parsers as parsers

#===============================================================================
# IOMiscTest
#===============================================================================
class UFOParserTest(unittest.TestCase):
    """Test class for ufo_expression_parsers.py"""

    def setUp(self):
        if not hasattr(UFOParserTest, 'model'):
            self.cmd = Cmd.MasterCmd()
            self.cmd.exec_cmd("import model loop_qcd_qed_sm")
            UFOParserTest.model = self.cmd._curr_model
            
        if not hasattr(self, 'calc'):
            self.calc = parsers.UFOExpressionParserFortran(UFOParserTest.model)
        if not hasattr(self, 'mp_calc'):
            self.mp_calc = parsers.UFOExpressionParserMPFortran(UFOParserTest.model)

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
                  '(CONDIF(DCMPLX(mt).NE.(0d0,0d0),DCMPLX((DCMPLX(0.000000d+00,1.000000d+00)*g**3)/(4.800000d+01*pi**2)),DCMPLX(0.000000d+00)))'),
                ('(0.0 if z.imag==0.0 and z.real==0.0 else ( cmath.log(z) + 2*cmath.pi*1j if (z.real < 0.0 and z.imag < 0.0) else cmath.log(z) ) )',
                 '(CONDIF(dimag(z).EQ.0.000000d+00.AND.dble(z).EQ.0.000000d+00,DCMPLX(0.000000d+00),DCMPLX((CONDIF((dble(z).LT.0.000000d+00.AND.dimag(z).LT.0.000000d+00),DCMPLX(log(z)+2.000000d+00*pi*DCMPLX(0d0, 1.000000d+00)),DCMPLX(log(z)))))))'),
                 ('recms(CMSParam==1.0 and WT != 0.0,4 + reglog(64) + 3*reglog(cmath.pi) + 3*reglog(MU_R**2/MT**2))',
                  'RECMS(cmsparam.EQ.1.000000d+00.AND.wt.NE.0.000000d+00,DCMPLX(4.000000d+00+reglog(DCMPLX(6.400000d+01))+3.000000d+00*reglog(DCMPLX(pi))+3.000000d+00*reglog(DCMPLX(mu_r**2/mt**2))))')
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


    def test_UFOExpressionParserPythonIF(self):
        """ Tests that the UFO parsers for conditional 'if' statement replacement
        works properly."""
        
        # Make sure the constructor works properly
        wrong_arguments = ["{'a':1,'b':2",
                           [('a',1),('b',2)],
                           "vtcewx",
                           "{'a'1,'b':2}"]
        for arg in wrong_arguments:
            self.assertRaises(parsers.ModelError,
                                       parsers.UFOExpressionParserPythonIF, arg)
        
        right_arguments = [{'a':1,'b':2},
                          "{'a':1.0,'b':2}",
                          {'a':1.0j,'b':2},
                          {'a':1,'b':2+3.0j},
                          "{'a':1,'b':2+3.0j}",
                          {'a':1.0,'b':2,'c':'po'}
                          ]
        for arg in right_arguments:
            parsers.UFOExpressionParserPythonIF(arg)  
        
        # Use the following parser for tests
        # With all definitions
        calc = parsers.UFOExpressionParserPythonIF(
                                           {'a':0j,'b':1.0,'c':2+1.0j,'abcd':3})
        tests = [
          ('True if b==1 else False',('True',1)),
          ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',('(rfehfrg+csc(a))',1)),
          ('(arg(b) if abs(a-c)>abs(b*1j) else 2) if True else Miaou',('(arg(b))',2)),
          ('True if a else False',('False',1)),
          ('(1 if abcd else 2) if b!=0 else ( ( 5 if c-1j==2 else 0 ) if abcd==3 else 4)',('(1)',4)),
          ('(1 if a else 2) if b==0 else ( ( 5 if c-1j==2 else 0 ) if abcd==3 else 4)',('((5))',4)),
          ]
        for test in tests:
            self.assertEqual(calc.parse(test[0]),test[1])
            
        calc = parsers.UFOExpressionParserPythonIF({'a':0j,'b':1.0})
        # With only partial definitions
        tests = [
          ('True if b==1 else False',('True',1)),
          ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',
               ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',0)),
          ('(arg(b) if abs(a-c)>abs(b*1j) else 2) if True else Miaou',
             ('(arg(b) if abs(a-c)>abs(b*1j) else 2)',1)),
          ('True if c else False',('True if c!=0.0 else False',0)),
          ('(1 if abcd else 2) if b!=0 else ( ( 5 if c-1j==2 else 0 ) if abcd==3 else 4)',
           ('(1 if abcd!=0.0 else 2)',1)),
          ('( (5) if abcd==3 else 4)',('((5) if abcd==3 else 4)',0)),
          ]
        for test in tests:
            self.assertEqual(calc.parse(test[0]),test[1])

        # With do defined variables
        calc = parsers.UFOExpressionParserPythonIF({})
        tests = [
          ('True if b==1 else False',('True if b==1 else False',0)),
          ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',
               ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',0)),
          ('(arg(b) if abs(a-c)>abs(b*1j) else 2) if True else Miaou',
             ('(arg(b) if abs(a-c)>abs(b*1j) else 2)',1)),
          ('True if c else False',('True if c!=0.0 else False',0)),
          ('(1 if abcd else 2) if b!=0 else (( 5 if c-1j==2 else 0 ) if abcd==3 else 4)',
           ('(1 if abcd!=0.0 else 2) if b!=0 else ((5 if c-1j==2 else 0) if abcd==3 else 4)',0)),
          ('( (5) if abcd==3 else 4)',('((5) if abcd==3 else 4)',0)),
          ]
        for test in tests:
            self.assertEqual(calc.parse(test[0]),test[1])

        # With None as defined variables
        calc = parsers.UFOExpressionParserPythonIF()
        tests = [
          ('True if b==1 else False',('True if b==1 else False',0)),
          ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',
               ('(rfehfrg+csc(a)) if (a==0 and (abcd+b)*1j==4.0j) else Miaou',0)),
          ('(arg(b) if abs(a-c)>abs(b*1j) else 2) if True else Miaou',
             # I know it is a bit ugly, 'True!=0.0', but hey, if the model builder
             # is stupid enough to write this then it's not my fault. And besides,
             # it still work, so mehhh.
             ('(arg(b) if abs(a-c)>abs(b*1j) else 2) if True!=0.0 else Miaou',0)),
          ('True if c else False',('True if c!=0.0 else False',0)),
          ('(1 if abcd else 2) if b!=0 else (( 5 if c-1j==2 else 0 ) if abcd==3 else 4)',
           ('(1 if abcd!=0.0 else 2) if b!=0 else ((5 if c-1j==2 else 0) if abcd==3 else 4)',0)),
          ('( (5) if abcd==3 else 4)',('((5) if abcd==3 else 4)',0)),
          ]
        for test in tests:
            self.assertEqual(calc.parse(test[0]),test[1])

    def test_parse_fortran_fct(self):
        """Test that we can parse a series of expression including
        1j and .real"""
        
        tests = [('1j', 'DCMPLX(0d0, 1.000000d+00)'),
                 ('3+3j', '3.000000d+00+DCMPLX(0d0, 3.000000d+00)'),
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
                  ('(x*y.imag)', '(x*dimag(y))'),
                  ('complexconjugate(z)', 'conjg(DCMPLX(z))'),
                  ('z.conjugate()', 'conjg(DCMPLX(z))')
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

    def test_parse_special_fortran_fct(self):
        """Test that we parse a few special functions defined in ufo parsers"""

        tests = [('cond(a,b,c)','COND(DCMPLX(a),DCMPLX(b),DCMPLX(c))'),
                 ('reglog(z)','reglog(DCMPLX(z))'),
                 ('reglogp(z)','reglogp(DCMPLX(z))'),
                 ('reglogm(z)','reglogm(DCMPLX(z))'),
                 ('arg(z)','arg(DCMPLX(z))'),
                 ('recms(cms==1.0,z)','RECMS(cms.EQ.1.000000d+00,DCMPLX(z))'),
                 ('grreglog(logswitch,z1,z2)','grreglog(DBLE(logswitch),DCMPLX(z1) , DCMPLX(z2))'),
                 ('regsqrt(z)','regsqrt(DCMPLX(z))'),
                 ('crecms(cms==1.0,z)','crecms(cms.EQ.1.000000d+00,DCMPLX(z))'),
                 ('abs(1.0)', 'ABS(1.000000d+00)'),
                 ('abs(float(1))' , 'ABS(REAL(1.000000d+00))'),
                 ('abs(int(1))' , 'ABS(INTEGER(1.000000d+00))')
                 ]
        

        for toParse, sol in tests:
            self.assertEqual(self.calc.parse(toParse), sol)
        
        # same for MP
        tests = [('cond(a,b,c)','MP_COND(CMPLX(mp__a,KIND=16),CMPLX(mp__b,KIND=16),CMPLX(mp__c,KIND=16))'),
                 ('reglog(z)','mp_reglog(CMPLX((mp__z),KIND=16))'),
                 ('reglogp(z)','mp_reglogp(CMPLX((mp__z),KIND=16))'),
                 ('reglogm(z)','mp_reglogm(CMPLX((mp__z),KIND=16))'),
                 ('arg(z)','mp_arg(CMPLX((mp__z),KIND=16))'),
                 ('recms(cms==1.0,z)','MP_RECMS(mp__cms.EQ.1.000000e+00_16,CMPLX(mp__z,KIND=16))'),
                 ('grreglog(logswitch,z1,z2)','MP_grreglog(mp__logswitch,CMPLX(mp__z1, KIND=16) , CMPLX(mp__z2, KIND=16))'),
                 ('regsqrt(z)','MP_regsqrt(CMPLX(mp__z, KIND=16))'),
                 ('crecms(cms==1.0,z)','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX(mp__z, KIND=16))'),
                 ('crecms(cms==1.0,z*y)','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX(mp__z*mp__y, KIND=16))'),
                 ('crecms(cms==1.0,z+y)','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX(mp__z+mp__y, KIND=16))'),
                 ('crecms(cms==1.0, (x*y) )','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX((mp__x*mp__y), KIND=16))'),
                 ('crecms(cms==1.0, 2*y )','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX(2.000000e+00_16*mp__y, KIND=16))'),
                 ('crecms(cms==1.0, -y )','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX(-mp__y, KIND=16))'),
                 ('crecms(cms==1.0, -1*y )','MP_crecms(mp__cms.EQ.1.000000e+00_16,CMPLX(-1.000000e+00_16*mp__y, KIND=16))'),
                 ('abs(1.0)', 'ABS(1.000000e+00_16)'),
                 ('abs(float(x))' , 'ABS(REAL((mp__x), KIND=16))')
                 ]


        for toParse, sol in tests:
            self.assertEqual(self.mp_calc.parse(toParse), sol)
