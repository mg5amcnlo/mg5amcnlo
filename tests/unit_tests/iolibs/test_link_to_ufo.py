################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
import copy
import subprocess
import shutil
import os

import tests.unit_tests as unittest

from madgraph import MG4DIR, MG5DIR

import madgraph.core.base_objects as base_objects
import madgraph.iolibs.export_v4 as export_v4
import models.import_ufo as import_ufo
import madgraph.iolibs.files as files
import madgraph.iolibs.import_v4 as import_v4
import madgraph.iolibs.ufo_expression_parsers as ufo_expression_parsers
from madgraph.iolibs import save_load_object



file_dir_path = os.path.dirname(os.path.realpath( __file__ ))
root_path = os.path.join(file_dir_path, os.pardir, os.pardir, os.pardir)


# Test UFO Expression parsers

class TestUFOExpressionParsers(unittest.TestCase):
        
    def setUp(self):
        pass
    
    def test_parse_to_fortran(self):
        """Test Python to Fortran expression parser"""
        
        ufo_to_fortran = ufo_expression_parsers.UFOExpressionParserFortran()
        expr = 'cmath.sqrt(2)'
        converted = ufo_to_fortran.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, 'sqrt(dcmplx(2.000000d+00))')
        
        expr = 'cmath.sqrt(2.)'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 'sqrt(dcmplx(2.000000d+00))')
        
        expr = 'randomfunction(2.)'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 'randomfunction(2.000000d+00)')
        
        expr = 'function(2., a+b)'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 'function(2.000000d+00,a+b)')
        
        expr = 'cmath.sqrt(2.5)'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 'sqrt(dcmplx(2.500000d+00))')

        expr = 'cmath.testfcn(2.00, a+b)'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 'testfcn(2.000000d+00,a+b)')
                
        expr = 'cmath.sin(2.00)'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 'sin(2.000000d+00)')
                
        expr = '(ee**2*IMAG/(2.*sw**2) * (cmath.sin(cmath.sqrt(2)*ee)**2/3.))'
        converted = ufo_to_fortran.parse(expr)
        self.assertEqual(converted, 
        '(ee**2*imag/(2.000000d+00*sw**2)*(sin(sqrt(dcmplx(2.000000d+00))*ee)**2/3.000000d+00))')
    
    def test_convert_number_to_fortran(self):
        """ test it can convert number in fortran string"""
        
        py2f77 = ufo_expression_parsers.UFOExpressionParserFortran()
        expr = str(2)
        converted = py2f77.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.000000d+00')  
        
        expr = str(0.23)
        converted = py2f77.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.300000d-01')  
        
        expr = str(2.5e6)
        converted = py2f77.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.500000d+06')
        
        expr = str(0.0000116639)  
        converted = py2f77.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '1.166390d-05')        
        
    def test_parse_to_cpp(self):
        """Test Python to C++ expression parser"""
        
        ufo_to_pythia8 = ufo_expression_parsers.UFOExpressionParserCPP()
        expr = 'cmath.sqrt(2)'
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, 'sqrt(2.)')
        
        expr = 'randomfunction(2.)'
        converted = ufo_to_pythia8.parse(expr)
        self.assertEqual(converted, 'randomfunction(2.)')
        
        expr = 'function(2.0000000, a+b)'
        converted = ufo_to_pythia8.parse(expr)
        self.assertEqual(converted, 'function(2.,a+b)')
        
        expr = 'cmath.sqrt(2.5)'
        converted = ufo_to_pythia8.parse(expr)
        self.assertEqual(converted, 'sqrt(2.5)')

        expr = 'cmath.testfcn(2.00, a+b)'
        converted = ufo_to_pythia8.parse(expr)
        self.assertEqual(converted, 'testfcn(2.,a+b)')
        
        expr = '(ee**2*IMAG/(2.*sw**2) * (cmath.sin(cmath.sqrt(2)*ee)**2/3.))'
        converted = ufo_to_pythia8.parse(expr)
        self.assertEqual(converted, 
        '(pow(ee,2.)*IMAG/(2.*pow(sw,2.))*(pow(sin(sqrt(2.)*ee),2.)/3.))')
    
    def test_convert_number_to_cpp(self):
        """ test it can convert number in C++ string"""
        
        ufo_to_pythia8 = ufo_expression_parsers.UFOExpressionParserCPP()
        expr = str(2)
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.')  
        
        expr = str(2.00000)
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.')  
        
        expr = str(0.23)
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '0.23')  
        
        expr = '2.5e6'
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.5e6')
        
        expr = '2.5e+6'
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.5e+6')
        
        expr = '.5e-6'
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '.5e-6')
        
        expr = str(1.5)  
        converted = ufo_to_pythia8.parse(expr)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '1.5')        
        
class TestImportUFO(unittest.TestCase):
    """ check if we can import properly a model """
    
    def test_simple_import(self):
        """ check that basic quantity are define """
        
        #remove pkl file
        try:
            model_path = os.path.join(MG5DIR, 'models', 'sm')
            os.remove(os.path.join(model_path,'model.pkl'))
        except:
            pass
        
        import_ufo._import_once = []
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path)
    
        self.assertNotEqual(model.get('particles'),None)
        self.assertNotEqual(model.get('particles'),[], "empty particles list")
    
        self.assertNotEqual(model.get('interactions'),None)
        self.assertNotEqual(model.get('interactions'),[])    
        
        
        # try with the pickle:
        sm_path = import_ufo.find_ufo_path('sm')
        model = import_ufo.import_model(sm_path)
    
        self.assertNotEqual(model.get('particles'),None)
        self.assertNotEqual(model.get('particles'),[], "empty particles list")
    
        self.assertNotEqual(model.get('interactions'),None)
        self.assertNotEqual(model.get('interactions'),[])            
        


