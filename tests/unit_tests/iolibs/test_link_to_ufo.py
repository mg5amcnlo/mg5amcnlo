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
import unittest
import os

import madgraph.iolibs.convert_ufo2mg4 as ufo2mg4

file_dir_path = os.path.dirname(os.path.realpath( __file__ ))
root_path = os.path.join(file_dir_path, os.pardir, os.pardir, os.pardir)

class TestPythonToFrotran(unittest.TestCase):
    
    def test_convert_str(self):
        """ python to fortran expression is working"""
        
        expr = 'cmath.sqrt(2)'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, 'dsqrt(2.000000d+00)')
        
        expr = 'sqrt(2)' 
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 'dsqrt(2.000000d+00)')
        
        expr = 'sqrt(2.)'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 'dsqrt(2.000000d+00)')
        
        expr = 'sqrt(2.5)'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 'dsqrt(2.500000d+00)')

        
        expr = '(ee**2*IMAG/(2.*sw**2) * (cmath.sin(sqrt(2)*ee)/3.'
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted, 
        '(ee**2*imag/(2.000000d+00*sw**2) * (dsin(dsqrt(2.000000d+00)*ee)/3.000000d+00')
    
    def test_convert_number(self):
        """ test it can convert number in fortran string"""
        
        expr = 2
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.000000d+00')  
        
        expr = 0.23
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.300000d-01')  
        
        expr = 2.5e6
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '2.500000d+06')
        
        expr = 0.0000116639  
        converted = ufo2mg4.python_to_fortran(expr)
        self.assertEqual(converted.__class__, ufo2mg4.python_to_fortran)
        self.assertTrue(isinstance(converted, str))
        self.assertEqual(converted, '1.166390d-05')        
        
        
class CheckFileCreate():
    """Check that the files are correctly created"""

    output_path = '/tmp/' # work only on LINUX but that's ok for the test routine
    created_files =[]

    def assertFileContains(self, filename, solution):
        """ Check the content of a file """

        current_value = open(self.give_pos(filename)).read()
        self.assertEqual(current_value, solution)

    def FileContent(self, filename):
        return open(self.give_pos(filename)).read()

    def ReturnFile(self, filename):
        return open(self.give_pos(filename))

    def give_pos(self, filename):
        """ take a name and a change it in order to have a valid path in the output directory """
        
        return os.path.join(self.output_path, filename)

    def clean_files(self):
        """ suppress all the files linked to this test """
        
        for filename in self.created_files:
            print filename
            try:
                os.remove(self.give_pos(filename))
            except OSError:
                pass
    

class TestModelCreation(unittest.TestCase, CheckFileCreate):

    created_files = ['couplings.f', 'couplings1.f', 'intparam_definition.inc', 
                    ]

    # clean all the tested files before and after any test
    setUP = CheckFileCreate.clean_files
    tearDown = CheckFileCreate.clean_files

    def test_intparam_definition_creation(self):
        """ test the creation of a valid intparam_definition"""
        solution = """ BLABLA"""

        # Create the intparam_definition.inc
        sm_model_path = os.path.join(root_path, 'models', 'sm') 
        converter =  ufo2mg4.convert_model_to_mg4(sm_model_path, self.output_path)
        converter.create_intparam_def()

        # Check that any definition appears only once:
        alreadydefine = []
        for line in self.ReturnFile('intparam_definition.inc'):
            if '=' not in line:
                continue
            new_definition = line.split('=')[0]
            # Check that is the firsttime that this definition is done
            self.assertFalse(new_definition in alreadydefine)

        # Check that the output stays the same
        self.assertFileContains('intparam_definition.inc',
                                 solution)

    def test_couplings_creation(self):
        """ test the creation of a valid couplings.f"""
        
        # Check that the output stays the same
        self.assertFileContains(self.FileContent('couplings.f'), 
                                """" BLABLA """)
