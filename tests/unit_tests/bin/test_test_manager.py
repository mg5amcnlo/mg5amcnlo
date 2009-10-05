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

"""Unit test library for the Misc routine library in the I/O package"""

import sys
sys.path+=['../../../bin','.','./bin']

import unittest
import test_manager
import inspect

class Unittest_on_TestinModule(unittest.TestCase):
    #This class test to find it's own property so it need some shortcut
    test_path='./tests/unit_tests/bin/' 
    name='test.manager.Unittest_on_TestinModule'
    
    def setUp(self):
        """ basic building of the class to test """
        self.testmodule=test_manager.Test_in_module(package=self.test_path)

    def tearDown(self):
        pass

    def test_buiding_in_iter(self):
        """ Test_in_module.__iter__ is able to build the list """

        for i in self.testmodule:
            break
        self.assert_(len(self.testmodule)>1)

    def test_collect_dir(self):
        """ Test_in_module.collect_dir should detect subdirectory and file """

        self.testmodule=test_manager.Test_in_module(package= \
                         './tests/unit_tests')
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_collect_dir' in self.testmodule)

    def test_collect_dir_with_restriction_file(self):
        """ Test_in_module.collect_dir pass corectly restriction rule on file"""
        self.testmodule.restrict_to('test_test_manager.py')
#        self.testmodule.collect_dir('./tests/unit_tests/bin')
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_collect_dir'  in self.testmodule)

    def test_collect_dir_with_restriction_file2(self):
        """ Test_in_module.collect_dir pass corectly restriction rule on file"""
        self.testmodule.restrict_to('./tests/unit_tests/bin/test_test_manager.py')
#        self.testmodule.collect_dir('./tests/unit_tests/bin')
#        print self.testmodule
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_collect_dir'  in self.testmodule)

    def test_collect_dir_with_restriction_class(self):
        """ Test_in_module.collect_dir pass corectly restriction rule on class"""

        self.testmodule.restrict_to('Unittest_on_TestinModule')
#        self.testmodule.collect_dir('./tests/unit_tests/bin')
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_collect_dir'  in self.testmodule)

    def test_collect_dir_with_restriction_fct(self):
        """ Test_in_module.collect_dir pass corectly restriction rule on fct"""

        self.testmodule.restrict_to('test_check_valid.*')
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_collect_dir' not in self.testmodule)
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_check_valid_on_file' in self.testmodule)


    def test_collect_file_wrong_arg(self):
        """ Test_in_module.collect_file fails on wrong argument"""

        for input in [1,{1:2},[1],str,int,list,'alpha']:
            self.assertRaises(test_manager.Test_in_module.ERRORTestManager, \
                              self.testmodule.collect_file, input)

    def test_collect_file(self):
        """ Test_in_module.collect_file find the different class in a file """
        
        self.testmodule.collect_file('./tests/unit_tests/bin/test_test_manager.py')
        self.assert_('tests.unit_tests.bin.test_test_manager.Unittest_on_TestinModule.test_collect_file' in \
                         self.testmodule)

    def test_collect_function_wrong_arg(self):
        """ Test_in_module.collect_function fails on wrong argument """

        for input in [1,{1:2},[1],'alpha',str,int,list]:
            self.assertRaises(test_manager.Test_in_module.ERRORTestManager, \
                              self.testmodule.collect_function, input)


    def test_collect_function(self):
        """ Test_in_module.collect_function find the test function """
        
        self.testmodule.collect_function(Unittest_on_TestinModule)
        self.assert_('Unittest_on_TestinModule.test_collect_function' in \
                         self.testmodule)

        for name in self.testmodule:
            name=name.split('.')[-1]
            self.assertTrue(name.startswith('test'))

    def test_output_are_function(self):
        """ Test_in_module.collect_function returns test funcions only """
        self.testmodule.collect_function(Unittest_on_TestinModule)
        mytest = unittest.TestSuite()
        collect = unittest.TestLoader()
        for test_fct in self.testmodule:
            try:
                collect.loadTestsFromName( \
                    'tests.unit_tests.bin.test_test_manager.'+test_fct)
            except:
                self.fail('non callable object are returned')

            
    def test_restrict_to_inputcheck(self):
        """ Test_in_module.restrict_to fail on non string/list input """

        input1={}
        self.assertRaises(test_manager.Test_in_module.ERRORTestManager, \
                              self.testmodule.restrict_to, input1)
        input1=1
        self.assertRaises(test_manager.Test_in_module.ERRORTestManager, \
                              self.testmodule.restrict_to, input1)
        import re

        input1=re.compile('''1''')
        self.assertRaises(test_manager.Test_in_module.ERRORTestManager, \
                              self.testmodule.restrict_to, input1)

    def test_check_valid_wrong_arg(self):
        """ Test_in_module.check_valid raise error if not str in input """

        for input in [1,{1:2},[1]]:
            self.assertRaises(test_manager.Test_in_module.ERRORTestManager, \
                              self.testmodule.check_valid, input)

    def test_check_valid_on_module(self):
        """ Test_in_module.check_valid recognizes module """

        expression = 'bin'
        self.testmodule.restrict_to(expression)
        pos = './bin'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = './bin2'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'bin'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = '../test/bin'
        self.assertTrue(self.testmodule.check_valid(pos))

    def test_check_valid_on_file(self):
        """ Test_in_module.check_valid recognizes file """

        expression = 'test'
        self.testmodule.restrict_to(expression)
        pos = 'test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = './test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = '../test/test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = 'test.pyc'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'onetest.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test/file.py'
        self.assertFalse(self.testmodule.check_valid(pos))

        expression = 'test.py'
        self.testmodule.restrict_to(expression)
        pos = 'test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = './test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = '../test/test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = 'test.pyc'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'onetest.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test/file.py'
        self.assertFalse(self.testmodule.check_valid(pos))

        expression = 'test.test.py'
        self.testmodule.restrict_to(expression)
        pos = 'test/test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = './test.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'bin/test/test.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test.pyc'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'onetest.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test/file.py'
        self.assertFalse(self.testmodule.check_valid(pos))

        expression = './test/test.py'
        self.testmodule.restrict_to(expression)
        pos = 'test/test.py'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = './test.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'bin/test/test.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test.pyc'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'onetest.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test/file.py'
        self.assertFalse(self.testmodule.check_valid(pos))


    def test_check_valid_on_class(self):
        """ Test_in_module.check_valid recognizes class of test """
        
        expression = 'Unittest_on_TestinModule'
        self.testmodule.restrict_to(expression)
        name = 'Unittest_on_TestinModule'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'test.Unittest_on_TestinModule'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'I.am.Your.Father'
        self.assertFalse(self.testmodule.check_valid(name))

    def test_check_valid_on_function(self):
        """ Test_in_module.check_valid recognizes functions """

        expression='test_search'
        self.testmodule.restrict_to(expression)
        name='test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name='test.test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name='It.is.impossible'
        self.assertFalse(self.testmodule.check_valid(name))

        expression='test_check_valid.*'
        self.testmodule.restrict_to(expression)
        name='module.test_check_valid_on_function'
        self.assertTrue(self.testmodule.check_valid(name))

    def test_check_valid_with_re(self):
        """ Test_in_module.check_valid should work with re """

        expression = 'test.*'
        self.testmodule.restrict_to(expression,)
        name = 'test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'valid.test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'one_test'
        self.assertFalse(self.testmodule.check_valid(name))

        expression = 'test'
        import re
        re_opt = re.I
        self.testmodule.restrict_to(expression,re_opt)
        name = 'test'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'TEST'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'one_test'
        self.assertFalse(self.testmodule.check_valid(name))

    def test_check_valid_with_list_restriction(self):
        """ Test_in_module.check_valid should work with list in restrict """
        
        expression = ['test.*','iolibs']
        self.testmodule.restrict_to(expression)
        name = 'test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'iolibs'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'data'
        self.assertFalse(self.testmodule.check_valid(name))

    def test_status_file_on_file(self):
        """ Test_in_module.status_file recognizes file """

        status=self.testmodule.status_file(self.test_path+\
                                               'test_test_manager.py')
        self.assertEqual(status,'file')
        status=self.testmodule.status_file(self.test_path+\
                             '../../../README')
        self.assertFalse(status)
        status=self.testmodule.status_file(self.test_path+\
                             '__init__.py')
        self.assertFalse(status)
        status=self.testmodule.status_file(self.test_path+\
                             '__init__.pyc')
        self.assertFalse(status)


    def test_status_file_on_dir(self):
        """ Test_in_module.status_file  doesn't detect non module dir """

        status=self.testmodule.status_file(self.test_path+\
                             '../bin')
        self.assertEqual(status,'module')

        status=self.testmodule.status_file(self.test_path+\
                             '../../../apidoc')
        self.assertFalse(status)

    
    def test_passin_pyformat(self):
        """ convert from linux position format to python include """

        input={'test.py':'test', 'Source/test.py':'Source.test', \
                   'Source//test.py':'Source.test', './test.py':'test'}
        for key,value in input.items():
            self.assertEqual( \
                self.testmodule.passin_pyformat(key),
                value)

        input=['test','../Source.py','/home/data.py',1,str]
        for data in input:
            self.assertRaises(test_manager.Test_in_module.ERRORTestManager
                              ,self.testmodule.passin_pyformat, data)
        

    def test_add_to_possibility(self):
        """ convert name in different matching possibility """
        # the sanity of the output is checked by check_valid test,
        # this test the format of the output
        
        output=self.testmodule.format_possibility('./bin/test.py')
        for name in output:
            self.assertEqual(output.count(name),1)
        self.assert_(len(output)>3)

        output=self.testmodule.format_possibility('bin.test')
        for name in output:
            self.assertEqual(output.count(name),1)
        self.assert_(len(output)>1)

if __name__ == "__main__":
    mytest = unittest.TestSuite()
    suite = unittest.TestLoader()
    suite=suite.loadTestsFromTestCase(Unittest_on_TestinModule)
    mytest.addTest(suite)
    unittest.TextTestRunner(verbosity=2).run(mytest)


#    unittest.main()
