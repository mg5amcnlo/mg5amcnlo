################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Unit test library for the Misc routine library in the I/O package"""

from __future__ import absolute_import
import contextlib
import io
import tests.unit_tests as unittest
import tests.test_manager as test_manager

#===============================================================================
# TestUnmatchedExpression
#===============================================================================
class TestUnmatchedExpression(unittest.TestCase):
    """Check that a test name asked for on the command line but selecting no
    test at all is reported, so that a renamed/removed test can not keep on
    being silently 'OK' in the CI."""

    test_path = 'tests/unit_tests/various'

    def finder(self, expression):
        return test_manager.TestFinder(package=self.test_path,
                                       expression=expression)

    def test_unmatched_on_unknown_name(self):
        """a name which matches nothing is reported"""

        finder = self.finder(['test_no_such_test_anywhere'])
        self.assertEqual(finder.unmatched_expressions(),
                         ['test_no_such_test_anywhere'])

    def test_unmatched_keeps_the_valid_names(self):
        """only the names selecting nothing are reported"""

        finder = self.finder(['test_unmatched_on_unknown_name',
                              'test_no_such_test_anywhere'])
        self.assertEqual(finder.unmatched_expressions(),
                         ['test_no_such_test_anywhere'])

    def test_unmatched_on_class_module_and_regexp(self):
        """a class, a module or a regular expression can select a test as well"""

        for expression in ['TestUnmatchedExpression', 'test_test_finder',
                           'test_test_finder.py',
                           'tests/unit_tests/various/test_test_finder.py',
                           'test_unmatched_on_class.*']:
            finder = self.finder([expression])
            self.assertEqual(finder.unmatched_expressions(), [],
                             'wrongly reported as unknown: %s' % expression)

    def test_unmatched_on_directory(self):
        """a directory below the searched package selects all its tests"""

        finder = test_manager.TestFinder(package='tests/unit_tests',
                                         expression=['various'])
        self.assertEqual(finder.unmatched_expressions(), [])

    def test_unmatched_empty_when_nothing_requested(self):
        """the bulk modes (no name given) never report anything"""

        self.assertEqual(self.finder('').unmatched_expressions(), [])
        self.assertEqual(self.finder([]).unmatched_expressions(), [])

    def test_only_a_plain_name_is_fatal(self):
        """a regular expression is allowed to select nothing, a name is not"""

        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            fatal = test_manager.report_unmatched(['test_gone', 'test_gone_.*'],
                                                  self.test_path)
        self.assertEqual(fatal, ['test_gone'])
        self.assertIn('test_gone_.*', report.getvalue())

#===============================================================================
# TestTestFinder
#===============================================================================
if 0:
 class TestTestFinder(unittest.TestCase):
    """The TestCase class for the test module"""

    #This class test to find it's own property so it need some shortcut
    test_path = './tests/unit_tests/various/'

    def setUp(self):
        """ basic building of the class to test """
        self.testmodule = test_manager.TestFinder(package=self.test_path)

    def tearDown(self):
        pass

    def test_buiding_in_iter(self):
        """ TestFinder.__iter__ is able to build the list """

        for dummy in self.testmodule:
            break
        self.assertGreater(len(self.testmodule), 1)

    def test_collect_dir(self):
        """ TestFinder.collect_dir should detect subdirectory and file """

        self.testmodule = test_manager.TestFinder(package=\
                         './tests/unit_tests')
        self.assertIn(
            'tests.unit_tests.various.test_test_finder.' + \
                         'TestTestFinder.test_collect_dir',
            self.testmodule
        )

    def test_collect_dir_with_restriction_file(self):
        " TestFinder.collect_dir pass correctly restriction rule on file "
        self.testmodule.restrict_to('test_test_finder.py')
        self.assertIn(
            'tests.unit_tests.various.test_test_finder.' + \
                     'TestTestFinder.test_collect_dir',
            self.testmodule
        )

    def test_collect_dir_with_restriction_file2(self):
        " TestFinder.collect_dir pass correctly restriction rule on file "
        self.testmodule.restrict_to('./tests/unit_tests/various/' + 
                                    'test_test_finder.py')
        self.assertIn(
            'tests.unit_tests.various.test_test_finder.' + \
                         'TestTestFinder.test_collect_dir',
            self.testmodule
        )

    def test_collect_dir_with_restriction_class(self):
        " TestFinder.collect_dir pass correctly restriction rule on class "

        self.testmodule.restrict_to('TestTestFinder')
        self.assertIn(
            'tests.unit_tests.various.test_test_finder.' + \
                     'TestTestFinder.test_collect_dir',
            self.testmodule
        )

    def test_collect_dir_with_restriction_fct(self):
        " TestFinder.collect_dir pass correctly restriction rule on fct "

        self.testmodule.restrict_to('test_check_valid.*')
        self.assertNotIn(
            'tests.unit_tests.various.test_test_finder.' + \
                     'TestTestFinder.test_collect_dir',
            self.testmodule
        )
        self.assertIn(
            'tests.unit_tests.various.test_test_finder.' + \
                     'TestTestFinder.test_check_valid_on_file',
            self.testmodule
        )

    def test_collect_file_wrong_arg(self):
        """ TestFinder.collect_file fails on wrong argument"""

        for input in [1, {1:2}, [1], str, int, list, 'alpha']:
            self.assertRaises(test_manager.TestFinder.TestFinderError, \
                              self.testmodule.collect_file, input)

    def test_collect_file(self):
        """ TestFinder.collect_file find the different class in a file """

        self.testmodule.collect_file('./tests/unit_tests/various/' + \
                                     'test_test_finder.py')
        self.assertIn(
            'tests.unit_tests.various.test_test_finder.' + \
                     'TestTestFinder.test_collect_file',
            self.testmodule
        )

    def test_collect_function_wrong_arg(self):
        """ TestFinder.collect_function fails on wrong argument """

        for input in [1, {1:2}, [1], 'alpha', str, int, list]:
            self.assertRaises(test_manager.TestFinder.TestFinderError, \
                              self.testmodule.collect_function, input)

    def test_collect_function(self):
        """ TestFinder.collect_function find the test function """

        self.testmodule.collect_function(TestTestFinder)
        self.assertIn(
            'TestTestFinder.test_collect_function',
            self.testmodule
        )

        for name in self.testmodule:
            name = name.split('.')[-1]
            self.assertTrue(name.startswith('test'))

    def test_output_are_function(self):
        """ TestFinder.collect_function returns test functions only """
        self.testmodule.collect_function(TestTestFinder)
        collect = unittest.TestLoader()
        for test_fct in self.testmodule:
            try:
                collect.loadTestsFromName(\
                    'tests.unit_tests.various.test_test_finder.' + test_fct)
            except:
                self.fail('non callable object are returned')

    def test_restrict_to_inputcheck(self):
        """ TestFinder.restrict_to fail on non string/list input """

        input1 = {}
        self.assertRaises(test_manager.TestFinder.TestFinderError, \
                              self.testmodule.restrict_to, input1)
        input1 = 1
        self.assertRaises(test_manager.TestFinder.TestFinderError, \
                              self.testmodule.restrict_to, input1)
        import re

        input1 = re.compile('''1''')
        self.assertRaises(test_manager.TestFinder.TestFinderError, \
                              self.testmodule.restrict_to, input1)

    def test_check_valid_wrong_arg(self):
        """ TestFinder.check_valid raise error if not str in input """

        for input in [1, {1:2}, [1]]:
            self.assertRaises(test_manager.TestFinder.TestFinderError, \
                              self.testmodule.check_valid, input)

    def test_check_valid_on_module(self):
        """ TestFinder.check_valid recognizes module """

        expression = 'various'
        self.testmodule.restrict_to(expression)
        pos = './various'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = './various2'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'various'
        self.assertTrue(self.testmodule.check_valid(pos))
        pos = '../test/various'
        self.assertTrue(self.testmodule.check_valid(pos))

    def test_check_valid_on_file(self):
        """ TestFinder.check_valid recognizes file """

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
        pos = 'various/test/test.py'
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
        pos = 'various/test/test.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test.pyc'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'onetest.py'
        self.assertFalse(self.testmodule.check_valid(pos))
        pos = 'test/file.py'
        self.assertFalse(self.testmodule.check_valid(pos))

    def test_check_valid_on_class(self):
        """ TestFinder.check_valid recognizes class of test """

        expression = 'Unittest_on_TestinModule'
        self.testmodule.restrict_to(expression)
        name = 'Unittest_on_TestinModule'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'test.Unittest_on_TestinModule'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'I.am.Your.Father'
        self.assertFalse(self.testmodule.check_valid(name))

    def test_check_valid_on_function(self):
        """ TestFinder.check_valid recognizes functions """

        expression = 'test_search'
        self.testmodule.restrict_to(expression)
        name = 'test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'test.test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'It.is.impossible'
        self.assertFalse(self.testmodule.check_valid(name))

        expression = 'test_check_valid.*'
        self.testmodule.restrict_to(expression)
        name = 'module.test_check_valid_on_function'
        self.assertTrue(self.testmodule.check_valid(name))

    def test_check_valid_with_re(self):
        """ TestFinder.check_valid should work with re """

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
        self.testmodule.restrict_to(expression, re_opt)
        name = 'test'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'TEST'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'one_test'
        self.assertFalse(self.testmodule.check_valid(name))

    def test_check_valid_with_list_restriction(self):
        """ TestFinder.check_valid should work with list in restrict """

        expression = ['test.*', 'iolibs']
        self.testmodule.restrict_to(expression)
        name = 'test_search'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'iolibs'
        self.assertTrue(self.testmodule.check_valid(name))
        name = 'data'
        self.assertFalse(self.testmodule.check_valid(name))

    def test_passin_pyformat(self):
        """ Test convert from linux position format to python include """

        input = {'test.py':'test', 'Source/test.py':'Source.test', \
                   'Source//test.py':'Source.test', './test.py':'test'}
        for key, value in input.items():
            self.assertEqual(\
                self.testmodule.passin_pyformat(key),
                value)

        input = ['test', '../Source.py', '/home/data.py', 1, str]
        for data in input:
            self.assertRaises(test_manager.TestFinder.TestFinderError
                              , self.testmodule.passin_pyformat, data)

    def test_add_to_possibility(self):
        """ Test convert name in different matching possibility """
        # the sanity of the output is checked by check_valid test,
        # this test the format of the output

        output = self.testmodule.format_possibility('./various/test.py')
        for name in output:
            self.assertEqual(output.count(name), 1)
        self.assertGreater(len(output), 3)

        output = self.testmodule.format_possibility('various.test')
        for name in output:
            self.assertEqual(output.count(name), 1)
        self.assertGreater(len(output), 1)

    def test_status_file_on_file(self):
        """ TestFinder.status_file recognizes file """

        self.testmodule.go_to_root()
        status = self.testmodule.status_file(self.test_path + \
                                               'test_test_finder.py')
        self.assertEqual(status, 'file')
        status = self.testmodule.status_file(self.test_path + \
                             '../../../README')
        self.assertFalse(status)
        status = self.testmodule.status_file(self.test_path + \
                             '__init__.py')
        self.assertFalse(status)
        status = self.testmodule.status_file(self.test_path + \
                             '__init__.pyc')
        self.assertFalse(status)

        self.testmodule.go_to_initpos()

    def test_status_file_on_dir(self):
        """ TestFinder.status_file  doesn't detect non module dir """

        self.testmodule.go_to_root()
        status = self.testmodule.status_file(self.test_path + \
                             '../various')
        self.assertEqual(status, 'module')

        status = self.testmodule.status_file(self.test_path + \
                             '../../../apidoc')
        self.assertFalse(status)

        self.testmodule.go_to_initpos()


    def test_new_assertAlmostEqual(self):
        """Test the replacement of assertAlmostEqual"""

        self.assertAlmostEqual(0., 0.00000004)
        self.assertRaises(AssertionError, self.assertAlmostEqual, 0., 0.00000006)
        self.assertAlmostEqual(1e-15, 1.0000005e-15)
        self.assertRaises(AssertionError,
                          self.assertAlmostEqual, 1e-15, 1.0000006e-15)
        self.assertAlmostEqual(0.000000004, 0., 8)
        self.assertRaises(AssertionError, self.assertAlmostEqual,
                          0.000000006, 0., 8)
        self.assertAlmostEqual(1.00000004e-15, 1e-15, 8)
        self.assertRaises(AssertionError,
                          self.assertAlmostEqual, 1.00000006e-15, 1e-15, 8)
