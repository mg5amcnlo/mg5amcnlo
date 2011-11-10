#!/usr/bin/env python 
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

""" Manager for running the test library 

   This library offer a simple way to launch test.
   
   To run a test/class of test/test file/module of test/...
   you just have to launch 
   test_manager.run(NAME)
   or 
   test_manager.run(LIST_OF_NAME)

   the NAME can contain regular expression (in python re standard format)
"""

import sys

if not sys.version_info[0] == 2 or sys.version_info[1] < 6:
    sys.exit('MadGraph 5 works only with python 2.6 or later (but not python 3.X).\n\
               Please upgrate your version of python.')

import inspect
import logging
import logging.config
import optparse
import os
import re
import unittest


#Add the ROOT dir to the current PYTHONPATH
root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.insert(0, root_path)
# Only for profiling with -m cProfile!
#root_path = os.path.split(os.path.dirname(os.path.realpath(sys.argv[0])))[0]
#sys.path.append(root_path)

from madgraph import MG4DIR

#position of MG_ME
MGME_dir = MG4DIR

#===============================================================================
# run
#===============================================================================
def run(expression='', re_opt=0, package='./tests/unit_tests', verbosity=1):
    """ running the test associated to expression. By default, this launch all 
    test inherited from TestCase. Expression can be the name of directory, 
    module, class, function or event standard regular expression (in re format)
    """

    #init a test suite
    testsuite = unittest.TestSuite()
    collect = unittest.TestLoader()
    for test_fct in TestFinder(package=package, expression=expression, \
                                   re_opt=re_opt):
        data = collect.loadTestsFromName(test_fct)
        testsuite.addTest(data)
        
    return unittest.TextTestRunner(verbosity=verbosity).run(testsuite)

#===============================================================================
# TestFinder
#===============================================================================
class TestFinder(list):
    """ Class introspecting the test module to find the available test.
    The routine collect_dir looks in all module/file to find the different 
    functions in different test class. This produce a list, on which external 
    routines can loop on. 
        
    In order to authorize definition and loop on this object on the same time,
    i.e: for test in TestFinder([opt])-. At each time a loop is started, 
    we check if a collect_dir ran before, and run it if necessary.
    """

    search_class = unittest.TestCase

    class TestFinderError(Exception):
        """Error associated to the TestFinder class."""
        pass

    def __init__(self, package='tests/', expression='', re_opt=0):
        """ initialize global variable for the test """

        list.__init__(self)

        self.package = package
        self.rule = []
        if self.package[-1] != '/': 
            self.package += '/'
        self.restrict_to(expression, re_opt)
        self.launch_pos = ''

    def _check_if_obj_build(self):
        """ Check if a collect is already done 
            Uses to have smart __iter__ and __contain__ functions
        """
        if len(self) == 0:
            self.collect_dir(self.package, checking=True)

    def __iter__(self):
        """ Check that a collect was performed (do it if needed) """
        self._check_if_obj_build()
        return list.__iter__(self)

    def __contains__(self, value):
        """ Check that a collect was performed (do it if needed) """
        self._check_if_obj_build()
        return list.__contains__(self, value)

    def collect_dir(self, directory, checking=True):
        """ Find the file and the subpackage in this package """

        #ensures that we are at root position
        move = False
        if self.launch_pos == '':
            move = True
            self.go_to_root()

        for name in os.listdir(os.path.join(root_path,directory)):
            local_check = checking

            status = self.status_file(os.path.join(root_path, directory,name))
                                      #directory + '/' + name)
            if status is None:
                continue

            if checking:
                if self.check_valid(directory + '/' + name):
                    local_check = False    #since now perform all the test

            if status == 'file':
                self.collect_file(directory + '/' + name, local_check)
            elif status == "module":
                self.collect_dir(directory + '/' + name, local_check)

        if move:
            self.go_to_initpos()

    def collect_file(self, filename, checking=True):
        """ Find the different class instance derivated of TestCase """

        pyname = self.passin_pyformat(filename)
        __import__(pyname)
        obj = sys.modules[pyname]
        #look at class
        for name in dir(obj):
            class_ = getattr(obj, name)
            if inspect.isclass(class_) and \
                    issubclass(class_, unittest.TestCase):
                if checking:
                    if self.check_valid(name):
                        check_inside = False
                    else:
                        check_inside = True
                else:
                    check_inside = False


                self.collect_function(class_, checking=check_inside, \
                                          base=pyname)

    def collect_function(self, class_, checking=True, base=''):
        """
        Find the different test function in this class
        test functions should start with test
        """
        if not inspect.isclass(class_):
            raise self.TestFinderError, 'wrong input class_'
        if not issubclass(class_, unittest.TestCase):
            raise self.TestFinderError, 'wrong input class_'

        #devellop the name
        if base:
            base += '.' + class_.__name__
        else:
            base = class_.__name__

        candidate = [base + '.' + name for name in dir(class_) if \
                       name.startswith('test')\
                       and inspect.ismethod(eval('class_.' + name))]

        if not checking:
            self += candidate
        else:
            self += [name for name in candidate if self.check_valid(name)]

    def restrict_to(self, expression, re_opt=0):
        """ 
        store in global the expression to fill in order to be a valid test 
        """

        if isinstance(expression, list):
            pass
        elif isinstance(expression, basestring):
            if expression in '':
                expression = ['.*'] #made an re authorizing all regular name
            else:
                expression = [expression]
        else:
            raise self.TestFinderError, 'obj should be list or string'

        self.rule = []
        for expr in expression:
            #fix the beginning/end of the regular expression
            if not expr.startswith('^'):
                expr = '^' + expr 
            if not expr.endswith('$'):
                expr = expr + '$' 
            self.rule.append(re.compile(expr, re_opt))

    def check_valid(self, name):
        """ check if the name correspond to the rule """

        if not isinstance(name, basestring):
            raise self.TestFinderError, 'check valid take a string argument'

        for specific_format in self.format_possibility(name):
            for expr in self.rule:
                if expr.search(specific_format):
                    return True
        return False

    @staticmethod
    def status_file(name):
        """ check if a name is a module/a python file and return the status """
        if os.path.isfile(os.path.join(root_path, name)):
            if name.endswith('.py') and '__init__' not in name:
                return 'file'
        elif os.path.isdir(os.path.join(root_path, name)):
            if os.path.isfile(os.path.join(root_path, name , '__init__.py')):
                return 'module'

    @classmethod
    def passin_pyformat(cls, name):
        """ transform a relative position in a python import format """

        if not isinstance(name, basestring):
            raise cls.TestFinderError, 'collect_file takes a file position'

        name = name.replace('//', '/') #sanity
        #deal with begin/end
        if name.startswith('./'):
            name = name[2:]
        if not name.endswith('.py'):
            raise cls.TestFinderError, 'Python files should have .py extension'
        else:
            name = name[:-3]

        if name.startswith('/'):
            raise cls.TestFinderError, 'path must be relative'
        if '..' in name:
            raise cls.TestFinderError, 'relative position with \'..\' is' + \
                ' not supported for the moment'

        #replace '/' by points -> Python format
        name = name.replace('/', '.')

        #position
        return name

    def format_possibility(self, name):
        """ return the different string derivates from name in order to 
        scan all the different format authorizes for a restrict_to 
        format authorizes:
        1) full file position
        2) name of the file (with extension)
        3) full file position whithour extension
        4) name of the file (whithout extension)
        5) name of the file (but suppose name in python format)
        6) if name is a python file, try with a './' and with package pos
        """

        def add_to_possibility(possibility, val):
            """ add if not exist """
            if val not in possibility:
                possibility.append(val)
        #end local def

        #sanity
        if name.startswith('./'): 
            name = name[2:]
        name = name.replace('//', '/')
        # init with solution #
        out = [name]

        # add solution 2
        new_pos = name.split('/')[-1]
        add_to_possibility(out, new_pos)

        #remove extension and add solution3 and 6
        if name.endswith('.py'):
            add_to_possibility(out, './' + name)
            add_to_possibility(out, self.package + name)
            name = name[:-3]
        add_to_possibility(out, name)

        #add solution 4
        new_pos = name.split('/')[-1]
        add_to_possibility(out, new_pos)

        #add solution 5
        new_pos = name.split('.')[-1]
        add_to_possibility(out, new_pos)

        return out

    def go_to_root(self):
        """ 
        go to the root directory of the module.
        This ensures that the script works correctly whatever the position
        where is launched
        """
        #self.launch_pos = os.path.realpath(os.getcwd())
        #self.root_path = root_path
        #os.chdir(root_path)

    def go_to_initpos(self):
        """ 
        go to the root directory of the module.
        This ensures that the script works correctly whatever the position
        where is launched
        """
        #os.chdir(self.launch_pos)
        #self.launch_pos = ''

if __name__ == "__main__":

    usage = "usage: %prog [expression1]... [expressionN] [options] "
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", default=1,
                      help="defined the verbosity level [%default]")
    parser.add_option("-r", "--reopt", type="int", default=0,
                  help="regular expression tag [%default]")
    parser.add_option("-p", "--path", default='tests/unit_tests',
                  help="position to start the search (from root)  [%default]")
    parser.add_option("-l", "--logging", default='CRITICAL',
        help="logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL) [%default]")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        args = ''

    if options.path == 'U':
        options.path = 'tests/unit_tests'
    elif options.path == 'P':
        options.path = 'tests/parallel_tests'
    elif options.path == 'A':
        options.path = 'tests/acceptance_tests'


    try:
        logging.config.fileConfig(os.path.join(root_path,'tests','.mg5_logging.conf'))
        logging.root.setLevel(eval('logging.' + options.logging))
        logging.getLogger('madgraph').setLevel(eval('logging.' + options.logging))
        logging.getLogger('cmdprint').setLevel(eval('logging.' + options.logging))
        logging.getLogger('tutorial').setLevel('ERROR')
    except:
        pass

    #logging.basicConfig(level=vars(logging)[options.logging])
    run(args, re_opt=options.reopt, verbosity=options.verbose, \
            package=options.path)
    
#some example
#    run('iolibs')
#    run('test_test_manager.py')
#    run('./tests/unit_tests/bin/test_test_manager.py')
#    run('IOLibsMiscTest')
#    run('TestTestFinder')
#    run('test_check_valid_on_file')
#    run('test_collect_dir.*') # '.*' stands for all possible char (re format)


