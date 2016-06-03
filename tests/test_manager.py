#!/usr/bin/env python 
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
    sys.exit('MadGraph5_aMC@NLO works only with python 2.6 or later (but not python 3.X).\n\
               Please upgrate your version of python.')

import inspect
import tarfile
import logging
import logging.config
import optparse
import os
import re
import unittest
import time
import datetime
import shutil
import glob
from functools import wraps

#Add the ROOT dir to the current PYTHONPATH
root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.insert(0, root_path)
# Only for profiling with -m cProfile!
#root_path = os.path.split(os.path.dirname(os.path.realpath(sys.argv[0])))[0]
#sys.path.append(root_path)

import tests.IOTests
import aloha
import aloha.aloha_lib as aloha_lib

from madgraph import MG4DIR
from madgraph.interface.extended_cmd import Cmd
from madgraph.iolibs.files import cp, ln, mv
import madgraph.various.misc as misc


#position of MG_ME
MGME_dir = MG4DIR

colored = "\x1b[1;%dm%s\x1b[0m"

IOTestManager = tests.IOTests.IOTestManager

path = os.path
pjoin = path.join

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = path.abspath(os.path.join(_file_path,'input_files'))
_hc_comparison_files = pjoin(_input_file_path,'IOTestsComparison')
_hc_comparison_tarball = pjoin(_input_file_path,'IOTestsComparison.tar.bz2')
_hc_comparison_modif_log = pjoin(_input_file_path,'IOTestsRefModifs.log')


class MyTextTestRunner(unittest.TextTestRunner):
    
    bypassed = []

    def run(self, test):
        "Run the given test case or test suite."
        keyboardstop=False
        MyTextTestRunner.stream = self.stream
        result = self._makeResult()
        startTime = time.time()
        try:
            test(result)
        except KeyboardInterrupt:
            keyboardstop=True
            pass
        except:
            raise
        stopTime = time.time()
        timeTaken = float(stopTime - startTime)
        result.printErrors()
        self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()
        if not result.wasSuccessful():
            self.stream.write("FAILED (")
            failed, errored = map(len, (result.failures, result.errors))
            if failed:
                self.stream.writeln("failures=%d)" % failed)
                self.stream.writeln(' '.join([str(t[0]).split()[0] for t in result.failures]))
            if errored:
                if failed: 
                    self.stream.write("FAILED ( ")
                self.stream.writeln(" errors=%d)" % errored)
            self.stream.writeln(' '.join([str(t[0]).split()[0] for t in result.errors]))
        else:
            self.stream.writeln("OK")
        if self.bypassed:
            self.stream.writeln("Bypassed %s:" % len(self.bypassed))
            self.stream.writeln(" ".join(self.bypassed))
        if keyboardstop:
            self.stream.writeln("Some of the tests Bypassed due to Ctrl-C")
        return result 

    def run_border(self, test, to_check):
        "Run the given test case or test suite."
        MyTextTestRunner.stream = self.stream
        result = self._makeResult()
        startTime = time.time()
        test(result)
        stopTime = time.time()
        timeTaken = float(stopTime - startTime)
        result.printErrors()
        self.stream.writeln(result.separator2)
        run = result.testsRun
        #self.stream.writeln("Ran %d test%s in %.3fs" %
        #                    (run, run != 1 and "s" or "", timeTaken))
        #self.stream.writeln()
        if not result.wasSuccessful():
            self.stream.write("FAILED (")
            failed, errored = map(len, (result.failures, result.errors))
            if failed:
                self.stream.write("failures=%d" % failed)
            if errored:
                if failed: self.stream.write(", ")
                self.stream.write("errors=%d" % errored)
            self.stream.writeln(")")
            print to_check
            to_check= to_check.rsplit('.',1)[1]
            print to_check
            if result.failures:
                print 'fail', to_check,[str(R[0]) for R in result.failures]
            if result.errors:
                print 'errors',to_check,[str(R[0]) for R in result.errors]

            if any(to_check in str(R[0]) for R in result.failures) or\
               any(to_check in str(R[0]) for R in result.errors):
                sys.exit(0)
        #else:
        #    self.stream.writeln("OK")
        #if self.bypassed:
        #    self.stream.writeln("Bypassed %s:" % len(self.bypassed))
        #    self.stream.writeln(" ".join(self.bypassed))
        return result 


            
#===============================================================================
# run
#===============================================================================
def run(expression='', re_opt=0, package='./tests/unit_tests', verbosity=1,
        timelimit=0):
    """ running the test associated to expression. By default, this launch all 
    test inherited from TestCase. Expression can be the name of directory, 
    module, class, function or event standard regular expression (in re format)
    """

    #init a test suite
    testsuite = unittest.TestSuite()
    collect = unittest.TestLoader()
    TestSuiteModified.time_limit =  float(timelimit)
    for test_fct in TestFinder(package=package, expression=expression, \
                                   re_opt=re_opt):
        data = collect.loadTestsFromName(test_fct)        
        assert(isinstance(data,unittest.TestSuite))        
        data.__class__ = TestSuiteModified
        testsuite.addTest(data)
        
    output =  MyTextTestRunner(verbosity=verbosity).run(testsuite)
    
    
    
    
    if TestSuiteModified.time_limit < 0:
        ff = open(pjoin(root_path,'tests','time_db'), 'w')
        ff.write('\n'.join(['%s %s' % a  for a in TestSuiteModified.time_db.items()]))
        ff.close()

    return output
    #import tests
    #print 'runned %s checks' % tests.NBTEST
    #return out

#===============================================================================
# run
#===============================================================================
def run_border_search(to_crash='',expression='', re_opt=0, package='./tests/unit_tests', verbosity=1,
        timelimit=0):
    """ running the test associated to expression one by one. and follow them by the to_crash one
        up to the time that to_crash is actually crashing. Then the run stops and print the list of the 
        routine tested. Then the code re-run itself(via a fork) to restrict the list. 
        The code stops when the list is of order 1. The order of the test is randomize at each level!
    """
    #init a test suite
    collect = unittest.TestLoader()
    TestSuiteModified.time_limit =  float(timelimit)
    all_test = TestFinder(package=package, expression=expression, re_opt=re_opt)
    import random
    random.shuffle(all_test)
    print "to_crash"
    to_crash = TestFinder(package=package, expression=to_crash, re_opt=re_opt)
    to_crash.collect_dir(package, checking=True)

    for test_fct in all_test:
        testsuite = unittest.TestSuite()
        data = collect.loadTestsFromName(test_fct)        
        assert(isinstance(data,unittest.TestSuite))        
        data.__class__ = TestSuiteModified
        testsuite.addTest(data)
        data = collect.loadTestsFromName(to_crash[0])        
        assert(isinstance(data,unittest.TestSuite))        
        data.__class__ = TestSuiteModified
        testsuite.addTest(data)
        # Running it
        print "run it for %s" % test_fct
        output =  MyTextTestRunner(verbosity=verbosity).run_border(testsuite, to_crash[0])
    
    return output
    #import tests
    #print 'runned %s checks' % tests.NBTEST
    #return out

#===============================================================================
# listIOTests
#===============================================================================

def listIOTests(arg=['']):
    """ Listing the IOtests associated to expression and returning them as a
    list of tuples (folderName,testName).     
    """
    
    if len(arg)!=1 or not isinstance(arg[0],str):
        print "Exactly one argument, and in must be a string, not %s."%arg
        return
    arg=arg[0]
    
    IOTestManager.testFolders_filter = arg.split('/')[0].split('&')
    IOTestManager.testNames_filter = arg.split('/')[1].split('&')    
    IOTestManager.filesChecked_filter = '/'.join(arg.split('/')[2:]).split('&')
 
    all_tests = []
    
    # The version below loads the test so it is slow because all tests are setUp
    # and therefore loaded. The other method is however less accurate because it
    # might be that the reference file have not been generated yet
#    for IOTestsClass in IOTestFinder():
#        IOTestsClass().setUp()
#    all_tests = IOTestManager.all_tests.keys()
    
    # Extract the tarball for hardcoded comparison if necessary
    if not path.isdir(_hc_comparison_files):
        if path.isfile(_hc_comparison_tarball):
            tar = tarfile.open(_hc_comparison_tarball,mode='r:bz2')
            tar.extractall(path.dirname(_hc_comparison_files))
            tar.close()
        else:
            os.makedirs(_hc_comparison_files)

    # We look through the uncompressed tarball for the name of the folders and
    # test. It is however less accurate since it might be that some test
    # reference folder have not been generated yet
    for dirPath in glob.glob(path.join(_hc_comparison_files,"*")):
        if path.isdir(dirPath): 
            folderName=path.basename(dirPath)
            for testPath in glob.glob(path.join(_hc_comparison_files,\
                                                               folderName,"*")):
                if path.isdir(testPath):
                    all_tests.append((folderName,path.basename(testPath)))                  

    return all_tests

#===============================================================================
# runIOTests
#===============================================================================
def runIOTests(arg=[''],update=True,force=0,synchronize=False):
    """ running the IOtests associated to expression. By default, this launch all 
    the tests created in classes inheriting IOTests.     
    """
    
    # Update the tarball, while removing the .backups.
    def noBackUps(tarinfo):
        if tarinfo.name.endswith('.BackUp'):
            return None
        else:
            return tarinfo
    
    if synchronize:
        print "Please, prefer updating the reference file automatically "+\
                                                          "rather than by hand."
        tar = tarfile.open(_hc_comparison_tarball, "w:bz2")
        tar.add(_hc_comparison_files, \
                  arcname=path.basename(_hc_comparison_files), filter=noBackUps)
        tar.close()
        # I am too lazy to work out the difference with the existing tarball and
        # put it in the log. So this is why one should refrain from editing the
        # reference files by hand.
        text = " \nModifications performed by hand on %s at %s in"%(\
                         str(datetime.date.today()),misc.format_timer(0.0)[14:])
        text += '\n   MadGraph5_aMC@NLO v. %(version)s, %(date)s\n'%misc.get_pkg_info()
        log = open(_hc_comparison_modif_log,mode='a')
        log.write(text)
        log.close()
        print "INFO:: Ref. tarball %s updated"%str(_hc_comparison_tarball)
            
        return
    
    if len(arg)!=1 or not isinstance(arg[0],str):
        print "Exactly one argument, and in must be a string, not %s."%arg
        return
    arg=arg[0]

    # Extract the tarball for hardcoded comparison if necessary
    
    if not path.isdir(_hc_comparison_files):
        if path.isfile(_hc_comparison_tarball):
            tar = tarfile.open(_hc_comparison_tarball,mode='r:bz2')
            tar.extractall(path.dirname(_hc_comparison_files))
            tar.close()
        else:
            os.makedirs(_hc_comparison_files)
    
    # Make a backup of the comparison file directory in order to revert it if
    # the user wants to ignore the changes detected (only when updating the refs)
    hc_comparison_files_BackUp = _hc_comparison_files+'_BackUp'
    if update and path.isdir(_hc_comparison_files):
        if path.isdir(hc_comparison_files_BackUp):        
            shutil.rmtree(hc_comparison_files_BackUp)
        shutil.copytree(_hc_comparison_files,hc_comparison_files_BackUp)

    IOTestManager.testFolders_filter = arg.split('/')[0].split('&')
    IOTestManager.testNames_filter = arg.split('/')[1].split('&')
    IOTestManager.filesChecked_filter = '/'.join(arg.split('/')[2:]).split('&')
    #print "INFO:: Using folders %s"%str(IOTestManager.testFolders_filter)    
    #print "INFO:: Using test names %s"%str(IOTestManager.testNames_filter)         
    #print "INFO:: Using file paths %s"%str(IOTestManager.filesChecked_filter)
    
    # Initiate all the IOTests from all the setUp()
    IOTestsInstances = []
    start = time.time()
    for IOTestsClass in IOTestFinder():
        # Instantiate the class
        IOTestsInstances.append(IOTestsClass())
        # Run the setUp
        IOTestsInstances[-1].setUp()
        # Find the testIO defined and use them in load mode only, we will run
        # them later here.
        IOTestsFunctions = IOTestFinder()
        IOTestsFunctions.collect_function(IOTestsClass,prefix='testIO')
        for IOTestFunction in IOTestsFunctions:
            start = time.time()
            # Add all the tests automatically (i.e. bypass filters) if the 
            # specified test is the name of the IOtest. the [7:] is to
            # skip the testIO prefix
            name_filer_bu = None         
            if IOTestFunction.split('.')[-1][7:] in \
                                                 IOTestManager.testNames_filter:
                name_filer_bu = IOTestManager.testNames_filter
                IOTestManager.testNames_filter = ['ALL']
                existing_tests = IOTestManager.all_tests.keys()
                
            eval('IOTestsInstances[-1].'+IOTestFunction.split('.')[-1]+\
                                                             '(load_only=True)')
            if name_filer_bu:
                new_tests = [test[0] for test in IOTestManager.all_tests.keys() \
                                                  if test not in existing_tests]
                IOTestManager.testNames_filter = name_filer_bu + new_tests
                name_filer_bu = None
            
            setUp_time = time.time() - start
            if setUp_time > 0.5:                
                print colored%(34,"Loading IOtest %s is slow (%s)"%
                        (colored%(32,'.'.join(IOTestFunction.split('.')[-3:])),
                                             colored%(34,'%.2fs'%setUp_time)))
    
    if len(IOTestsInstances)==0:
        print "No IOTest found."
        return
    
    # runIOTests cannot be made a classmethod, so I use an instance, but it does 
    # not matter which one as no instance attribute will be used.
    try:
        modifications = IOTestsInstances[-1].runIOTests( update = update, 
           force = force, verbose=True, testKeys=IOTestManager.all_tests.keys())
    except KeyboardInterrupt:
        if update:
            # Remove the BackUp of the reference files.
            if not path.isdir(hc_comparison_files_BackUp):
                print "\nWARNING:: Update interrupted and modifications already "+\
                                              "performed could not be reverted."
            else:
                shutil.rmtree(_hc_comparison_files)
                mv(hc_comparison_files_BackUp,_hc_comparison_files)
                print colored%(34,
                "\nINFO:: Update interrupted, existing modifications reverted.")
            sys.exit(0)
        else:
            print "\nINFO:: IOTest runs interrupted."
            sys.exit(0)
 
    tot_time = time.time() - start
    
    if modifications == 'test_over':
        print colored%(32,"\n%d IOTests "%len(IOTestManager.all_tests.keys()))+\
                    "successfully tested in %s."%(colored%(34,'%.2fs'%tot_time))
        sys.exit(0)
    elif not isinstance(modifications,dict):
        print "Error during the files update."
        sys.exit(0)

    if len(modifications['missing'])>0:
        text = '\n'
        text += colored%(31,
               "The following files were not generated by the tests, fix this!")        
        text += '\n'+'\n'.join(["   %s"%mod for mod in modifications['missing']])
        print text
        modifications['missing'] = []

    if sum(len(v) for v in modifications.values())>0:
        # Display the modifications
        text = colored%(34, " \nModifications performed on %s at %s in"%(\
                        str(datetime.date.today()),misc.format_timer(0.0)[14:]))
        text += colored%(34, 
        '\n   MadGraph5_aMC@NLO v. %(version)s, %(date)s\n'%misc.get_pkg_info())
        for key in modifications.keys():
            if len(modifications[key])==0:
                continue
            text += colored%(32,"The following reference files have been %s :"%key)
            text += '\n'+'\n'.join(["   %s"%mod for mod in modifications[key]])
            text += '\n'
        print text
        try:
            answer = Cmd.timed_input(question=
          "Do you want to apply the modifications listed above? [y/n] >",default="y")
        except KeyboardInterrupt:
            answer = 'n'

        if answer == 'y':
            log = open(_hc_comparison_modif_log,mode='a')
            log.write(text)
            log.close()
            if IOTestManager._compress_ref_fodler:
                tar = tarfile.open(_hc_comparison_tarball, "w:bz2")
                tar.add(_hc_comparison_files, \
                      arcname=path.basename(_hc_comparison_files), filter=noBackUps)
                tar.close()
                print colored%(32,"INFO:: tarball %s updated"%str(_hc_comparison_tarball))
            else:
                print colored%(32,"INFO:: Reference %s updated"%\
                                    str(os.path.basename(_hc_comparison_files)))
                if len(modifications['created'])>0:
                    print colored%(31,"Some ref. files have been created; add "+\
                      "them to the revision with\n  "+
                      "bzr add tests/input_files/IOTestsComparison")
           
                # Make sure to remove the BackUp files
                filelist = glob.glob(os.path.join(_hc_comparison_files,
                                                            '*','*','*.BackUp'))
                for f in filelist:
                    os.remove(f)
        else:
            if path.isdir(hc_comparison_files_BackUp):
                shutil.rmtree(_hc_comparison_files)
                shutil.copytree(hc_comparison_files_BackUp,_hc_comparison_files)
                print colored%(32,"INFO:: No modifications applied.")
            else:
                print colored%(31,
                 "ERROR:: Could not revert the modifications. No backup found.")
    else:
        print colored%(32,"\nNo modifications performed. No update necessary.")
    
    # Remove the BackUp of the reference files.
    if path.isdir(hc_comparison_files_BackUp):
        shutil.rmtree(hc_comparison_files_BackUp)

class TimeLimit(Exception): pass
#===============================================================================
# TestSuiteModified
#===============================================================================
class TestSuiteModified(unittest.TestSuite):
    """ This is a wrapper for the default implementation of unittest.TestSuite 
    so that we can add the decorator for the resetting of the global variables
    everytime the TestSuite is __call__'ed., hence avoiding side effects from 
    them."""
    
    time_limit = 1
    time_db = {}

    @tests.IOTests.set_global()
    def __call__(self, *args, **kwds):
    
        time_db = TestSuiteModified.time_db
        time_limit = TestSuiteModified.time_limit
        if not time_db and time_limit > 0:
            if not os.path.exists(pjoin(root_path, 'tests','time_db')):
                TestSuiteModified.time_limit = -1
            else:
                #for line in open(pjoin(root_path, 'tests','time_db')):
                #        print line.split()
                TestSuiteModified.time_db = dict([(' '.join(line.split()[:-1]), float(line.split()[-1]))  
                         for line in open(pjoin(root_path, 'tests','time_db'))
                         ])
                time_db = TestSuiteModified.time_db
            
        if str(self) in time_db and time_db[str(self)] > abs(time_limit):
            MyTextTestRunner.stream.write('T')
            #print dir(self._tests[0]), type(self._tests[0]),self._tests[0] 
            MyTextTestRunner.bypassed.append(str(self._tests[0]).split()[0])
            return

        
        start = time.time()
        super(TestSuiteModified,self).__call__(*args,**kwds)
        if not str(self) in time_db:
            TestSuiteModified.time_db[str(self)] = time.time() - start
            TestSuiteModified.time_limit *= -1
        
        
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
            start = time.time()
            self.collect_dir(self.package, checking=True)
            print 'loading test takes %ss'  % (time.time()-start)
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
        
        start = time.time()
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
                
        time_to_load = time.time() - start
        if time_to_load > 0.1:
            logging.critical("file %s takes a long time to load (%.4fs)" % (pyname, time_to_load))

    def collect_function(self, class_, checking=True, base='', prefix='test'):
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
                       name.startswith(prefix)\
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

#===============================================================================
# IOTestFinder
#===============================================================================
class IOTestFinder(TestFinder):
    """ Class introspecting the test modules to find the available IOTest classes.
    The routine collect_dir looks in all module/file to find the different 
    functions in different test class. This produce a list, on which external 
    routines can loop on. 
        
    In order to authorize definition and loop on this object on the same time,
    i.e: for test in TestFinder([opt])-. At each time a loop is started, 
    we check if a collect_dir ran before, and run it if necessary.
    """
    class IOTestFinderError(Exception):
        """Error associated to the TestFinder class."""
        pass

    def __init__(self, package='tests/', expression='', re_opt=0):
        """ initialize global variable for the test """
        if expression!='' or re_opt!=0:
            raise IOTestFinderError('Only use IOTestFinder for searching for'+\
                                                                 ' all classes')
        super(IOTestFinder,self).__init__(package,expression,re_opt)

    def collect_file(self, filename, checking=True):
        """ Find the different class instance derivated of TestCase """
        
        start = time.time()
        pyname = self.passin_pyformat(filename)
        __import__(pyname)
        obj = sys.modules[pyname]
        #look at class
        for name in dir(obj):
            class_ = getattr(obj, name)
            if inspect.isclass(class_) and class_!=tests.IOTests.IOTestManager and \
                                issubclass(class_, tests.IOTests.IOTestManager):
                self.append(class_)

        time_to_load = time.time() - start
        if time_to_load > 0.1:
            logging.critical("file %s takes a long time to load (%.4fs)" % \
                                                         (pyname, time_to_load))

if __name__ == "__main__":

    help = """
    Detailed information about the IOTests at the wiki webpage:
https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/DevelopmentPage/CodeTesting

    Use the argument -U to update the hardcoded tests used by the IOTests.
    When provided with no argument, it will update everything.
    Otherwise  it can be called like this:
    
                ./test_manager.py -U "folders/testNames/filePaths"

    the arguments between '/' are specified according to this format
    (For each of the three category, you can use the keyword 'ALL' to select 
     all of the IOTests in this category)

           folders   -> "folder1&folder2&folder3&etc..."
           testNames -> "testName1&testName2&testName3&etc..."
           filePaths -> "filePath1&filePath2&filePath3&etc..."    
    
    Notice that the filePath use a file path relative to
    the position SubProcess/<P0_proc_name>/ in the output.
    You are allowed to use the parent directory specification ".."
    You can use the synthax [regexp] instead of a specific filename.
    This includes only the files in this directory matching it.
    > Ex. '../../Source/DHELAS/[.+\.(inc|f)]' matches any file in DHELAS
    with extension .inc or .f
    Also, you can prepend '-' to the folder or test name to veto it instead of
    selecting it.
    > Ex. '-longTest' considers all tests but the one named
    'longTest' one (synthax not available for filenames).
    If you prepend '+' to the folder or test name, then you will include all 
    items in this category which starts with what follows '+'.
    > Ex. '+short' includes all IOTests starting with 'short'
    To bypass the monitoring of the modifications of the files with a name of
    a file already reviewed, you can use -f. To bypass ALL monitoring, use -F
    (this is not recommended).
    
    Finally, you can run the test only from here too. Same synthax as above,
    but use the option -R
    
    And you can quickly run a given IOTest with
        ./test_manager.py -R testName
    or all IOtest of a give group with
        ./test_manager.py -R -g groupName
        
    You can list all tests in the reference folder with
        ./test_manager.py -L
    while possibly also specifying what kind of test to list with the same
    syntax as for -R or -U, i.e.
        ./test_manager.py -L +short/ALL/ALL
    for example.
    """

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
    parser.add_option("-F", "--force", action="store_true", default=False,
        help="Force the update, bypassing its monitoring by the user")
    parser.add_option("-f", "--semiForce", action="store_true", default=False,
        help="Bypass monitoring of ref. file only if another ref. file with "+\
                                    "the same name has already been monitored.")
    parser.add_option("-U", "--IOTestsUpdate", action="store_true", default=False,
          help="Process the IOTests to update them.")
    parser.add_option("-R", "--IOTestsRun", action="store_true", default=False,
          help="Process the IOTests to Run them.")
    parser.add_option("-L", "--ListIOTests", action="store_true", default=False,
          help="Lists all IOTests in the reference Folder.")
    parser.add_option("-g", "--group", action="store_true", default=False,
        help="Specifies you want to run all tests belonging to a group of a "+\
                                                                  "given name.")    
    parser.add_option("-s", "--synchronize", action="store_true", default=False,
          help="Replace the IOTestsComparison.tar.bz2 tarball with the "+\
                                      "content of the folder IOTestsComparison")
    parser.add_option("-t", "--timed", default="Auto",
          help="limit the duration of each test. Negative number re-writes the information file.")    

    parser.add_option("", "--border_effect", default=None,
          help="Define the test which are sensitive to a border effect, the test will find which test creates this border effect")        

    parser.add_option("-N", "--notification", default=45,
          help="Running time, below which no notification is raised. (-1 for no notification)")        

    
    (options, args) = parser.parse_args()

    if options.IOTestsUpdate:
        options.IOTests = 'U'
    elif options.IOTestsRun:
        options.IOTests = 'R'
    else:
        options.IOTests = 'No'

    if options.ListIOTests:
        options.IOTests = 'L'        

    if options.IOTests=='No':
        if len(args) == 0:
            args = ''
    else:
        if len(args)>1:
            print "Specify at most one argument to specify what IOTests to run."
        if len(args) == 0:
            args = ['ALL/ALL/ALL']

        if options.group:
            if len(args)==1:
                args = ['%s/ALL/ALL'%str(args[0])]
            else:
                print "Specify the name of the IOTest group you want to run."
        else:
            specs = args[0].split('/')
            if len(specs)==1:
                test_name = args[0] if args[0][:7]!='testIO_' else args[0][7:]
                args = ['ALL/%s/ALL'%test_name]
            elif len(specs)==2:
                args = ['%s/%s/ALL'%(specs[0],
                           (specs[1] if specs[1][:7]!='testIO_' else specs[1]))]
            elif len(specs)==3:
                args = ['%s/%s/%s'%(specs[0],
                  (specs[1] if specs[1][:7]!='testIO_' else specs[1]),specs[2])]
            else:
                print "The IOTest specification can include at most two '/'."                 

    if len(args) == 1 and args[0]=='help':
        print help
        sys.exit(0)

    if options.path == 'U':
        options.path = 'tests/unit_tests'
    elif options.path == 'P':
        options.path = 'tests/parallel_tests'
    elif options.path == 'A':
        options.path = 'tests/acceptance_tests'
        
    if options.timed == "Auto":
        if options.path == 'tests/unit_tests':
            options.timed = 1
        elif options.path == 'tests/parallel_tests':
            options.timed = 400
        elif options.path == 'tests/acceptance_tests':
            options.timed = 10
        else:
            options.timed = 0 

    start_time = time.time()

    try:
        logging.config.fileConfig(os.path.join(root_path,'tests','.mg5_logging.conf'))
        logging.root.setLevel(eval('logging.' + options.logging))
        logging.getLogger('madgraph').setLevel(eval('logging.' + options.logging))
        logging.getLogger('cmdprint').setLevel(eval('logging.' + options.logging))
        logging.getLogger('tutorial').setLevel('ERROR')
    except:
        pass
    
    if options.synchronize and IOTestManager._compress_ref_fodler:
        print "The tarball synchronization is not necessary since"+ \
          " MadGraph5_aMCatNLO is configured not to compress the references files."
    
    if options.IOTests=='No' and not options.synchronize:
        if not options.border_effect:
            #logging.basicConfig(level=vars(logging)[options.logging])
            output = run(args, re_opt=options.reopt, verbosity=options.verbose, \
                package=options.path, timelimit=options.timed)
        else:
            output = run_border_search(options.border_effect, args, re_opt=options.reopt, verbosity=options.verbose, \
                package=options.path, timelimit=options.timed)
    else:
        if options.IOTests=='L':
            print "Listing all tests defined in the reference files ..."
            print '\n'.join("> %s/%s"%(colored%(34,test[0]),colored%(32,test[1]))
            for test in listIOTests(args) if IOTestManager.need(test[0],test[1]))
            exit()
        if options.force:
            force = 10
        elif options.semiForce:
            force = 1
        else:
            force = 0 

        output = runIOTests(args,update=options.IOTests=='U',force=force,
                                                synchronize=options.synchronize)


    if 0 < float(options.notification) < time.time()-start_time:
        if isinstance(output, unittest.runner.TextTestResult):
            run = output.testsRun
            failed, errored, skipped = map(len, 
                               (output.failures, output.errors, output.skipped))
            output = "run: %s, failed: %s error: %s, skipped: %s" % \
                                                 (run, failed, errored, skipped)
        misc.apple_notify("tests finished", str(output))
#some example
#    run('iolibs')
#    run('test_test_manager.py')
#    run('./tests/unit_tests/bin/test_test_manager.py')
#    run('IOLibsMiscTest')
#    run('TestTestFinder')
#    run('test_check_valid_on_file')
#    run('test_collect_dir.*') # '.*' stands for all possible char (re format)
#    python tests/test_manager.py test_decay.py -l INFO|less


