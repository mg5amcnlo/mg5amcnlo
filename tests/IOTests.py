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

import copy
import os
import sys
import shutil
import re
import glob
import tarfile
import datetime
import unittest
import subprocess
import pydoc
import tempfile
from functools import wraps 

import aloha
import aloha.aloha_lib as aloha_lib

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(root_path)

import madgraph.various.misc as misc

import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.loop.loop_exporters as loop_exporters

from madgraph.interface.extended_cmd import Cmd

from madgraph.iolibs.files import cp, ln, mv
from madgraph import MadGraph5Error

pjoin = os.path.join
path = os.path

colored = "\x1b[1;%dm%s\x1b[0m"

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = path.abspath(os.path.join(_file_path,'input_files'))
_hc_comparison_files = pjoin(_input_file_path,'IOTestsComparison')
_hc_comparison_tarball = pjoin(_input_file_path,'IOTestsComparison.tar.bz2')


def set_global(loop=False, unitary=True, mp=False, cms=False):
    """This decorator set_global() which make sure that for each test
    the global variable are returned to their default value. This decorator can
    be modified with the new global variables to come and will potenitally be
    different than the one in test_aloha."""
    def deco_set(f):
        @wraps(f)
        def deco_f_set(*args, **opt):
            old_loop = aloha.loop_mode
            old_gauge = aloha.unitary_gauge
            old_mp = aloha.mp_precision
            old_cms = aloha.complex_mass
            aloha.loop_mode = loop
            aloha.unitary_gauge = unitary
            aloha.mp_precision = mp
            aloha.complex_mass = cms
            aloha_lib.KERNEL.clean()
            try:
                out =  f(*args, **opt)
            except:
                aloha.loop_mode = old_loop
                aloha.unitary_gauge = old_gauge
                aloha.mp_precision = old_mp
                aloha.complex_mass = old_cms
                raise
            aloha.loop_mode = old_loop
            aloha.unitary_gauge = old_gauge
            aloha.mp_precision = old_mp
            aloha.complex_mass = old_cms
            aloha_lib.KERNEL.clean()
            return out
        return deco_f_set
    return deco_set

class IOTest(object):
    """ IOTest runner and attribute container. It can be overloaded depending on
    what kind of IO test will be necessary later """

    # Handy definitions
    proc_files = ['[^.+\.(f|dat|inc)$]','MadLoop5_resources/[^ML5_.*\.dat]']
    # Some model files are veto because they are sourced by dictionaries whose 
    # order is random.
    model_files = ['../../Source/MODEL/[^.+\.(f|inc)$]',
                   '-../../Source/MODEL/lha_read.f',
                   '-../../Source/MODEL/param_read.inc',
                   '-../../Source/MODEL/param_write.inc']            
    helas_files = ['../../Source/DHELAS/[^.+\.(f|inc)$]']
    
    # We also exclude the helas_files because they are sourced from unordered
    # dictionaries.
    all_files = proc_files+model_files

    def __init__(self, test_instance=None,
                       procdef=None,
                       exporter=None,
                       helasModel=None,
                       testedFiles=None,
                       outputPath=None):
        """ Can be overloaded to add more options if necessary.
        The format above is typically useful because we don't aim at
        testing all processes for all exporters and all model, but we 
        choose certain combinations which spans most possibilities.
        Notice that the process and model can anyway be recovered from the 
        LoopAmplitude object, so it does not have to be specified here."""

        if testedFiles is None:
            raise MadGraph5Error, "TestedFiles must be specified in IOTest."
        
        if outputPath is None:
            raise MadGraph5Error, "outputPath must be specified in IOTest."
        
        self.testedFiles = testedFiles
        self.test_instance = test_instance
        self.procdef = procdef
        self.helasModel = helasModel
        self.exporter_name = exporter
        # Security mesure
        if not str(path.dirname(_file_path)) in str(outputPath) and \
                                        not str(outputPath).startswith('/tmp/'):
            raise MadGraph5Error, "OutputPath must be within MG directory or"+\
                                                                     " in /tmp/"            
        else:
            self.outputPath = outputPath

    @set_global() 
    def run(self, IOTestManagerInstance=None):
        """ Run the test and returns the path where the files have been 
        produced and relative to which the paths in TestedFiles are specified. """
        self.clean_output()

        model = self.procdef.get('model')
        self.exporter = self.test_instance.get_exporter_withName(\
                                                            self.exporter_name)
        myloopamp = loop_diagram_generation.LoopAmplitude(self.procdef)
        isOptimized = isinstance(self.exporter, \
                           loop_exporters.LoopProcessOptimizedExporterFortranSA) 
        hel_amp=loop_helas_objects.LoopHelasMatrixElement(\
                                        myloopamp,optimized_output=isOptimized)

        self.exporter.copy_v4template(model.get('name'))
        self.exporter.generate_loop_subprocess(hel_amp, self.helasModel)
        wanted_lorentz = hel_amp.get_used_lorentz()
        wanted_couplings = list(set(sum(hel_amp.get_used_couplings(),[])))
        self.exporter.convert_model_to_mg4(model,wanted_lorentz,wanted_couplings)
            
        proc_name='P'+hel_amp.get('processes')[0].shell_string()
        return pjoin(self.outputPath,'SubProcesses',proc_name)
    
    def clean_output(self,IOTestManagerInstance=None):
        """ Remove the output_path if existing. Careful!"""
        if not str(path.dirname(_file_path)) in str(self.outputPath) and \
                                   not str(self.outputPath).startswith('/tmp/'):
            raise MadGraph5Error, "Cannot safely remove %s."%str(self.outputPath)
        else:
            if path.isdir(self.outputPath):
                shutil.rmtree(self.outputPath)

# The decorator below allows for easily creating a new CustomIOTest by just
# wrapping a test function whose name starts with testIO which generates files
# to be compared against reference ones. It should return a base_path from where
# the relative path of the list of files to be compared points. See the
# example in <MG_root>/tests/unit_tests/test_IOTest_example.py for more details
def createIOTest(groupName=None, testName=None):

    def createIOTest_decorator(GenerateFilefunc):
        """Decorator easing the creation of a new IOTest"""
        
        filesToCheck = []
        prevent_cleanUp = False
        pathsToErase = []
        targetFinder = re.compile(r"(?P<type>(target|clean)):\s*(?P<file>\S*)")
        for target in re.finditer(targetFinder, GenerateFilefunc.__doc__):
            if target.group('type')=='target':
                filesToCheck.append(target.group('file'))
            if target.group('type')=='clean':
                if target.group('file')=='False':
                    prevent_cleanUp = True
                else:
                    pathsToErase.append(target.group('file'))

        def pathsToClean():                
            return pathsToErase, prevent_cleanUp
        
        @wraps(GenerateFilefunc)
        def __wrapper(*args, **kwargs):
            
            if testName==None:
                # We know that the function must start with "testIO_"
                newTestName = GenerateFilefunc.__name__[7:]
            else:
                newTestName=testName

            MyCustomTest = CustomIOTest(filesToCheck,GenerateFilefunc, 
                                                                  pathsToClean)
            # The first argument should be self
            if groupName==None:
                testGroup=args[0].__class__.__name__
            else:
                testGroup=groupName
            args[0].addIOTest(testGroup,newTestName,MyCustomTest)
    
            if 'load_only' in kwargs and kwargs['load_only']:
                return
    
            # Feel free to change the arguments here to modify how the IOTest are
            # run when launched with directly from the test_manager without -i.
            args[0].runIOTests(update=False, force=10, verbose=False, 
                                            testKeys=[(testGroup, newTestName)])
    
        return __wrapper
    return createIOTest_decorator

class CustomIOTest(IOTest):
    """ Allow to simply implement individual IOTests"""
    
    # One must provide here the list of paths of the files to check relative
    # to the absolute path returned by run()
    testedFiles = []
    
    run_function = None
    clean_function = None
    temporary_folder = None    
    
    def __init__(self, files, run_f, clean_f = None):
        """ Initialize the custom IOTest with the three necessary components"""

        self.testedFiles = files
        self.run_function = run_f
        self.clean_function = clean_f

    @set_global()
    def run(self, *args, **kwargs):
        """This function must perform actions to create the files and return
        the absolute path from which the paths in the variable all_files are
        defined. Also stores here the temporary folder in which files are
        created."""
        
        args[0].IOpath = tempfile.mkdtemp('', 'TMPIOTest', None)
        self.temporary_folder = args[0].IOpath
        custom_path = self.run_function(*args, **kwargs)
        if custom_path is None:
            return args[0].IOpath
        else:
            return custom_path        
    
    def clean_output(self, *args, **kwargs):
        """Clean up the file created. Called at the end of the test run."""
        
        pathsToClean = [self.temporary_folder]
        
        if not self.clean_function is None:
            paths, prevent_cleanUp = self.clean_function(*args, **kwargs)
            pathsToClean.extend(paths)
            if prevent_cleanUp:
                print colored%(31,
                    "Clean up of the following of temporary folders prevented:")
                for path in pathsToClean:
                    print colored%(31,"  > %s"%str(path))

        try:
            for path in pathsToClean:
                shutil.rmtree(path)
        except OSError:
            pass               

#===============================================================================
# IOTestManager
#===============================================================================
class IOTestManager(unittest.TestCase):
    """ A helper class to perform tests based on the comparison of files output 
    by exporters against hardcoded ones. """
    
    # Global variable to decide if the reference folder has to be compressed or not
    _compress_ref_fodler = False
    
    # Define a bunch of paths useful
    _input_file_path = path.abspath(os.path.join(_file_path,'input_files'))
    _mgme_file_path = path.abspath(os.path.join(_file_path, *([os.path.pardir]*1)))
    _loop_file_path = pjoin(_mgme_file_path,'Template','loop_material')
    _cuttools_file_path = pjoin(_mgme_file_path, 'vendor','CutTools')
    _hc_comparison_files = pjoin(_input_file_path,'IOTestsComparison')

    # The tests loaded are stored here
    # Each test is stored in a dictionary with entries of the format:
    # {(folder_name, test_name) : IOTest}      
    all_tests = {}
    
    # filesChecked_filter allows to filter files which are checked.
    # These filenames should be the path relative to the
    # position SubProcess/<P0_proc_name>/ in the output. Notice that you can
    # use the parent directory keyword ".." and instead of the filename you
    # can exploit the syntax [regexp] (brackets not part of the regexp)
    # Ex. ['../../Source/DHELAS/[.+\.(inc|f)]']
    # You can also prepend '-' to a filename to veto it (it cannot be a regexp
    # in this case though.)
    filesChecked_filter = ['ALL']
    # To filter what tests you want to use, edit the tag ['ALL'] by the
    # list of test folders and names you want in.
    # You can prepend '-' to the folder or test name to veto it instead of
    # selecting it. Typically, ['-longTest'] considers all tests but the
    # longTest one (synthax not available for filenames)
    testFolders_filter = ['ALL']
    testNames_filter = ['ALL'] 
    
    def __init__(self,*args,**opts):
        """ Add the object attribute my_local_tests."""
        # Lists the keys for the tests of this particular instance
        self.instance_tests = []
        super(IOTestManager,self).__init__(*args,**opts)    
    
    def setUp(self):
        """ Dummy function possibly overloaded by the daughters """
        pass
    
    def runTest(self,*args,**opts):
        """ This method is added so that one can instantiate this class """
        raise MadGraph5Error, 'runTest in IOTestManager not supposed to be called.'
    
    def assertFileContains(self, source, solution):
        """ Check the content of a file """
        list_cur=source.read().split('\n')
        list_sol=solution.split('\n')
        while 1:
            if '' in list_sol:
                list_sol.remove('')
            else:
                break
        while 1:
            if '' in list_cur:
                list_cur.remove('')
            else:
                break            
        for a, b in zip(list_sol, list_cur):
            self.assertEqual(a,b)
        self.assertEqual(len(list_sol), len(list_cur))

    @classmethod
    def need(cls,folderName=None, testName=None):
        """ Returns True if the selected filters do not exclude the testName
        and folderName given in argument. Specify None to disregard the filters
        corresponding to this category."""
        
        if testName is None and folderName is None:
            return True
        
        if not testName is None:
            pattern = [f[1:] for f in cls.testNames_filter if f.startswith('+')]
            chosen = [f for f in cls.testNames_filter if \
                            not f.startswith('-') and not f.startswith('+')]
            veto = [f[1:] for f in cls.testNames_filter if f.startswith('-')]
            if testName in veto:
                return False
            if chosen!=['ALL'] and not testName in chosen:
                if not any(testName.startswith(pat) for pat in pattern):
                    return False

        if not folderName is None:
            pattern = [f[1:] for f in cls.testFolders_filter if f.startswith('+')]
            chosen = [f for f in cls.testFolders_filter if \
                            not f.startswith('-') and not f.startswith('+')]
            veto = [f[1:] for f in cls.testFolders_filter if f.startswith('-')]
            if folderName in veto:
                return False
            if chosen!=['ALL'] and not folderName in chosen:
                if not any(folderName.startswith(pat) for pat in pattern):
                    return False
        
        if not folderName is None and not testName is None:
            if (folderName,testName) in cls.all_tests.keys() and \
               (folderName,testName) in cls.instance_tests:
                return False

        return True

    @classmethod
    def toFileName(cls, file_path):
        """ transforms a file specification like ../../Source/MODEL/myfile to
        %..%..%Source%MODEL%myfile """
        fpath = copy.copy(file_path)
        if not isinstance(fpath, str):
            fpath=str(fpath)
        if '/' not in fpath:
            return fpath
        
        return '%'+'%'.join(file_path.split('/'))

    @classmethod        
    def toFilePath(cls, file_name):
        """ transforms a file name specification like %..%..%Source%MODEL%myfile
        to ../../Source/MODEL/myfile"""
        
        if not file_name.startswith('%'):
            return file_name
        
        return pjoin(file_name[1:].split('%'))
    
#    def test_IOTests(self):
#        """ A test function in the mother so that all childs automatically run
#        their tests when scanned with the test_manager. """
#        
#        # Set it to True if you want info during the regular test_manager.py runs
#        self.runIOTests(verbose=False)
    
    def addIOTest(self, folderName, testName, IOtest):
        """ Add the test (folderName, testName) to class attribute all_tests. """
        
        if not self.need(testName=testName, folderName=folderName):
            return

        # Add this to the instance test_list
        if (folderName, testName) not in self.instance_tests:
            self.instance_tests.append((folderName, testName))
            
        # Add this to the global test_list            
        if (folderName, testName) in self.all_tests.keys() and \
                          self.all_tests[(folderName, testName)]!=(IOtest,self):
            raise MadGraph5Error, \
                          "Test (%s,%s) already defined."%(folderName, testName)
        else:
            # We store the manager with self here too because it might have
            # variables related to its IOTests stored in it so that we will
            # give this instance back when calling IOtest.run(self).
            self.all_tests[(folderName, testName)] = (IOtest, self)

    def runIOTests(self, update = False, force = 0, verbose=False, \
                                                       testKeys='instanceList'):
        """ Run the IOTests for this instance (defined in self.instance_tests)
            and compare the files of the chosen tests against the hardcoded ones
            stored in tests/input_files/IOTestsComparison. If you see
            an error in the comparison and you are sure that the newest output
            is correct (i.e. you understand that this modification is meant to
            be so). Then feel free to automatically regenerate this file with
            the newest version by doing 
            
              ./test_manager -i U folderName/testName/fileName
                
            If update is True (meant to be used by __main__ only) then
            it will create/update/remove the files instead of testing them.
            The argument tests can be a list of tuple-keys describing the tests
            to cover. Otherwise it is the instance_test list.
            The force argument must be 10 if you do not want to monitor the 
            modifications on the updated files. If it is 0 you will monitor
            all modified file and if 1 you will monitor each modified file of
            a given name only once.
        """
        # First make sure that the tarball need not be untarred
        # Extract the tarball for hardcoded in all cases to make sure the 
        # IOTestComparison folder is synchronized with it.
        if IOTestManager._compress_ref_fodler:
            if path.isdir(_hc_comparison_files):
                try:
                    shutil.rmtree(_hc_comparison_files)
                except IOError:
                    pass
            if path.isfile(_hc_comparison_tarball):
                tar = tarfile.open(_hc_comparison_tarball,mode='r:bz2')
                tar.extractall(path.dirname(_hc_comparison_files))
                tar.close()
            else:
                raise MadGraph5Error, \
              "Could not find the comparison tarball %s."%_hc_comparison_tarball
        else:
            if not path.isdir(_hc_comparison_files):
                raise MadGraph5Error, \
              "Could not find the comparison tarball %s."%_hc_comparison_tarball

        # In update = True mode, we keep track of the modification to 
        # provide summary information
        modifications={'updated':[],'created':[], 'removed':[], 'missing':[]}
        
        # List all the names of the files for which modifications have been
        # reviewed at least once.The approach taken here is different than
        # with the list refusedFolder and refusedTest.
        # The key of the dictionary are the filenames and the value are string
        # determining the user answer for that file.
        reviewed_file_names = {}
        
        # Chose what test to cover
        if testKeys == 'instanceList':
            testKeys = self.instance_tests
        
        if verbose: print "\n== "+colored%(32,"Operational mode")+\
            " : file %s ==\n"%(colored%(34,('UPDATE' if update else 'TESTING')))
        for (folder_name, test_name) in testKeys:
            try:
                (iotest, iotestManager) = self.all_tests[(folder_name, test_name)]
            except KeyError:
                raise MadGraph5Error, 'Test (%s,%s) could not be found.'\
                                                       %(folder_name, test_name)
            if verbose: print "Processing %s in %s"%(
                                colored%(32,test_name),colored%(34,folder_name))
            
            files_path = iotest.run(iotestManager)
            try:
                pass
#                files_path = iotest.run(iotestManager)
            except Exception as e: 
                iotest.clean_output()
                if not verbose:
                    raise e
                else:
                    print colored%(31,"  Test %s "%test_name+\
                              "crashed with the following error:\n  %s."%str(e))
                    continue

            # First create the list of files to check as the user might be using
            # regular expressions.
            filesToCheck=[]
            # Store here the files reckognized as veto rules (with filename
            # starting with '-')
            veto_rules = []
            for fname in iotest.testedFiles:
                # Disregard the veto rules
                regexp_finder = re.compile(
                r'^(?P<veto>-)?(?P<root_folder>.*)(\/)?\[(?P<regexp>.*)\]$')
                found = regexp_finder.search(fname)
                if not found is None:
                    # folder without the final /
                    base_path = pjoin(files_path,found.group('root_folder'))
                    regexp = re.compile(found.group('regexp'))
                    # In filesToCheck, we must remove the files_path/ prepended
                    for root, dirnames, filenames in os.walk(base_path):
                        for file in filenames:
                            if not regexp.search(str(os.path.relpath(
                                    pjoin(root,file),base_path))) is None and \
                                              not path.islink(pjoin(root,file)):
                                new_target = os.path.relpath(
                                                   pjoin(root,file),files_path)
                                if found.group('veto')=='-':
                                    veto_rules.append(new_target)
                                else:
                                    filesToCheck.append(new_target)
                else:
                    fn = fname[1:] if fname.startswith('-') else fname
                    if (not path.exists(pjoin(files_path,fn))) or path.islink(pjoin(files_path,fn)):
                        if force in [0,1]:
                            answer = Cmd.timed_input(question=
"""The IOTest %s does not create file '%s'. Fix it! [type 'enter'] >"""\
                                    %(test_name,fn),default="y")
                        modifications['missing'].append(
                        "%s/%s/%s"%(folder_name,test_name,path.basename(fname)))
                        if verbose: print "    > [ %s ] "%(colored%(31,"MISSING"))+\
                          "%s/%s/%s"%(folder_name,test_name,path.basename(fname))
                    else:
                        if fname.startswith('-'):
                            veto_rules.append(fn)
                        else:
                            filesToCheck.append(fn)

            # Apply the trimming of the veto rules
            filesToCheck = [f for f in filesToCheck if f not in veto_rules]

            if update:
                # Remove files which are no longer used for comparison
                activeFiles = [self.toFileName(f) for f in filesToCheck]
                for file in glob.glob(pjoin(_hc_comparison_files,folder_name,\
                                                                test_name,'*')):
                    # Ignore the .BackUp files and directories. Also ignore
                    # a file which was previously flagged missing because it
                    # was explicitly specified in the list of files that the
                    # test *must* provide.
                    if path.basename(file).endswith('.BackUp') or \
                       path.isdir(file) or \
                       pjoin(folder_name,test_name,path.basename(file)) in \
                                                       modifications['missing']:
                        continue
                    if path.basename(file) not in activeFiles:
                        if force==0 or (force==1 and \
                         path.basename(file) not in reviewed_file_names.keys()):
                            answer = Cmd.timed_input(question=
"""Obsolete ref. file %s in %s/%s detected, delete it? [y/n] >"""\
                                    %(path.basename(file),folder_name,test_name)
                                                                   ,default="y")
                            reviewed_file_names[path.basename(file)] = answer
                        elif (force==1 and \
                             path.basename(file) in reviewed_file_names.keys()):
                            answer = reviewed_file_names[path.basename(file)]
                        else:
                            answer = 'Y'
                            
                        if answer not in ['Y','y','']:
                            if verbose: 
                                print "    > [ %s ] "%(colored%(31,"IGNORED"))+\
                          "file deletion %s/%s/%s"%(folder_name,test_name,
                                                            path.basename(file))
                            continue

                        os.remove(file)
                        if verbose: print "    > [ %s ] "%(colored%(31,"REMOVED"))+\
                          "%s/%s/%s"%(folder_name,test_name,path.basename(file))
                        modifications['removed'].append(
                                            '/'.join(str(file).split('/')[-3:]))

                    
            # Make sure it is not filtered out by the user-filter
            if self.filesChecked_filter!=['ALL']:
                new_filesToCheck = []
                for file in filesToCheck:
                    # Try if it matches any filter
                    for filter in self.filesChecked_filter:
                        # A regular expression
                        if filter.endswith(']'):
                            split=filter[:-1].split('[')
                            # folder without the final /
                            folder=split[0][:-1]
                            if folder!=path.dirname(pjoin(file)):
                                continue
                            search = re.compile('['.join(split[1:]))
                            if not search.match(path.basename(file)) is None:
                                new_filesToCheck.append(file)
                                break    
                        # Just the exact filename
                        elif filter==file:
                            new_filesToCheck.append(file)
                            break
                filesToCheck = new_filesToCheck
            
            # Now we can scan them and process them one at a time
            # Keep track of the folders and testNames the user did not want to
            # create
            refused_Folders = []
            refused_testNames = []
            for fname in filesToCheck:
                file_path = path.abspath(pjoin(files_path,fname))
                self.assertTrue(path.isfile(file_path),
                                            'File %s not found.'%str(file_path))
                comparison_path = pjoin(_hc_comparison_files,\
                                    folder_name,test_name,self.toFileName(fname))
                if not update:
                    if not os.path.isfile(comparison_path):
                        iotest.clean_output()
                        if not verbose:
                            raise MadGraph5Error,\
                                "Missing ref. files for test %s\n"%test_name+\
                                "Create them with './test_manager.py -U %s'"%test_name
                            continue
                        else:
                            print colored%(31,'The ref. file %s'
                            %str('/'.join(comparison_path.split('/')[-3:]))+' does not exist.')
                            print colored%(34,'Consider creating it with '+
                                            './test_manager.py -U %s'%test_name)
                            exit(0)
                    goal = open(comparison_path).read()%misc.get_pkg_info()
                    if not verbose:
                        self.assertFileContains(open(file_path), goal)
                    else:
                        try:
                            self.assertFileContains(open(file_path), goal)
                        except AssertionError:
                            if verbose: 
                                print "    > %s differs from the reference."%fname
                            
                else:                        
                    if not path.isdir(pjoin(_hc_comparison_files,folder_name)):
                        if force==0:
                            if folder_name in refused_Folders:
                                continue
                            answer = Cmd.timed_input(question=
"""New folder %s detected, create it? [y/n] >"""%folder_name
                                                                   ,default="y")
                            if answer not in ['Y','y','']:
                                refused_Folders.append(folder_name)
                                if verbose: print "    > [ %s ] folder %s"\
                                           %(colored%(31,"IGNORED"),folder_name)
                                continue
                        if verbose: print "    > [ %s ] folder %s"%\
                                            (colored%(32,"CREATED"),folder_name)                        
                        os.makedirs(pjoin(_hc_comparison_files,folder_name))
                    if not path.isdir(pjoin(_hc_comparison_files,folder_name,
                                                                    test_name)):
                        if force==0:
                            if (folder_name,test_name) in refused_testNames:
                                continue
                            answer = Cmd.timed_input(question=
"""New test %s/%s detected, create it? [y/n] >"""%(folder_name,test_name)
                                                                   ,default="y")
                            if answer not in ['Y','y','']:
                                refused_testNames.append((folder_name,test_name))
                                if verbose: print "    > [ %s ] test %s/%s"\
                                 %(colored%(31,"IGNORED"),folder_name,test_name)
                                continue
                        if verbose: print "    > [ %s ] test %s/%s"\
                                 %(colored%(32,"CREATED"),folder_name,test_name)
                        os.makedirs(pjoin(_hc_comparison_files,folder_name,
                                                                    test_name))
                    # Transform the package information to make it a template
                    file = open(file_path,'r')
                    target=file.read()
                    # So that if % appear, we cast them to %% which are not formatted.
                    target = target.replace('%','%%')
                    # So that the version and date is automatically updated
                    target = target.replace('MadGraph5_aMC@NLO v. %(version)s, %(date)s'\
                                                           %misc.get_pkg_info(),
                                          'MadGraph5_aMC@NLO v. %(version)s, %(date)s')
                    target = target.replace('v%(version)s (%(date)s)'\
                                                           %misc.get_pkg_info(),
                                                      'v%(version)s (%(date)s)')
                    file.close()
                    if os.path.isfile(comparison_path):
                        file = open(comparison_path,'r')
                        existing = file.read()
                        file.close()
                        if existing == target:
                            continue
                        else:
                            # Copying the existing reference as a backup
                            tmp_path = pjoin(_hc_comparison_files,folder_name,\
                                        test_name,self.toFileName(fname)+'.BackUp')
                            if os.path.isfile(tmp_path):
                                os.remove(tmp_path)
                            file = open(tmp_path,'w')
                            file.write(target)
                            file.close()
                            if force==0 or (force==1 and path.basename(\
                            comparison_path) not in reviewed_file_names.keys()):
                                text = \
"""File %s in test %s/%s differs by the following (reference file first):
"""%(fname,folder_name,test_name)
                                text += misc.Popen(['diff',str(comparison_path),
                                  str(tmp_path)],stdout=subprocess.PIPE).\
                                                                communicate()[0]
                                # Remove the last newline
                                if text[-1]=='\n':
                                    text=text[:-1]
                                if (len(text.split('\n'))<15):
                                    print text
                                else:
                                    pydoc.pager(text)
                                    print "Difference displayed in editor."
                                answer = ''
                                while answer not in ['y', 'n']:
                                    answer = Cmd.timed_input(question=
"""Ref. file %s differs from the new one (see diff. before), update it? [y/n/h/r] >"""%fname
                                                                   ,default="y")
                                    if answer not in ['y','n']:
                                        if answer == 'r':
                                            pydoc.pager(text)
                                        else:
                                            print "reference path: %s" % comparison_path
                                            print "code returns: %s" % tmp_path
                                
                                
                                os.remove(tmp_path)
                                reviewed_file_names[path.basename(\
                                                      comparison_path)] = answer        
                            elif (force==1 and path.basename(\
                                comparison_path) in reviewed_file_names.keys()):
                                answer = reviewed_file_names[path.basename(\
                                                               comparison_path)]
                            else:
                                answer = 'Y'
                            if answer not in ['Y','y','']:
                                if verbose: print "    > [ %s ] %s"%\
                                                  (colored%(31,"IGNORED"),fname)
                                continue
                            
                            # Copying the existing reference as a backup
                            back_up_path = pjoin(_hc_comparison_files,folder_name,\
                                         test_name,self.toFileName(fname)+'.BackUp')
                            if os.path.isfile(back_up_path):
                                os.remove(back_up_path)
                            cp(comparison_path,back_up_path)
                            if verbose: print "    > [ %s ] %s"\
                                                 %(colored%(32,"UPDATED"),fname)
                            modifications['updated'].append(
                                      '/'.join(comparison_path.split('/')[-3:]))
                    else:
                        if force==0 or (force==1 and path.basename(\
                            comparison_path) not in reviewed_file_names.keys()):
                            answer = Cmd.timed_input(question=
"""New file %s detected, create it? [y/n] >"""%fname
                                                                   ,default="y")
                            reviewed_file_names[path.basename(\
                                                      comparison_path)] = answer
                        elif (force==1 and path.basename(\
                                comparison_path) in reviewed_file_names.keys()):
                            answer = reviewed_file_names[\
                                                 path.basename(comparison_path)]
                        else:
                            answer = 'Y'
                        if answer not in ['Y','y','']:
                            if verbose: print "    > [ %s ] %s"%\
                                                  (colored%(31,"IGNORED"),fname)
                            continue
                        if verbose: print "    > [ %s ] %s"%\
                                                  (colored%(32,"CREATED"),fname)
                        modifications['created'].append(
                                      '/'.join(comparison_path.split('/')[-3:]))
                    file = open(comparison_path,'w')
                    file.write(target)
                    file.close()

            # Clean the iotest output
            iotest.clean_output()

        # Monitor the modifications when in creation files mode by returning the
        # modifications dictionary.
        if update:
            return modifications
        else:
            return 'test_over'
 
