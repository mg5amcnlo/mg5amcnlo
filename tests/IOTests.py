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

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(root_path)

import madgraph.various.misc as misc

from madgraph.interface.extended_cmd import Cmd

from madgraph.iolibs.files import cp, ln, mv
from madgraph import MadGraph5Error

pjoin = os.path.join
path = os.path

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = path.abspath(os.path.join(_file_path,'input_files'))
_hc_comparison_files = pjoin(_input_file_path,'IOTestsComparison')
_hc_comparison_tarball = pjoin(_input_file_path,'IOTestsComparison.tar.bz2')

class IOTest(object):
    """ IOTest runner and attribute container. It can be overloaded depending on
    what kind of IO test will be necessary later """

    # Handy definitions
    proc_files = ['[^.+\.(f|dat|inc)$]']
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

    def __init__(self, hel_amp=None,
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
        self.hel_amp = hel_amp 
        self.helasModel = helasModel
        self.exporter = exporter
        # Security mesure
        if not str(path.dirname(_file_path)) in str(outputPath) and \
                                        not str(outputPath).startswith('/tmp/'):
            raise MadGraph5Error, "OutputPath must be within MG directory or"+\
                                                                     " in /tmp/"            
        else:
            self.outputPath = outputPath
    
    def run(self):
        """ Run the test and returns the path where the files have been 
        produced and relative to which the paths in TestedFiles are specified. """
        
        self.clean_output()
        model = self.hel_amp.get('processes')[0].get('model')
        self.exporter.copy_v4template(model.get('name'))
        self.exporter.generate_loop_subprocess(self.hel_amp, self.helasModel)
        wanted_lorentz = self.hel_amp.get_used_lorentz()
        wanted_couplings = list(set(sum(self.hel_amp.get_used_couplings(),[])))
        self.exporter.convert_model_to_mg4(model,wanted_lorentz,wanted_couplings)
            
        proc_name='P'+self.hel_amp.get('processes')[0].shell_string()
        return pjoin(self.outputPath,'SubProcesses',proc_name)
    
    def clean_output(self):
        """ Remove the output_path if existing. Careful!"""
        if not str(path.dirname(_file_path)) in str(self.outputPath) and \
                                   not str(self.outputPath).startswith('/tmp/'):
            raise MadGraph5Error, "Cannot safely remove %s."%str(self.outputPath)
        else:
            if path.isdir(self.outputPath):
                shutil.rmtree(self.outputPath)

#===============================================================================
# IOTestManager
#===============================================================================
class IOTestManager(unittest.TestCase):
    """ A helper class to perform tests based on the comparison of files output 
    by exporters against hardcoded ones. """
    
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
    
    def test_IOTests(self):
        """ A test function in the mother so that all childs automatically run
        their tests when scanned with the test_manager. """
        
        # Set it to True if you want info during the regular test_manager.py runs
        self.runIOTests(verbose=False)
    
    def addIOTest(self, folderName, testName, IOtest):
        """ Add the test (folderName, testName) to class attribute all_tests. """
        
        if not self.need(testName=testName, folderName=folderName):
            return

        # Add this to the instance test_list
        if (folderName, testName) not in self.instance_tests:
            self.instance_tests.append((folderName, testName))
            
        # Add this to the global test_list            
        if (folderName, testName) in self.all_tests.keys() and \
                                  self.all_tests[(folderName, testName)]!=IOtest:
            raise MadGraph5Error, \
                          "Test (%s,%s) already defined."%(folderName, testName)
        else:
            self.all_tests[(folderName, testName)] = IOtest

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

        # In update = True mode, we keep track of the modification to 
        # provide summary information
        modifications={'updated':[],'created':[], 'removed':[]}
        
        # List all the names of the files for which modifications have been
        # reviewed at least once.The approach taken here is different than
        # with the list refusedFolder and refusedTest.
        # The key of the dictionary are the filenames and the value are string
        # determining the user answer for that file.
        reviewed_file_names = {}
        
        # Chose what test to cover
        if testKeys == 'instanceList':
            testKeys = self.instance_tests
        
        if verbose: print "\n== Operational mode : file %s ==\n"%\
                                           ('UPDATE' if update else 'TESTING')
        for (folder_name, test_name) in testKeys:
            try:
                iotest=self.all_tests[(folder_name, test_name)]
            except KeyError:
                raise MadGraph5Error, 'Test (%s,%s) could not be found.'\
                                                       %(folder_name, test_name)
            if verbose: print "Processing %s in %s"%(test_name,folder_name)
            files_path = iotest.run()

            # First create the list of files to check as the user might be using
            # regular expressions.
            filesToCheck=[]
            # Store here the files reckognized as veto rules (with filename
            # starting with '-')
            veto_rules = []
            for fname in iotest.testedFiles:
                # Disregard the veto rules
                if fname.endswith(']'):
                    split=fname[:-1].split('[')
                    # folder without the final /
                    folder=split[0][:-1]
                    search = re.compile('['.join(split[1:]))
                    # In filesToCheck, we must remove the files_path/ prepended
                    filesToCheck += [ f[(len(str(files_path))+1):]
                           for f in glob.glob(pjoin(files_path,folder,'*')) if \
                               (not search.match(path.basename(f)) is None and \
                                      not path.isdir(f) and not path.islink(f))]
                elif fname.startswith('-'):
                    veto_rules.append(fname[1:])
                else:
                    filesToCheck.append(fname)
            
            # Apply the trimming of the veto rules
            filesToCheck = [f for f in filesToCheck if f not in veto_rules]
            
            if update:
                # Remove files which are no longer used for comparison
                activeFiles = [self.toFileName(f) for f in filesToCheck]
                for file in glob.glob(pjoin(_hc_comparison_files,folder_name,\
                                                                test_name,'*')):
                    # Ignore the .BackUp files and directories
                    if path.basename(file).endswith('.BackUp') or\
                                                               path.isdir(file):
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
                                print "    > [ IGNORED ] file deletion "+\
                          "%s/%s/%s"%(folder_name,test_name,path.basename(file))
                            continue

                        os.remove(file)
                        if verbose: print "    > [ REMOVED ] %s/%s/%s"\
                                    %(folder_name,test_name,path.basename(file))
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
                        raise MadGraph5Error, 'The file %s'%str(comparison_path)+\
                                                              ' does not exist.'
                    goal = open(comparison_path).read()%misc.get_pkg_info()
                    if not verbose:
                        self.assertFileContains(open(file_path), goal)
                    else:
                        try:
                            self.assertFileContains(open(file_path), goal)
                        except AssertionError:
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
                                if verbose: print "    > [ IGNORED ] folder %s"\
                                                                    %folder_name
                                continue
                        if verbose: print "    > [ CREATED ] folder %s"%folder_name
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
                                if verbose: print "    > [ IGNORED ] test %s/%s"\
                                                        %(folder_name,test_name)
                                continue
                        if verbose: print "    > [ CREATED ] test %s/%s"\
                                                        %(folder_name,test_name)
                        os.makedirs(pjoin(_hc_comparison_files,folder_name,
                                                                    test_name))
                    # Transform the package information to make it a template
                    file = open(file_path,'r')
                    target=file.read()
                    target = target.replace('MadGraph 5 v. %(version)s, %(date)s'\
                                                           %misc.get_pkg_info(),
                                          'MadGraph 5 v. %(version)s, %(date)s')
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
                                        test_name,self.toFileName(fname)+'.tmp')
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
                                answer = Cmd.timed_input(question=
"""Ref. file %s differs from the new one (see diff. before), update it? [y/n] >"""%fname
                                                                   ,default="y")
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
                                if verbose: print "    > [ IGNORED ] %s"%fname
                                continue
                            
                            # Copying the existing reference as a backup
                            back_up_path = pjoin(_hc_comparison_files,folder_name,\
                                         test_name,self.toFileName(fname)+'.BackUp')
                            if os.path.isfile(back_up_path):
                                os.remove(back_up_path)
                            cp(comparison_path,back_up_path)
                            if verbose: print "    > [ UPDATED ] %s"%fname
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
                            if verbose: print "    > [ IGNORED ] %s"%fname
                            continue
                        if verbose: print "    > [ CREATED ] %s"%fname
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
 
